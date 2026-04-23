import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import typing as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from .utils import get_logger
from .featurizers import Featurizer
from torch_geometric.data import Data, Dataset

logg = get_logger()

class ComplexData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'd_edge_index':
            return self.d_x.size(0)
        if key == 'p_edge_index':
            return self.p_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class InMemoryListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
        
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def process_single_sample(row, d_struc_feat, d_seq_feat, t_struc_feat, t_seq_feat):
    smi = str(row['Drug'])
    t_key = str(row['target_key'])
    t_seq_str = row['Target']
    label = float(row['Y'])

    d_seq = d_seq_feat(smi) # Tensor
    # d_struc (PyG Data)
    d_graph = d_struc_feat.get_graph_data(smi)
    
    t_seq = t_seq_feat(t_seq_str) # Tensor
    # t_struc (PyG Data)
    t_graph = t_struc_feat.get_protein_data(t_key)

    data = ComplexData(
        # Drug Graph
        d_x=d_graph.x,
        d_edge_index=d_graph.edge_index,
        d_edge_attr=d_graph.edge_attr,
        d_seq=d_seq,
        
        # Protein Graph
        p_x=t_graph.x,
        p_edge_index=t_graph.edge_index,
        p_edge_attr=t_graph.edge_attr, 
        p_seq=t_seq,

        p_pkt_mask=getattr(t_graph, 'pkt_mask', None),
        
        # Label
        y=torch.tensor([label], dtype=torch.float)
    )
    return data

def get_task_dir(task_name: str):
    task_paths = {
        'davis': './dataset/Davis',
        'kiba': './dataset/KIBA',
    }
    return Path(task_paths[task_name.lower()]).resolve()

def create_fold_setting_cold(df, fold_seed, frac, entities):
    if isinstance(entities, str):
        entities = [entities]
        

    train_frac, val_frac, cal_frac, test_frac = frac
    
    test_entity_instances = [df[e].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values for e in entities]
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0: raise ValueError("No test samples found.")
    

    remain_df1 = df.copy()
    for i, e in enumerate(entities):
        remain_df1 = remain_df1[~remain_df1[e].isin(test_entity_instances[i])]

    cal_target_frac = cal_frac / (1.0 - test_frac)
    cal_entity_instances = [remain_df1[e].drop_duplicates().sample(frac=cal_target_frac, replace=False, random_state=fold_seed).values for e in entities]
    cal = remain_df1.copy()
    for entity, instances in zip(entities, cal_entity_instances):
        cal = cal[cal[entity].isin(instances)]
        

    remain_df2 = remain_df1.copy()
    for i, e in enumerate(entities):
        remain_df2 = remain_df2[~remain_df2[e].isin(cal_entity_instances[i])]


    val_target_frac = val_frac / (1.0 - test_frac - cal_frac)
    val_entity_instances = [remain_df2[e].drop_duplicates().sample(frac=val_target_frac, replace=False, random_state=fold_seed).values for e in entities]
    val = remain_df2.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0: raise ValueError("No validation samples found.")

    train = remain_df2.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return train.reset_index(drop=True), val.reset_index(drop=True), cal.reset_index(drop=True), test.reset_index(drop=True)


class DTADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str,
        drug_seq_featurizer: Featurizer,
        drug_struc_featurizer: Featurizer,   
        target_seq_featurizer: Featurizer,
        target_struc_featurizer: Featurizer, 
        dataset_name: str = "Davis",
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        use_cold_spilt: bool = False,
        use_test: bool = False,
        cold: str = 'Drug',
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0, index_col=0, sep=",",
        **kwargs
        ):
        
        super().__init__()
        self.dataset_name = dataset_name
        self.use_cold_spilt = use_cold_spilt
        self.cold = cold
        self.use_test = use_test
        self._device = device
        self._seed = seed
        self._data_dir = Path(data_dir)
        self._train_path = Path("process.csv")
        self._csv_kwargs = {"header": header, "index_col": index_col, "sep": sep}

        self._loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "follow_batch": ['d_x', 'p_x']
        }
        
        self._shuffle = shuffle

        self.drug_seq_featurizer = drug_seq_featurizer
        self.drug_struc_featurizer = drug_struc_featurizer
        self.target_seq_featurizer = target_seq_featurizer
        self.target_struc_featurizer = target_struc_featurizer

        # 添加了 cal_data_list 容器
        self.train_data_list = []
        self.val_data_list = []
        self.cal_data_list = []
        self.test_data_list = []

    def prepare_data(self):
        if not self.drug_struc_featurizer.path.exists():
            logg.warning(f" Drug H5 missing.")
        if not self.target_struc_featurizer.path.exists():
            logg.warning(f" Protein H5 missing.")
        self._dataframes = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)

    def setup(self, stage: T.Optional[str] = None):
        self._dataframes = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)
        
        # Preload Logic (H5 -> RAM)
        print(" Preloading H5 Features into Memory...")
        all_drugs = self._dataframes['Drug'].unique()
        all_targets = self._dataframes['Target'].unique()
        all_target_keys = self._dataframes['target_key'].unique()
        
        self.drug_seq_featurizer.preload(all_drugs)
        self.drug_struc_featurizer.preload(all_drugs) 
        self.target_seq_featurizer.preload(all_targets)
        self.target_struc_featurizer.preload(all_target_keys) 
        print(" All H5 features loaded.")

        if self.use_cold_spilt:
            df_train, df_val, df_cal, df_test = create_fold_setting_cold(
                self._dataframes, self._seed, [0.7, 0.1, 0.1, 0.1], self.cold
            )
        elif self.use_test:
            df_train = df_val = df_cal = df_test = self._dataframes
        else:
            temp_train, temp = train_test_split(self._dataframes, test_size=0.3, random_state=self._seed)
            df_train = temp_train
            df_val, temp2 = train_test_split(temp, test_size=2/3, random_state=self._seed)
            df_cal, df_test = train_test_split(temp2, test_size=0.5, random_state=self._seed)

        def _df_to_list(df, desc):
            data_list = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
                data = process_single_sample(
                    row, 
                    self.drug_struc_featurizer, 
                    self.drug_seq_featurizer, 
                    self.target_struc_featurizer, 
                    self.target_seq_featurizer
                )
                data_list.append(data)
            return data_list

        self.train_data_list = _df_to_list(df_train, "Train Data")
        self.val_data_list = _df_to_list(df_val, "Val Data")
        self.cal_data_list = _df_to_list(df_cal, "Cal Data") 
        self.test_data_list = _df_to_list(df_test, "Test Data")
        
        print(f"Cache Complete. Train: {len(self.train_data_list)}, Val: {len(self.val_data_list)}, Cal: {len(self.cal_data_list)}, Test: {len(self.test_data_list)}")

    def train_dataloader(self, seed=None, domain=None):
        dataset = InMemoryListDataset(self.train_data_list)
        return DataLoader(dataset, shuffle=self._shuffle, **self._loader_kwargs), len(dataset) // self._loader_kwargs['batch_size']

    def val_dataloader(self, domain=None):
        dataset = InMemoryListDataset(self.val_data_list)
        return DataLoader(dataset, shuffle=False, **self._loader_kwargs), len(dataset) // self._loader_kwargs['batch_size']

    def cal_dataloader(self, domain=None):
        dataset = InMemoryListDataset(self.cal_data_list)
        return DataLoader(dataset, shuffle=False, **self._loader_kwargs), len(dataset) // self._loader_kwargs['batch_size']

    def test_dataloader(self, domain=None):
        dataset = InMemoryListDataset(self.test_data_list)
        return DataLoader(dataset, shuffle=False, **self._loader_kwargs), len(dataset) // self._loader_kwargs['batch_size']