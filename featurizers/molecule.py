import os
import copy
import pickle
import math
import torch
import torch.nn as nn
from rdkit.Chem import AllChem
import numpy as np



class unimolFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 512, 
        save_dir: Path = Path().absolute(),
    ):
        super().__init__("unimol", shape, save_dir)
        if not self.path.exists():

    def _transform(self, seq: str) -> torch.Tensor:

        return torch.zeros(self.shape, dtype=torch.float32)

    def write_to_disk(self, seq_list, verbose=True) -> None:
        raise NotImplementedError()

    def preload(
        self,
        seq_list: T.List[str],
        verbose: bool = True,
        write_first: bool = False,
    ) -> None:


        if not self.path.exists():
            return

        try:
            with h5py.File(self.path, "r") as h5fi:
                if 'features' not in h5fi or 'smiles' not in h5fi:
                    return


                all_feats = h5fi['features'][:] # [N, 512]
                all_smiles = h5fi['smiles'][:]  # [N] (bytes)


                decoded_smiles = [s.decode('utf-8') if isinstance(s, bytes) else s for s in all_smiles]
                smiles_map = {s: i for i, s in enumerate(decoded_smiles)}

                found_count = 0
                
                for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                    seq_key = seq.strip()
                    
                    if seq_key in smiles_map:
                        idx = smiles_map[seq_key]
                        feats = torch.from_numpy(all_feats[idx]).float()
                        if self._on_cuda:
                            feats = feats.to(self.device)
                        self._features[seq] = feats
                        found_count += 1
                    else:
                        feats = torch.zeros(self.shape, dtype=torch.float32)
                        if self._on_cuda:
                            feats = feats.to(self.device)
                        self._features[seq] = feats

        except Exception as e:
            
        self._update_device(self.device)
        self._preloaded = True

class MolGraphFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        # 78 = Atom Features
        super().__init__("MolGraphSparse", 78, save_dir)
        
        self.h5_path = save_dir / "molgraphnorbf_features.h5"
        self.smiles_map = {}
        

        self.cache_x = None         
        self.cache_edge_index = None 
        self.cache_edge_attr = None  
        
        self.ptr_node = None         
        self.ptr_edge = None         

    def preload(self, seq_list: T.List[str], verbose: bool = True):

        if not self.h5_path.exists():
            logg.error(f" H5 file not found: {self.h5_path}")
            logg.error("Please run 'gen_mol_sparse.py' first.")
            raise FileNotFoundError(f"{self.h5_path} missing")

        logg.info(f"[{self.name}] Loading features from {self.h5_path}...")
        
        try:
            with h5py.File(self.h5_path, 'r') as f:

                raw_smiles = f['smiles'][:]
                decoded_smiles = [s.decode('utf-8') if isinstance(s, bytes) else s for s in raw_smiles]
                self.smiles_map = {s: i for i, s in enumerate(decoded_smiles)}
                
                self.cache_x = torch.from_numpy(f['x'][:]).float()
                self.cache_edge_index = torch.from_numpy(f['edge_index'][:]).long()
                self.cache_edge_attr = torch.from_numpy(f['edge_attr'][:]).float()
                
                self.ptr_node = torch.from_numpy(f['ptr_node'][:]).long()
                self.ptr_edge = torch.from_numpy(f['ptr_edge'][:]).long()

            logg.info(f"[{self.name}]  Loaded {len(self.smiles_map)} molecules into memory.")
            self._preloaded = True
            
        except Exception as e:
            logg.error(f"Error loading Mol H5: {e}")
            raise e

    def _transform(self, seq: str):
        return torch.zeros(1)

    def get_graph_data(self, smiles: str):
        if not self._preloaded:
            raise RuntimeError("Please call preload() first!")

        idx = self.smiles_map.get(str(smiles).strip())
        
        if idx is None:
            return Data(
                x=torch.zeros(1, 78), 
                edge_index=torch.zeros(2, 0).long(), 
                edge_attr=torch.zeros(0, 6).float()
            )
            
        node_start = self.ptr_node[idx]
        node_end = self.ptr_node[idx+1]
        
        edge_start = self.ptr_edge[idx]
        edge_end = self.ptr_edge[idx+1]
        
        x = self.cache_x[node_start : node_end]
        edge_attr = self.cache_edge_attr[edge_start : edge_end]
        
        edge_index = self.cache_edge_index[:, edge_start : edge_end]
        edge_index = edge_index - node_start 

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)