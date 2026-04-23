import sys
import pickle
import os
import torch
import hashlib
import pickle as pk
import typing as T
from pathlib import Path
from .base import Featurizer
from ..utils import get_logger
import h5py
import numpy as np
import pandas as pd
from torch_geometric.data import Data
logg = get_logger()
MODEL_CACHE_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
FOLDSEEK_MISSING_IDX = 20
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


class ESMFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        super().__init__("ESM", 1280, save_dir)

        import esm


        torch.hub.set_dir(r'/home/.cache/torch/hub/checkpoints')    
        last_part = os.path.basename(save_dir)


        if last_part == 'KIBA':
            self._max_len = 1310
        else:
            self._max_len = 1210         


        model_path = r'/home/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt' 
        self._esm_model, self._esm_alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)

        self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
        self._register_cuda("model", self._esm_model)

    def _transform(self, seq: str): 
        seq = seq.upper() 
        if len(seq) > self._max_len - 2: 
            seq = seq[: self._max_len - 2] 
  
        batch_labels, batch_strs, batch_tokens = self._esm_batch_converter(
            [("sequence", seq)]
        )
        batch_tokens = batch_tokens.to(self.device)

        results = self._cuda_registry["model"][0](
            batch_tokens, repr_layers=[33], return_contacts=True
        )
        token_representations = results["representations"][33]
        tokens = token_representations[0, 1 : len(seq) + 1]
        return tokens.mean(0)



class ProteinGNNFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):

        super().__init__("ProteinTEGNN", 61, save_dir)
        
        self.h5_path = save_dir / "pocket3graph_features.h5"
        self.key_map = {}
        

        self.cache_x = None          # [Total_Nodes, 49]
        self.cache_edge_index = None # [2, Total_Edges]
        self.cache_edge_attr = None  
        self.cache_pkt_mask = None   # [Total_Nodes, 3]
        
        self.ptr_node = None         # [Num_Prots + 1]
        self.ptr_edge = None         # [Num_Prots + 1]

    def preload(self, seq_list: T.List[str], verbose: bool = True):
        if not self.h5_path.exists():
            logg.error(f"H5 file not found: {self.h5_path}")
            raise FileNotFoundError(f"{self.h5_path} missing")

        logg.info(f"[{self.name}] Loading ALL features into RAM from {self.h5_path}...")
        
        try:
            with h5py.File(self.h5_path, 'r') as f:
                raw_keys = f['target_keys'][:]
                decoded_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in raw_keys]
                self.key_map = {k: i for i, k in enumerate(decoded_keys)}
                
                self.cache_x = torch.from_numpy(f['x'][:]).float()
                self.cache_edge_index = torch.from_numpy(f['edge_index'][:]).long()
                self.cache_edge_attr = torch.from_numpy(f['edge_attr'][:]).float()
                self.cache_pkt_mask = torch.from_numpy(f['pkt_mask'][:]).float()
                
                self.ptr_node = torch.from_numpy(f['ptr_node'][:]).long()
                self.ptr_edge = torch.from_numpy(f['ptr_edge'][:]).long()

            logg.info(f"[{self.name}]  Loaded {len(self.key_map)} proteins into RAM.")
            self._preloaded = True
            
        except Exception as e:
            logg.error(f"Error loading Protein H5: {e}")
            raise e

    def _transform(self, seq: str):
        return torch.zeros(1)

    def get_protein_data(self, target_key: str):
        if not self._preloaded:
            raise RuntimeError("Please call preload() first!")

        idx = self.key_map.get(str(target_key))
        if idx is None:

            return Data(
                x=torch.zeros(1, 49), 
                edge_index=torch.zeros(2, 0).long(), 
                edge_attr=torch.zeros(0, 16).float(), 
                pkt_mask=torch.zeros(1, 3)
            )
        
        node_start = self.ptr_node[idx]
        node_end = self.ptr_node[idx+1]
        
        edge_start = self.ptr_edge[idx]
        edge_end = self.ptr_edge[idx+1]
        
        x = self.cache_x[node_start : node_end]       # [N, 49]
        pkt_mask = self.cache_pkt_mask[node_start : node_end] # [N, 3] 
        
        edge_index = self.cache_edge_index[:, edge_start : edge_end] # [2, E]
        
        edge_attr = self.cache_edge_attr[edge_start : edge_end]      # [E, 16]
        

        edge_index = edge_index - node_start

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pkt_mask=pkt_mask)

