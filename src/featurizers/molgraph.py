import os
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")

CONFIG = {
    "Davis": {
        "csv_path": "/home/RCP-DTA/dataset/Davis/process.csv",
        "output_name": "molgraphnorbf_features.h5" 
    },
    "KIBA": {
        "csv_path": "/home/RCP-DTA/dataset/KIBA/process.csv",
        "output_name": "molgraphnorbf_features.h5"
    }
}

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding(atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
         'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
         'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
         'Pt', 'Hg', 'Pb', 'Unknown']) +
    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
    [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ], dtype=np.float32)

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None, None, None
    

    
    features = []
    for atom in mol.GetAtoms():
        features.append(atom_features(atom))
    
    x = np.array(features, dtype=np.float32)
    
    rows, cols = [], []
    edge_attrs_list = []
    
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        b_feat = bond_features(bond)
        
        
        rows.extend([start, end])
        cols.extend([end, start])
        

        edge_attrs_list.append(b_feat)
        edge_attrs_list.append(b_feat)
        
    if len(rows) == 0:
        return x, np.zeros((2, 0), dtype=np.longlong), np.zeros((0, 6), dtype=np.float32)

    edge_index = np.array([rows, cols], dtype=np.longlong)
    
    edge_attr = np.stack(edge_attrs_list, axis=0)
    
    return x, edge_index, edge_attr

class MolGraphH5Generator:
    def __init__(self, task):
        self.cfg = CONFIG[task]
        self.csv_path = self.cfg["csv_path"]
        self.save_dir = os.path.dirname(self.csv_path)
        self.save_path = os.path.join(self.save_dir, self.cfg["output_name"])
        
    def process(self):
        df = pd.read_csv(self.csv_path)
        if 'Drug' in df.columns:
            smiles_col = 'Drug'
        elif 'ligand' in df.columns:
            smiles_col = 'ligand'
        else:
            raise KeyError("CSV missing 'Drug' or 'ligand' column")

        unique_smiles = df[smiles_col].unique().tolist()
        
        all_x = []
        all_edge_index = []
        all_edge_attr = []
        
        ptr_node = [0]
        ptr_edge = [0]
        valid_smiles = []
        
        count = 0
        
        for smi in tqdm(unique_smiles, desc="Processing Molecules"):
            try:
                x, edge_index, edge_attr = smile_to_graph(smi)
                
                if x is None: continue
                
                num_nodes = x.shape[0]
                num_edges = edge_index.shape[1]
                
                current_offset = ptr_node[-1]
                global_edge_index = edge_index + current_offset

                all_x.append(x)
                all_edge_index.append(global_edge_index)
                all_edge_attr.append(edge_attr)
                valid_smiles.append(smi)
                
                ptr_node.append(ptr_node[-1] + num_nodes)
                ptr_edge.append(ptr_edge[-1] + num_edges)
                
                count += 1
                
            except Exception as e:
                # print(f"Error processing {smi}: {e}")
                pass
        
        if count == 0: return

        X_all = np.concatenate(all_x, axis=0) 
        E_all = np.concatenate(all_edge_index, axis=1) 
        EA_all = np.concatenate(all_edge_attr, axis=0)
        
        Ptr_node = np.array(ptr_node, dtype=np.longlong)
        Ptr_edge = np.array(ptr_edge, dtype=np.longlong)
        S = np.array([s.encode('utf-8') for s in valid_smiles], dtype="S")

        print(f" Processed {count} molecules.")
        print(f"   Node Dim: {X_all.shape[1]}")
        print(f"   Edge Dim: {EA_all.shape[1]} (Expected 6: Bond Types only)")
        
        with h5py.File(self.save_path, 'w') as f:
            f.create_dataset("x", data=X_all, compression="gzip")
            f.create_dataset("edge_index", data=E_all, compression="gzip")
            f.create_dataset("edge_attr", data=EA_all, compression="gzip")
            f.create_dataset("ptr_node", data=Ptr_node, compression="gzip")
            f.create_dataset("ptr_edge", data=Ptr_edge, compression="gzip")
            f.create_dataset("smiles", data=S, compression="gzip")
            
        print(f"H5 File Saved to: {self.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["Davis", "KIBA"])
    args = parser.parse_args()
    
    MolGraphH5Generator(args.task).process()
