import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from Bio import SeqIO
from scipy.spatial import distance_matrix
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")

CONFIG = {
  "Davis": {
    "csv_path": "/home/RCP-DTA/dataset/Davis/process.csv",
    "pdb_dir": "/home/dataprocessing/DavisPDB",
    "p2rank_dir": "/home/RCP-DTA/dataset/Davis/p2rank_results",
    "pssm_dir": "/home/RCP-DTA/dataset/Davis/davis_pssm_results",
    "fasta_path": "/home/RCP-DTA/dataset/Davis/davis_unique_proteins.fasta",
    "max_pockets": 3,
    "output_name": "pocket3graph_features.h5"
  },
  "KIBA": {
    "csv_path": "/home/RCP-DTA/dataset/KIBA/process.csv",
    "pdb_dir": "/home/dataprocessing/KIBAPDB",
    "p2rank_dir": "/home/RCP-DTA/dataset/KIBA/p2rank_results",
    "pssm_dir": "/home/RCP-DTA/dataset/KIBA/kiba_pssm_results",
    "fasta_path": "/home/RCP-DTA/dataset/KIBA/kiba_unique_proteins.fasta",
    "max_pockets": 3,
    "output_name": "pocket3graph_features.h5"
  }
}

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
THREE_TO_ONE = {
  'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
  'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
  'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


class RBFExpansion:
  def __init__(self, start=2.0, end=8.0, bins=16, gamma=16.0):
    self.centers = np.linspace(start, end, bins)
    self.gamma = gamma

  def __call__(self, dist):
    return np.exp(-self.gamma * (dist[:, None] - self.centers[None, :])**2)

rbf_expand = RBFExpansion(start=2.0, end=8.0, bins=16)

def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set: x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))

def get_sinusoidal_pos_encoding(seq_len, d_model=16):
  pe = np.zeros((seq_len, d_model), dtype=np.float32)
  position = np.arange(0, seq_len)[:, np.newaxis]
  div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
  pe[:, 0::2] = np.sin(position * div_term)
  pe[:, 1::2] = np.cos(position * div_term)
  return pe

def get_residue_features_basic(residue_1let, pe_vector):
  residue = residue_1let.upper()
  if residue not in pro_res_table: residue = 'X'
 
  # One-hot
  one_hot = np.array(one_of_k_encoding(residue, pro_res_table), dtype=np.float32)
 
  return np.concatenate([one_hot, pe_vector])

def _normalize(tensor, dim=-1):
  return torch.nn.functional.normalize(tensor, p=2, dim=dim, eps=1e-7)

def get_dihedrals_torch(coords_n, coords_ca, coords_c):
  """
   PyTorch  (Phi, Psi)
  : Numpy arrays [L, 3]
  : Numpy array [L, 4] -> [sin(phi), cos(phi), sin(psi), cos(psi)]
  """

  n = torch.from_numpy(coords_n)
  ca = torch.from_numpy(coords_ca)
  c = torch.from_numpy(coords_c)
  X = torch.stack([n, ca, c], dim=1).float()
 
  X_flat = torch.reshape(X, [3 * X.shape[0], 3])
 
  dX = X_flat[1:] - X_flat[:-1] # [3L-1, 3]
  U = _normalize(dX, dim=-1)
 
  u_2 = U[:-2]
  u_1 = U[1:-1]
  u_0 = U[2:]

  n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
  n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

  cosD = torch.sum(n_2 * n_1, -1)
  cosD = torch.clamp(cosD, -1 + 1e-7, 1 - 1e-7)
  D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

  D = F.pad(D, [1, 2])
  D = torch.reshape(D, [-1, 3])
 
  D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
 
  cos_phi = D_features[:, 0:1]
  cos_psi = D_features[:, 1:2]
  sin_phi = D_features[:, 3:4]
  sin_psi = D_features[:, 4:5]
 
  return torch.cat([sin_phi, cos_phi, sin_psi, cos_psi], dim=1).numpy()

def read_pssm(pssm_file):

  try:
    with open(pssm_file, 'r') as f:
      lines = f.readlines()
   
    pssm_features = []
    for line in lines:
      parts = line.split()
      if len(parts) > 0 and parts[0].isdigit():
        if len(parts) < 22: continue
        try:
          scores = [float(x) for x in parts[2:22]]
          if len(scores) == 20:
            pssm_features.append(scores)
        except ValueError:
          continue

    if len(pssm_features) == 0: return None
   
    pssm_array = np.array(pssm_features, dtype=np.float32)
    pssm_norm = 1.0 / (1.0 + np.exp(-pssm_array)) # Sigmoid 
    return pssm_norm
   
  except Exception as e:
    return None

def load_pssm_mapping(fasta_path):
  print(f"Loading PSSM mapping from {fasta_path}...")
  seq_map = {}
  try:
    for record in SeqIO.parse(fasta_path, "fasta"):
      seq_str = str(record.seq)
      seq_map[seq_str] = record.id
    print(f"Loaded {len(seq_map)} sequences for PSSM mapping.")
  except Exception as e:
    print(f"Error loading FASTA: {e}")
  return seq_map

def parse_protein_data(target_key, pdb_dir, p2rank_dir, max_pockets):
  pdb_filename = f"AF-{target_key}-F1-model_v6.pdb"
  csv_filename = f"AF-{target_key}-F1-model_v6.pdb_residues.csv"
 
  pdb_path = os.path.join(pdb_dir, pdb_filename)
  csv_path = os.path.join(p2rank_dir, csv_filename)
 
  if not os.path.exists(pdb_path):
    return None, None, None, None, None

  parser = PDBParser(QUIET=True)
  try:
    structure = parser.get_structure(target_key, pdb_path)
    model = structure[0]
  except:
    return None, None, None, None, None

  residues = []
  coords_ca = []
  coords_n = []
  coords_c = []
 
  res_id_map = {}
 
  idx = 0
  for chain in model:
    for res in chain:
      if res.id[0] != ' ': continue
      if not ('CA' in res and 'N' in res and 'C' in res):
        continue
     
      res_name_3 = res.get_resname()
      aa = THREE_TO_ONE.get(res_name_3, 'X')
     
      residues.append(aa)
      coords_ca.append(res['CA'].get_coord())
      coords_n.append(res['N'].get_coord())
      coords_c.append(res['C'].get_coord())
     
      chain_id = chain.get_id()
      res_num = res.id[1]
      key_id = f"{chain_id}_{res_num}"
      res_id_map[key_id] = idx
     
      idx += 1
     
  if len(residues) < 5: return None, None, None, None, None

  pocket_lists = [[] for _ in range(max_pockets)]
  if os.path.exists(csv_path):
    try:
      df = pd.read_csv(csv_path, skipinitialspace=True)
      if 'pocket' in df.columns and 'residue_label' in df.columns:
        for _, row in df.iterrows():
          p_id = int(row['pocket'])
          if p_id > 0 and p_id <= max_pockets:
            r_chain = str(row['chain']).strip()
            r_num = int(row['residue_label'])
            key_id = f"{r_chain}_{r_num}"
            if key_id in res_id_map:
              pocket_lists[p_id - 1].append(res_id_map[key_id])
    except Exception:
      pass
     
  return residues, np.array(coords_ca), np.array(coords_n), np.array(coords_c), pocket_lists


class ProteinH5Generator:
  def __init__(self, task):
    self.cfg = CONFIG[task]
    self.save_path = os.path.join(os.path.dirname(self.cfg['csv_path']), self.cfg['output_name'])
   
    if os.path.exists(self.cfg['fasta_path']):
      self.seq_to_pssm_id = load_pssm_mapping(self.cfg['fasta_path'])
    else:
      print("Warning: Fasta path for PSSM not found! PSSM will be zeros.")
      self.seq_to_pssm_id = {}
   
  def process(self):
    print(f"Generating Packed H5 ({self.cfg['output_name']})...")
    print("Feature Composition: OneHot(21) + PE(16) + PSSM(20) + Dihedrals(4) = 61 dims")
   
    df = pd.read_csv(self.cfg['csv_path'])
    df['target_key'] = df['target_key'].astype(str)
   
    key_to_seq = {}
    if 'Target' in df.columns:
      for _, row in df.iterrows():
        key_to_seq[str(row['target_key'])] = str(row['Target'])
   
    unique_keys = df['target_key'].unique()
   
    # 
    all_x_list = []    
    all_edge_index_list = []
    all_edge_attr_list = []
    all_pkt_mask_list = [] 
    ptr_node = [0]
    ptr_edge = [0]
    valid_keys = []
    max_pkts = self.cfg['max_pockets']
    count = 0
   
    for key in tqdm(unique_keys, desc="Processing"):
      try:
        # 1. 
        residues, coords_ca, coords_n, coords_c, pocket_lists = parse_protein_data(
          key, self.cfg['pdb_dir'], self.cfg['p2rank_dir'], max_pkts
        )
        if residues is None: continue
        num_nodes = len(residues)
       
        # 2. PSSM  (20 dim)
        seq_str = key_to_seq.get(key, "".join(residues))
        pssm_id = self.seq_to_pssm_id.get(seq_str)
        x_pssm = None
        if pssm_id:
          pssm_file = os.path.join(self.cfg['pssm_dir'], f"{pssm_id}.pssm")
          if os.path.exists(pssm_file):
            x_pssm = read_pssm(pssm_file)
       
        if x_pssm is None:
          x_pssm = np.zeros((num_nodes, 20), dtype=np.float32)
        elif x_pssm.shape[0] != num_nodes:
          min_len = min(num_nodes, x_pssm.shape[0])
          new_pssm = np.zeros((num_nodes, 20), dtype=np.float32)
          new_pssm[:min_len] = x_pssm[:min_len]
          x_pssm = new_pssm
       
        # 3.  (4 dim, PyTorch)
        x_dihedrals = get_dihedrals_torch(coords_n, coords_ca, coords_c)
       
        # 4.  (OneHot + PE) (37 dim)
        pe = get_sinusoidal_pos_encoding(num_nodes, d_model=16)
        x_basic = []
        for i, res in enumerate(residues):
          x_basic.append(get_residue_features_basic(res, pe[i]))
        x_basic = np.array(x_basic, dtype=np.float32)
       
        # 5.  (61 dims)
        x_final = np.concatenate([x_basic, x_pssm, x_dihedrals], axis=1)
       
        # 6.  (RBF on CA distance)
        dist_mat = distance_matrix(coords_ca, coords_ca)
        src, dst = np.where((dist_mat < 8.0) & (dist_mat > 1e-6))
       
        edge_attr = rbf_expand(dist_mat[src, dst].astype(np.float32)) # 16 dim
        edge_index = np.stack([src, dst], axis=0).astype(np.longlong) + ptr_node[-1]
       
        # 7.  Mask
        pkt_mask = np.zeros((num_nodes, max_pkts), dtype=np.float32)
        for k, indices in enumerate(pocket_lists):
          if k >= max_pkts: break
          for idx in indices:
            pkt_mask[idx, k] = 1.0
       
        # 8. 
        all_x_list.append(x_final)
        all_edge_index_list.append(edge_index)
        all_edge_attr_list.append(edge_attr)
        all_pkt_mask_list.append(pkt_mask)
        valid_keys.append(str(key))
       
        ptr_node.append(ptr_node[-1] + num_nodes)
        ptr_edge.append(ptr_edge[-1] + edge_index.shape[1])
        count += 1
       
      except Exception as e:
        # print(f"Error {key}: {e}")
        pass
   
    if count == 0: return

    # 9. 
    print(f" Processed {count} proteins.")
    X_all = np.concatenate(all_x_list, axis=0)
    E_all = np.concatenate(all_edge_index_list, axis=1)
    EA_all = np.concatenate(all_edge_attr_list, axis=0)
    P_all = np.concatenate(all_pkt_mask_list, axis=0)
   
    print(f" Node Dim: {X_all.shape[1]} (Expected 61)")
    print(f" Edge Dim: {EA_all.shape[1]}")

    with h5py.File(self.save_path, 'w') as f:
      f.create_dataset("x", data=X_all, compression="gzip")
      f.create_dataset("edge_index", data=E_all, compression="gzip")
      f.create_dataset("edge_attr", data=EA_all, compression="gzip")
      f.create_dataset("pkt_mask", data=P_all, compression="gzip")
      f.create_dataset("ptr_node", data=np.array(ptr_node), compression="gzip")
      f.create_dataset("ptr_edge", data=np.array(ptr_edge), compression="gzip")
      f.create_dataset("target_keys", data=np.array([k.encode('utf-8') for k in valid_keys]), compression="gzip")
     
    print(f"H5 Saved: {self.save_path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, required=True, choices=["Davis", "KIBA"])
  args = parser.parse_args()
 
  ProteinH5Generator(args.task).process()
