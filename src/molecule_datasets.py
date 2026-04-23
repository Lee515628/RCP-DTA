import os
import pickle
from itertools import chain, repeat

import networkx as nx
import numpy as np
import pandas as pd
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import (Data, InMemoryDataset, download_url, extract_zip)


def get_bond_feature_vector(bond):
    bt = bond.GetBondType()
    features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return torch.tensor(features, dtype=torch.float)

def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None, None, 0
    
    edge_indices = []
    edge_attrs = []


    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = get_bond_feature_vector(bond)
        

        edge_indices.append([i, j])
        edge_attrs.append(feat)
        edge_indices.append([j, i])
        edge_attrs.append(feat)


    num_atoms = mol.GetNumAtoms()
    self_loop_feat = torch.zeros(6, dtype=torch.float) #
    for k in range(num_atoms):
        edge_indices.append([k, k])
        edge_attrs.append(self_loop_feat)

    if not edge_indices: 
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 6), dtype=torch.float), num_atoms

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs, dim=0)
    
    return edge_index, edge_attr, num_atoms


allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [ 
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6], 
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC 
    ],
    'possible_bond_dirs' : [ 
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple_molebert(mol):

    num_atom_features = 2  
    atom_features_list = []
    for atom in mol.GetAtoms(): 
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)         
    # 转换为PyTorch长整型张量    
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds 边特征 + 边索引
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: ## 若分子存在化学键
        edges_list = [] # 存储边的两端原子索引
        edge_features_list = []# 存储边特征
        for bond in mol.GetBonds():# 遍历所有化学键
            i = bond.GetBeginAtomIdx()# 化学键起始原子索引
            j = bond.GetEndAtomIdx()# 化学键终止原子索引
            # 边特征：键类型的索引 + 键方向的索引
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))# 添加正向边
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))# 添加反向边
            edge_features_list.append(edge_feature) #键特征包含两种类型分别为Bond type和Bond direction

        #边索引：转换为PyG要求的COO格式（2行N列，第一行起点，第二行终点）
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # 边特征：转换为长整型张量
        edge_attr = torch.tensor(np.array(edge_features_list),
                                dtype=torch.long)
    else:   # 若分子无化学键空边索引和空边特征
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    #构建PyG的Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # 返回PyG图数据对象
    return data #得到图神经网络的数据格式Data
#标准版分子→PyG 图转换
def mol_to_graph_data_obj_simple(mol):
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []# 遍历所有原子
    for atom in mol.GetAtoms(): # 调用OGB工具函数提取原子特征（默认9维特征，含原子序数、杂化类型等）
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # # 2. 处理化学键（边特征 + 边索引）
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def create_standardized_mol_id(smiles):
    """ smiles -> inchi """

    if check_smiles_validity(smiles):
        #  # 移除立体化学信息
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)# SMILES→RDKit分子对象
        if mol is not None:
            # 处理含多个分子片段的SMILES
            # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)\
            # c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol) # 拆分分子片段
                largest_mol = get_largest_mol(mol_species_list) # 选择最大分子
                inchi = AllChem.MolToInchi(largest_mol)# 转换为InChI
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
    return# 若SMILES无效或处理失败，返回None


class MoleculeDatasetComplete(InMemoryDataset):
    def __init__(self, root, dataset='zinc250k', transform=None,
                 pre_transform=None, pre_filter=None, empty=False):

        self.root = root# 初始化数据集根目录
        self.dataset = dataset# 指定数据集名称
        self.transform = transform# 数据转换函数
        self.pre_filter = pre_filter# 数据过滤函数
        self.pre_transform = pre_transform# 数据预处理函数
        
        # 调用父类 InMemoryDataset 的初始化方法
        super(MoleculeDatasetComplete, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            # processed_paths[0] 是处理后数据的存储路径
            #self.data（合并的批量图数据）、self.slices存储每个样本的切片位置
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))

    def get(self, idx): #get 方法通过切片拆分出单个样本
        # 初始化一个空的 PyG Data 对象
        data = Data()# 遍历批量数据的所有键
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data#支持通过索引 idx 快速获取单个分子的图数据

    @property
    def raw_file_names(self):
        if self.dataset == 'davis':
            file_name_list = ['davis']
        elif self.dataset == 'kiba':
            file_name_list = ['kiba']
        else:
            file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):#指定处理后的图数据存储文件名，
        return 'geometric_data_processed.pt'

    def download(self):#表示数据集需要手动下载并放置到 raw 目录
        return

    def process(self):
        # 定义内部函数：共享的分子→图数据转换逻辑
        def shared_extractor(smiles_list, rdkit_mol_objs, labels):
            data_list = []# 存储所有样本的 PyG Data 对象
            data_smiles_list = [] #存储对应的 SMILES 字符串
            # 处理标签：若标签是1维数组（如 [0,1,0]），扩展为2维（如 [[0],[1],[0]]），适配 PyG 格式
            if labels.ndim == 1:
                labels = np.expand_dims(labels, axis=1)
            # 遍历每个分子
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol is None:
                    continue
                # 调用 `mol_to_graph_data_obj_simple` 函数：RDKit 分子→PyG 图数据
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data.id = torch.tensor([i])# 给数据添加索引 ID
                data.y = torch.tensor(labels[i]) # 添加标签
                data_list.append(data)# 加入数据列表
                data_smiles_list.append(smiles_list[i])
             # 返回处理后的图数据列表和 SMILES 列表
            return data_list, data_smiles_list

        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'geom':
            input_path = self.raw_paths[0]
            data_list, data_smiles_list = [], []
            input_df = pd.read_csv(input_path, sep=',', dtype='str')
            smiles_list = list(input_df['smiles'])
            for i in range(len(smiles_list)):
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol is not None:
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    data.id = torch.tensor([i])
                    data_list.append(data)
                    data_smiles_list.append(s)

        else:#未知数据集报错处理
            raise ValueError('Dataset {} not included.'.format(self.dataset))

        if self.pre_filter is not None:#筛选符合条件的数据
        # 仅保留满足 pre_filter(data) 为 True 的数据
            data_list = [data for data in data_list if self.pre_filter(data)]
        #批量预处理数据
        if self.pre_transform is not None: # 对每个数据对象应用预处理（如特征标准化、图结构增强等）
            data_list = [self.pre_transform(data) for data in data_list]

        # For ESOL and FreeSOlv, there are molecules with single atoms and empty edges.数据有效性校验
        valid_index = [] # 存储有效数据的索引
        neo_data_smiles_list, neo_data_list = [], []# 存储有效数据的 SMILES 和图数据
        for i, (smiles, data) in enumerate(zip(data_smiles_list, data_list)):
            if data.edge_attr.size()[0] == 0:
                print('Invalid\t', smiles, data)
                continue
            valid_index.append(i)
            assert data.edge_attr.size()[1] == 3
            assert data.edge_index.size()[0] == 2
            assert data.x.size()[1] == 9
            assert data.id.size()[0] == 1
            assert data.y.size()[0] == 1
            # 收集有效数据
            neo_data_smiles_list.append(smiles)
            neo_data_list.append(data)
        
        old_N = len(data_smiles_list)# 过滤前总数据量
        neo_N = len(valid_index)# 过滤后有效数据量
        print('{} invalid out of {}.'.format(old_N - neo_N, old_N))
        print(len(neo_data_smiles_list), '\t', len(neo_data_list))
        # 将有效数据的 SMILES 保存为 CSV 文件
        data_smiles_series = pd.Series(neo_data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        data_smiles_series.to_csv(saver_path, index=False, header=False)
        # 将有效图数据列表合并为批量数据 + 切片信息
        data, slices = self.collate(neo_data_list)
        torch.save((data, slices), self.processed_paths[0])

        return

def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values



def _load_toxcast_dataset(input_path):

    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def check_smiles_validity(smiles):#校验 SMILES 有效性
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:# 若转换成功，说明 SMILES 格式合法
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):#拆分多分子片段的 RDKit 分子对象

    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):#从分子列表中筛选原子数最多的分子

    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]
