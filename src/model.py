import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from torch.nn import utils 
import torch.nn.init as init
from torch_geometric.data import Batch
from .moleculegnn import MolTEGNNEncoder
from torch_geometric.utils import to_dense_batch

from .proteingnn import ProteinTEGNNEncoder
from torch_geometric.nn import global_mean_pool, global_add_pool

class FCNet(nn.Module): 
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2): 
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x) 


class LightMLPDecoder3(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=1, dropout=0.3):
        super().__init__()
     
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim// 2, out_dim)
        
        self.activation = nn.ELU() 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x


class TSFusion(nn.Module):
    def __init__(self, num_hidden_a, num_hidden_b, num_hidden):
        super(TSFusion, self).__init__()
        self.hidden = num_hidden
        self.w1 = nn.Parameter(torch.Tensor(num_hidden_a, num_hidden))
        self.w2 = nn.Parameter(torch.Tensor(num_hidden_b, num_hidden))
        self.bias = nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()

    def reset_parameter(self):
        stdv1 = 1. / math.sqrt(self.hidden)
        stdv2 = 1. / math.sqrt(self.hidden)
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, a, b):

        wa = torch.matmul(a, self.w1)
        wb = torch.matmul(b, self.w2)
        gated = wa + wb + self.bias
        gate = torch.sigmoid(gated)
        

        output = gate * a + (1 - gate) * b
        return output

class UniCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, n_heads, dropout=0.1, dim_feedforward_scale=1):
        super().__init__()
        
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=query_dim, kdim=key_dim, vdim=key_dim,
                                          num_heads=n_heads, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        hidden_dim = int(query_dim * dim_feedforward_scale)
        self.norm_ff = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query_emb, key_emb, key_padding_mask=None):
        # query_emb: [B, N, D] (Structure)
        # key_emb: [B, S, D] (Sequence)
        q = self.norm_q(query_emb)
        k = self.norm_k(key_emb)
        
        attn_out, _ = self.attn(query=q, key=k, value=k, key_padding_mask=key_padding_mask)
        
        x = query_emb + self.dropout(attn_out)
        x = x + self.ffn(self.norm_ff(x))
        return x




class DTAPredictor(nn.Module):
    def __init__(self, 
                 drug_node_dim=78, drug_pre_dim=None, 
                 prot_node_dim=49, prot_pre_dim=None, 
                 drug_dim=256, target_dim=256, 
                 n_heads=4, 
                 dropout_gnn=0.15,    
                 dropout_seq=0.3,    
                 dropout_attn=0.15,   
                 dropout_mlp=0.3,    
                 use_drug_struc=True, use_drug_seq=True,
                 use_prot_struc=True, use_prot_seq=True,
                 **kwargs):
        
        super().__init__()
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.use_drug_struc = use_drug_struc
        self.use_drug_seq = use_drug_seq
        self.use_prot_struc = use_prot_struc
        self.use_prot_seq = use_prot_seq
        self.dropout_seq_layer = nn.Dropout(dropout_seq)

        if self.use_drug_struc:

            self.drug_gnn = MolTEGNNEncoder(input_dim=drug_node_dim, output_dim=drug_dim, dropout=dropout_gnn)
            self.d_pool_norm = nn.LayerNorm(drug_dim)
            
        if self.use_drug_seq:

            self.drug_seq_proj = nn.Sequential(
                nn.Linear(drug_pre_dim, drug_dim * 2), nn.ReLU(), nn.Dropout(dropout_seq), 
                nn.Linear(drug_dim * 2, drug_dim)
            )

        if self.use_prot_struc:

            self.prot_tegnn = ProteinTEGNNEncoder(num_features_pro=prot_node_dim, output_dim=target_dim, dropout=dropout_gnn)
            self.p_pool_norm = nn.LayerNorm(target_dim)
            
        if self.use_prot_seq:

            self.prot_seq_proj = nn.Sequential(
                nn.Linear(prot_pre_dim, target_dim * 2), nn.ReLU(), nn.Dropout(dropout_seq), 
                nn.Linear(target_dim * 2, target_dim)
            )

        if self.use_drug_struc and self.use_drug_seq:
            self.drug_cra = UniCrossAttention(query_dim=drug_dim, key_dim=drug_dim, n_heads=n_heads, dropout=dropout_attn)
            self.drug_fusion = TSFusion(num_hidden_a=drug_dim, num_hidden_b=drug_dim, num_hidden=drug_dim)
        
        if self.use_prot_struc and self.use_prot_seq:
            self.prot_cra = UniCrossAttention(query_dim=target_dim, key_dim=target_dim, n_heads=n_heads, dropout=dropout_attn)
            self.prot_fusion = TSFusion(num_hidden_a=target_dim, num_hidden_b=target_dim, num_hidden=target_dim)

        self.relu = nn.ReLU()
        
        final_dim = 0
        if self.use_drug_struc or self.use_drug_seq: final_dim += drug_dim
        if self.use_prot_struc or self.use_prot_seq: final_dim += target_dim
        
        if final_dim == 0: raise ValueError("No features selected!")
        
        self.fused_norm = nn.LayerNorm(final_dim)
        

        self.affinity_layer = LightMLPDecoder3(in_dim=final_dim, hidden_dim=512, dropout=dropout_mlp)
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.MultiheadAttention):
                 if m.in_proj_weight is not None: init.xavier_normal_(m.in_proj_weight)

    def forward(self, batch):
        d_seq_raw = batch.d_seq
        p_seq_raw = batch.p_seq
        
        if d_seq_raw.dim() == 1:
            d_seq_raw = d_seq_raw.view(batch.num_graphs, -1) 
            
        if p_seq_raw.dim() == 1:
            p_seq_raw = p_seq_raw.view(batch.num_graphs, -1) 

        d_struc_pooled = None
        d_seq_vec = None
        d_final = None

        if self.use_drug_seq:
            d_seq_vec = self.drug_seq_proj(d_seq_raw) 

            d_seq_vec = self.dropout_seq_layer(self.relu(d_seq_vec))
            d_seq_emb = d_seq_vec.unsqueeze(1) 

        if self.use_drug_struc:
            batch_idx = batch.d_x_batch if hasattr(batch, 'd_x_batch') else batch.batch
            d_node_feats, _ = self.drug_gnn(
                x=batch.d_x, 
                edge_index=batch.d_edge_index, 
                edge_attr=batch.d_edge_attr, 
                batch=batch_idx
            )
            d_struc_nodes, d_struc_mask = to_dense_batch(d_node_feats, batch_idx)

        if self.use_drug_struc and self.use_drug_seq:
            d_struc_updated = self.drug_cra(query_emb=d_struc_nodes, key_emb=d_seq_emb, key_padding_mask=None)
            mask = d_struc_mask.unsqueeze(-1).float()
            d_struc_pooled = (d_struc_updated * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))
            d_struc_pooled = self.d_pool_norm(d_struc_pooled)
            d_final = self.drug_fusion(d_struc_pooled, d_seq_vec)
        
        elif self.use_drug_struc:
            mask = d_struc_mask.unsqueeze(-1).float()
            d_struc_pooled = (d_struc_nodes * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))
            d_final = self.d_pool_norm(d_struc_pooled)
        elif self.use_drug_seq:
            d_final = d_seq_vec

        p_struc_pooled = None
        p_seq_vec = None
        p_final = None

        if self.use_prot_seq:
            p_seq_vec = self.prot_seq_proj(p_seq_raw) 

            p_seq_vec = self.dropout_seq_layer(self.relu(p_seq_vec))
            p_seq_emb = p_seq_vec.unsqueeze(1)

        if self.use_prot_struc:
            p_batch_idx = batch.p_x_batch if hasattr(batch, 'p_x_batch') else batch.batch
            p_node_feats = self.prot_tegnn(
                x=batch.p_x, 
                edge_index=batch.p_edge_index, 
                edge_attr=batch.p_edge_attr, 
                batch=p_batch_idx, 
                pkt_mask=batch.p_pkt_mask
            )
            p_struc_nodes, p_struc_mask = to_dense_batch(p_node_feats, p_batch_idx)

        if self.use_prot_struc and self.use_prot_seq:
            p_struc_updated = self.prot_cra(query_emb=p_struc_nodes, key_emb=p_seq_emb, key_padding_mask=None)
            mask = p_struc_mask.unsqueeze(-1).float()
            p_struc_pooled = (p_struc_updated * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))
            p_struc_pooled = self.p_pool_norm(p_struc_pooled)
            p_final = self.prot_fusion(p_struc_pooled, p_seq_vec)
            
        elif self.use_prot_struc:
            mask = p_struc_mask.unsqueeze(-1).float()
            p_struc_pooled = (p_struc_nodes * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))
            p_final = self.p_pool_norm(p_struc_pooled)
        elif self.use_prot_seq:
            p_final = p_seq_vec

        v_cat = torch.cat([d_final, p_final], dim=-1) 
        v_cat = self.fused_norm(v_cat)
        affinity = self.affinity_layer(v_cat)
        
        return affinity.squeeze(-1)

