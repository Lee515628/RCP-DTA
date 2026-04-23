import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

class TEGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, use_virtual_feedback=True, activation=True):
        super().__init__()
        
        nn_model = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.conv = GINEConv(nn_model, train_eps=True)
        
        self.edge_lin = nn.Linear(16, in_dim)
        

        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout_val = dropout  
        self.use_virtual_feedback = use_virtual_feedback
        self.activation = activation 
        
        if use_virtual_feedback:
            self.virtual_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, pkt_mask, batch):

        edge_emb = self.edge_lin(edge_attr)
        
        x_res = self.conv(x, edge_index, edge_attr=edge_emb)
        

        if self.use_virtual_feedback and pkt_mask is not None:
            N, D = x_res.size()
            _, K = pkt_mask.size()

            x_weighted = x_res.unsqueeze(1) * pkt_mask.unsqueeze(2)
            
            x_weighted_flat = x_weighted.view(N, K * D)
            

            pocket_sum_flat = global_add_pool(x_weighted_flat, batch)
            pocket_count = global_add_pool(pkt_mask, batch)
            
            pocket_sum = pocket_sum_flat.view(-1, K, D)
            

            pocket_mean = pocket_sum / (pocket_count.unsqueeze(-1) + 1e-9)
            
            pocket_emb = self.virtual_proj(pocket_mean) # [B, K, D]
            
            node_pocket_emb = pocket_emb[batch]
            
            valid_feedback = node_pocket_emb * pkt_mask.unsqueeze(2)
            

            total_feedback = valid_feedback.sum(dim=1)
            
            x_res = x_res + total_feedback

        x_res = self.norm(x_res)
        
        if self.activation:
            x_res = self.relu(x_res)
            
        x_res = F.dropout(x_res, p=self.dropout_val, training=self.training)
        
        return x_res

class ProteinTEGNNEncoder(nn.Module):
    def __init__(self, num_features_pro=61, output_dim=256, dropout=0.1):
        super().__init__()
        
        hidden_dim = 64 
        

        self.layer1 = TEGNNLayer(
            in_dim=num_features_pro, 
            out_dim=hidden_dim, 
            dropout=dropout,
            use_virtual_feedback=True, 
            activation=True
        )
        
        # Layer 2: 64 -> 128
        self.layer2 = TEGNNLayer(
            in_dim=hidden_dim, 
            out_dim=hidden_dim * 2, 
            dropout=dropout,
            use_virtual_feedback=True, 
            activation=True
        )
        

        self.layer3 = TEGNNLayer(
            in_dim=hidden_dim * 2, 
            out_dim=output_dim, 
            dropout=0.0, 
            use_virtual_feedback=False, 
            activation=False 
        )

    def forward(self, x, edge_index, edge_attr, batch, pkt_mask=None):

        
        x = self.layer1(x, edge_index, edge_attr, pkt_mask, batch)
        x = self.layer2(x, edge_index, edge_attr, pkt_mask, batch)
        x = self.layer3(x, edge_index, edge_attr, pkt_mask, batch)
        
        return x
