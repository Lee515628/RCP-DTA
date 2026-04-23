import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

class MolTEGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=6, dropout=0.1, activation=True):
        super().__init__()
        

        nn_model = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.conv = GINEConv(nn_model, train_eps=True)
        

        self.edge_lin = nn.Linear(edge_dim, in_dim)
        

        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout_val = dropout
        self.activation = activation

    def forward(self, x, edge_index, edge_attr, batch):

        edge_emb = self.edge_lin(edge_attr) 
        

        x = self.conv(x, edge_index, edge_attr=edge_emb)
        

        x = self.norm(x)
        
        if self.activation:
            x = self.relu(x)
            
        x = F.dropout(x, p=self.dropout_val, training=self.training)
        
        return x

class MolTEGNNEncoder(nn.Module):
    def __init__(self, input_dim=78, hidden_dim=64, output_dim=256, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.layer1 = MolTEGNNLayer(in_dim=input_dim, out_dim=hidden_dim, edge_dim=6, dropout=dropout, activation=True)
        self.layer2 = MolTEGNNLayer(in_dim=hidden_dim, out_dim=hidden_dim * 2, edge_dim=6, dropout=dropout, activation=True)
        self.layer3 = MolTEGNNLayer(in_dim=hidden_dim * 2, out_dim=output_dim, edge_dim=6, dropout=0.0, activation=False)


    def forward(self, x, edge_index, edge_attr, batch):
        """
        x: [Total_Nodes, Dim]
        edge_index: [2, Total_Edges]
        edge_attr: [Total_Edges, Edge_Dim]
        batch: [Total_Nodes]
        """
        x = self.layer1(x, edge_index, edge_attr, batch)
        x = self.layer2(x, edge_index, edge_attr, batch)
        x = self.layer3(x, edge_index, edge_attr, batch)
        
        return x, batch