import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

from src.models.time_encoding import encode_time


class ETGNNClassifier(nn.Module):
    def __init__(self, input_dim=165, time_dim=16,
                 hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim + time_dim, hidden_dim // 4,
                             heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, 2, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = dropout
        self.time_dim = time_dim

    def forward(self, x, edge_index, time_step):
        encoded = encode_time(time_step, self.time_dim)
        combined = torch.cat((x, encoded), dim=1)
        h = self.conv1(combined, edge_index).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)
