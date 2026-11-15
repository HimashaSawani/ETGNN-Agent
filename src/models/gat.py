import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GATClassifier(nn.Module):
    def __init__(self, input_dim=165, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // 4, heads=4,
                             dropout=dropout)
        self.conv2 = GATConv(hidden_dim, 2, heads=1, concat=False,
                             dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
