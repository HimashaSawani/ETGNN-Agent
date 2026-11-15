import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from src.models.time_encoding import encode_time

class ResidualETGNNClassifier(nn.Module):
    """
    Model A: Temporal GAT with explicit ego-feature residual projections.
    Provides a stable fallback to ego-features without learnable edge gating.
    """
    def __init__(self, input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.time_dim = time_dim
        self.dropout = dropout
        
        in_dim = input_dim + time_dim
        # Layer 1: GAT (4 heads) + Ego Residual
        self.conv1 = GATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.res1 = nn.Linear(in_dim, hidden_dim)
        
        # Layer 2: GAT (1 head) + Ego Residual
        self.conv2 = GATConv(hidden_dim, 2, heads=1, concat=False, dropout=dropout)
        self.res2 = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, time_step):
        encoded = encode_time(time_step, self.time_dim)
        combined = torch.cat((x, encoded), dim=1)
        
        # Layer 1: Graph Aggregation + Ego-Feature Skip
        h_graph = self.conv1(combined, edge_index)
        h_res = self.res1(combined)
        h = F.relu(h_graph + h_res)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 2: Output Aggregation + Ego-Feature Skip
        out_graph = self.conv2(h, edge_index)
        out_res = self.res2(h)
        return out_graph + out_res


class GatedGATConv(MessagePassing):
    """
    Custom GATConv with endpoint-conditioned continuous edge-reliability gating.
    Scales message passing by continuous gate gamma in (0, 1).
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Linear projections for attention
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Edge Reliability Gating MLP (conditioned purely on endpoint features, no delta-t)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        # Initialize gate bias to neutral (sigmoid ~ 0.5)
        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, return_gates=False):
        num_nodes = x.size(0)
        
        # 1. Clean self-loops first to ensure exact alignment with gate vectors
        edge_index_nosl, _ = remove_self_loops(edge_index)
        if edge_index_nosl.numel() == 0:
            out = torch.zeros(num_nodes, self.heads * self.out_channels, device=x.device)
            gates = torch.empty(0, device=x.device)
            return (out + self.bias, gates) if return_gates else out + self.bias
            
        src_idx, dst_idx = edge_index_nosl[0], edge_index_nosl[1]
        
        # 2. Compute continuous edge-reliability gate gamma in (0, 1) from endpoint features
        endpoint_features = torch.cat([x[src_idx], x[dst_idx]], dim=-1)
        gates = self.gate_mlp(endpoint_features).squeeze(-1)  # shape: (E,)
        
        # 3. Add self-loops for central attention
        edge_index_sl, _ = add_self_loops(edge_index_nosl, num_nodes=num_nodes)
        
        # Self-loops have intrinsic reliability = 1.0 (ego feature cannot suppress itself)
        num_self_loops = num_nodes
        self_loop_gates = torch.ones(num_self_loops, device=x.device)
        extended_gates = torch.cat([gates, self_loop_gates], dim=0)  # shape: (E + N,)
        
        # 3. Attention Projections
        h_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
        h_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)
        
        out = self.propagate(edge_index_sl, h_src=h_src, h_dst=h_dst,
                             gates=extended_gates, size=None)
        out = out.view(-1, self.heads * self.out_channels) + self.bias
        
        return (out, gates) if return_gates else out

    def message(self, h_src_j, h_dst_i, edge_index_i, gates, index, ptr, size_i):
        # Attention coefficient
        alpha_src = (h_src_j * self.att_src).sum(dim=-1)
        alpha_dst = (h_dst_i * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)
        alpha = softmax(alpha, index, ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Gate modulation: scale neighbor message by continuous reliability gate gamma
        # gates: (E,), alpha: (E, heads), h_src_j: (E, heads, out_channels)
        gated_alpha = alpha * gates.view(-1, 1)
        return h_src_j * gated_alpha.unsqueeze(-1)


class ReliabilityAwareETGNNClassifier(nn.Module):
    """
    Model B: Reliability-Aware Temporal Graph Neural Network.
    Integrates continuous endpoint-conditioned edge gating with ego-residual fallback.
    """
    def __init__(self, input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.time_dim = time_dim
        self.dropout = dropout
        
        in_dim = input_dim + time_dim
        # Layer 1: Gated GAT (4 heads, output hidden_dim) + Residual Skip
        self.conv1 = GatedGATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.res1 = nn.Linear(in_dim, hidden_dim)
        
        # Layer 2: Gated GAT (1 head, output 2) + Residual Skip
        self.conv2 = GatedGATConv(hidden_dim, 2, heads=1, dropout=dropout)
        self.res2 = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, time_step, return_gates=False):
        encoded = encode_time(time_step, self.time_dim)
        combined = torch.cat((x, encoded), dim=1)
        
        # Layer 1
        if return_gates:
            h_graph, gates1 = self.conv1(combined, edge_index, return_gates=True)
        else:
            h_graph = self.conv1(combined, edge_index)
            gates1 = None
            
        h_res = self.res1(combined)
        h = F.relu(h_graph + h_res)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 2
        if return_gates:
            out_graph, gates2 = self.conv2(h, edge_index, return_gates=True)
        else:
            out_graph = self.conv2(h, edge_index)
            gates2 = None
            
        out_res = self.res2(h)
        logits = out_graph + out_res
        
        if return_gates:
            return logits, {"gates1": gates1, "gates2": gates2}
        return logits


class DeepParameterMatchedMLP(nn.Module):
    """
    Model C: Deep Parameter-Matched Feature-Only MLP Control (~31.7k parameters).
    Matches the depth, non-linearities, and representational capacity of the GNN models.
    """
    def __init__(self, input_dim=165, time_dim=16, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super().__init__()
        self.time_dim = time_dim
        self.dropout = dropout
        
        in_dim = input_dim + time_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 2)

    def forward(self, x, edge_index, time_step):
        encoded = encode_time(time_step, self.time_dim)
        combined = torch.cat((x, encoded), dim=1)
        
        h = F.relu(self.fc1(combined))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc3(h)
