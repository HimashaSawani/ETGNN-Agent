import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

def notebook_encode_time(time_step, time_dim=16):
    device = time_step.device
    half_dim = time_dim // 2
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -(np.log(10000.0) / (half_dim - 1)))
    args = time_step.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class GatedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3, enable_gate=True):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.enable_gate = enable_gate
        self.lin_src = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        if self.enable_gate:
            self.gate_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * in_channels, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
                torch.nn.Sigmoid()
            )
        self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))

    def forward(self, x, edge_index, return_gates=False):
        num_nodes = x.size(0)
        edge_index_nosl, _ = remove_self_loops(edge_index)
        if edge_index_nosl.numel() == 0:
            out = torch.zeros(num_nodes, self.heads * self.out_channels, device=x.device)
            gates = torch.empty(0, device=x.device)
            return (out + self.bias, gates) if return_gates else out + self.bias

        src_idx, dst_idx = edge_index_nosl[0], edge_index_nosl[1]
        if self.enable_gate:
            endpoint_features = torch.cat([x[src_idx], x[dst_idx]], dim=-1)
            gates = self.gate_mlp(endpoint_features).squeeze(-1)
        else:
            gates = torch.ones(edge_index_nosl.size(1), device=x.device)

        edge_index_sl, _ = add_self_loops(edge_index_nosl, num_nodes=num_nodes)
        self_loop_gates = torch.ones(num_nodes, device=x.device)
        extended_gates = torch.cat([gates, self_loop_gates], dim=0)

        h_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
        h_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)
        out = self.propagate(edge_index_sl, h_src=h_src, h_dst=h_dst, gates=extended_gates, size=None)
        out = out.view(-1, self.heads * self.out_channels) + self.bias
        return (out, gates) if return_gates else out

    def message(self, h_src_j, h_dst_i, edge_index_i, gates, index, ptr, size_i):
        alpha_src = (h_src_j * self.att_src).sum(dim=-1)
        alpha_dst = (h_dst_i * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)
        alpha = softmax(alpha, index, ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        gated_alpha = alpha * gates.view(-1, 1)
        return h_src_j * gated_alpha.unsqueeze(-1)

class ReliabilityAwareETGNNClassifier(torch.nn.Module):
    def __init__(self, input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3, enable_gate=True):
        super().__init__()
        self.time_dim = time_dim
        self.dropout = dropout
        in_dim = input_dim + time_dim
        self.conv1 = GatedGATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout, enable_gate=enable_gate)
        self.res1 = torch.nn.Linear(in_dim, hidden_dim)
        self.conv2 = GatedGATConv(hidden_dim, 2, heads=1, dropout=dropout, enable_gate=enable_gate)
        self.res2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, time_step, return_gates=False):
        encoded = notebook_encode_time(time_step, self.time_dim)
        combined = torch.cat((x, encoded), dim=1)
        if return_gates:
            h_graph, gates1 = self.conv1(combined, edge_index, return_gates=True)
        else:
            h_graph = self.conv1(combined, edge_index)
            gates1 = None
        h_res = self.res1(combined)
        h = F.relu(h_graph + h_res)
        h = F.dropout(h, p=self.dropout, training=self.training)
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

val_data = torch.load("data/processed/validation_graph.pt", map_location="cpu", weights_only=True)
test_data = torch.load("data/processed/test_graph.pt", map_location="cpu", weights_only=True)

print("=== DIAGNOSING GATE ACTIVATIONS FOR M3 (CE & FOCAL) ===")
ckpt_dir = Path("research_extension/phase13_advanced_investigation/checkpoints")

gate_stats = []
for m_type in ["M3_RG_ETGNN_CE", "M3_RG_ETGNN_Focal"]:
    for seed in [42, 43, 44, 45, 46]:
        ckpt_path = ckpt_dir / f"{m_type}_s{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ReliabilityAwareETGNNClassifier(enable_gate=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.no_grad():
            _, g_val = model(val_data['x'], val_data['edge_index'], val_data['time_step'], return_gates=True)
            _, g_test = model(test_data['x'], test_data['edge_index'], test_data['time_step'], return_gates=True)

        g1_v = g_val["gates1"].numpy()
        g2_v = g_val["gates2"].numpy()
        g1_t = g_test["gates1"].numpy()
        g2_t = g_test["gates2"].numpy()

        gate_stats.append({
            "model": m_type, "seed": seed,
            "val_g1_mean": g1_v.mean(), "val_g1_std": g1_v.std(), "val_g1_min": g1_v.min(), "val_g1_max": g1_v.max(),
            "val_g2_mean": g2_v.mean(), "val_g2_std": g2_v.std(),
            "test_g1_mean": g1_t.mean(), "test_g1_std": g1_t.std(), "test_g1_min": g1_t.min(), "test_g1_max": g1_t.max(),
            "test_g2_mean": g2_t.mean(), "test_g2_std": g2_t.std(),
        })

df_gates = pd.DataFrame(gate_stats)
print(df_gates.to_string())
df_gates.to_csv("research_extension/phase14_results_analysis/M3_GATE_ACTIVATION_DIAGNOSTICS.csv", index=False)

