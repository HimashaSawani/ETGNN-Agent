import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
import scipy.sparse as sp

device = torch.device("cpu")
print("=== LOCAL CPU PIPELINE DRY RUN ===")

DATA_DIR = Path("data/processed")
train_data = torch.load(DATA_DIR / "train_graph.pt", map_location="cpu", weights_only=True)
val_data = torch.load(DATA_DIR / "validation_graph.pt", map_location="cpu", weights_only=True)
test_data = torch.load(DATA_DIR / "test_graph.pt", map_location="cpu", weights_only=True)

from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

def encode_time(time_step, time_dim=16):
    half_dim = time_dim // 2
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(np.log(10000.0) / (half_dim - 1)))
    args = time_step.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class ETGNNClassifier(nn.Module):
    def __init__(self, input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim + time_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, 2, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.time_dim = time_dim

    def forward(self, x, edge_index, time_step):
        encoded = encode_time(time_step, self.time_dim)
        combined = torch.cat((x, encoded), dim=1)
        h = self.conv1(combined, edge_index).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)

class ResidualETGNNClassifier(nn.Module):
    def __init__(self, input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.time_dim = time_dim
        self.dropout = dropout
        in_dim = input_dim + time_dim
        self.conv1 = GATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.res1 = nn.Linear(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, 2, heads=1, concat=False, dropout=dropout)
        self.res2 = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, time_step):
        encoded = encode_time(time_step, self.time_dim)
        combined = torch.cat((x, encoded), dim=1)
        h_graph = self.conv1(combined, edge_index)
        h_res = self.res1(combined)
        h = F.relu(h_graph + h_res)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out_graph = self.conv2(h, edge_index)
        out_res = self.res2(h)
        return out_graph + out_res

# 1. Check historical M0 baseline loading
m0 = ETGNNClassifier()
m0_dict = torch.load("results/checkpoints/etgnn.pt", map_location="cpu", weights_only=False)
m0.load_state_dict(m0_dict.get("state_dict", m0_dict))
m0.eval()

# 2. Check 1 training step of ResidualETGNN
m1 = ResidualETGNNClassifier()
optimizer = torch.optim.Adam(m1.parameters(), lr=0.01)
t_mask = train_data['train_mask']
train_labels = train_data['y'][t_mask]
ce_weight = torch.tensor([1.0, float((train_labels == 0).sum() / (train_labels == 1).sum())])
criterion = nn.CrossEntropyLoss(weight=ce_weight)

m1.train()
optimizer.zero_grad()
out = m1(train_data['x'], train_data['edge_index'], train_data['time_step'])
loss = criterion(out[t_mask], train_data['y'][t_mask])
loss.backward()
optimizer.step()
print(f"M1 1-step train loss: {loss.item():.4f}")

# 3. Check Validation Threshold Tuning
v_mask = val_data['val_mask']
m1.eval()
with torch.no_grad():
    v_logits = m1(val_data['x'], val_data['edge_index'], val_data['time_step'])
    v_probs = F.softmax(v_logits[v_mask], dim=1)[:, 1].numpy()
    v_true = val_data['y'][v_mask].numpy()
    v_ap = average_precision_score(v_true, v_probs)
    best_tau = 0.5
    best_f1 = -1.0
    for tau in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(v_true, (v_probs >= tau).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
print(f"M1 Val AUPRC: {v_ap:.4f} | Optimal Tau: {best_tau:.2f} (Val F1: {best_f1:.4f})")

# 4. Check Degree Stratification on Test Graph
test_mask = test_data['test_mask']
test_true = test_data['y'][test_mask].numpy()
test_indices = torch.where(test_mask)[0]
dst_nodes = test_data['edge_index'][1]
deg_map = torch.bincount(dst_nodes, minlength=test_data['x'].size(0))
test_degrees = deg_map[test_indices].numpy()

s0_idx = np.where(test_degrees == 0)[0]
s1_idx = np.where((test_degrees >= 1) & (test_degrees <= 2))[0]
s2_idx = np.where(test_degrees >= 3)[0]

with torch.no_grad():
    t_logits = m1(test_data['x'], test_data['edge_index'], test_data['time_step'])
    t_probs = F.softmax(t_logits[test_mask], dim=1)[:, 1].numpy()

t_ap = average_precision_score(test_true, t_probs)
t_roc = roc_auc_score(test_true, t_probs)
t_pred = (t_probs >= best_tau).astype(int)
t_f1 = f1_score(test_true, t_pred, zero_division=0)

ap_s0 = average_precision_score(test_true[s0_idx], t_probs[s0_idx])
ap_s1 = average_precision_score(test_true[s1_idx], t_probs[s1_idx])
ap_s2 = average_precision_score(test_true[s2_idx], t_probs[s2_idx])

print(f"M1 Test AUPRC: {t_ap:.4f} | ROC-AUC: {t_roc:.4f} | F1: {t_f1:.4f}")
print(f"Degree Strata AUPRC -> S0 (k=0): {ap_s0:.4f} | S1 (k=1..2): {ap_s1:.4f} | S2 (k>=3): {ap_s2:.4f}")

# 5. Check Connected Components on Validation Graph
val_edge = val_data['edge_index'].numpy()
val_num_nodes = val_data['x'].shape[0]
adj = sp.coo_matrix((np.ones(val_edge.shape[1]), (val_edge[0], val_edge[1])), shape=(val_num_nodes, val_num_nodes))
n_comp, comp_labels = sp.csgraph.connected_components(adj, directed=False)
val_steps = val_data['time_step'].numpy()
cohort_mask = (val_steps >= 37) & (val_steps <= 40)
cohort_components = np.unique(comp_labels[cohort_mask])
print(f"Validation Graph Total Components: {n_comp} | In-Cohort (37-40) Components: {len(cohort_components)}")

print("\n[SUCCESS] Pipeline CPU Dry Run Completed Successfully! Zero runtime errors.")

