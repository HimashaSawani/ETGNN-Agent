import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from src.models.etgnn import ETGNNClassifier
from src.models.time_encoding import encode_time as orig_encode_time

def notebook_encode_time(time_step, time_dim=16):
    device = time_step.device
    half_dim = time_dim // 2
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -(np.log(10000.0) / (half_dim - 1)))
    args = time_step.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

test_data = torch.load("data/processed/test_graph.pt", map_location="cpu", weights_only=True)
test_mask = test_data['test_mask']
test_true = test_data['y'][test_mask].numpy()

m0 = ETGNNClassifier()
ckpt = torch.load("results/checkpoints/etgnn.pt", map_location="cpu", weights_only=False)
m0.load_state_dict(ckpt)
m0.eval()

# 1. With original time encoding:
with torch.no_grad():
    enc_orig = orig_encode_time(test_data['time_step'], 16)
    comb_orig = torch.cat((test_data['x'], enc_orig), dim=1)
    h_orig = m0.conv1(comb_orig, test_data['edge_index']).relu()
    out_orig = m0.conv2(h_orig, test_data['edge_index'])
    probs_orig = F.softmax(out_orig[test_mask], dim=1)[:, 1].numpy()

ap_orig = average_precision_score(test_true, probs_orig)
roc_orig = roc_auc_score(test_true, probs_orig)
print(f"Original Time Encoding -> Test AP: {ap_orig:.6f} | ROC: {roc_orig:.6f}")

# 2. With notebook time encoding:
with torch.no_grad():
    enc_nb = notebook_encode_time(test_data['time_step'], 16)
    comb_nb = torch.cat((test_data['x'], enc_nb), dim=1)
    h_nb = m0.conv1(comb_nb, test_data['edge_index']).relu()
    out_nb = m0.conv2(h_nb, test_data['edge_index'])
    probs_nb = F.softmax(out_nb[test_mask], dim=1)[:, 1].numpy()

ap_nb = average_precision_score(test_true, probs_nb)
roc_nb = roc_auc_score(test_true, probs_nb)
print(f"Notebook Time Encoding -> Test AP: {ap_nb:.6f} | ROC: {roc_nb:.6f}")

