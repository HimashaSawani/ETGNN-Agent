import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops

ROOT = Path(".")
PHASE14_DIR = ROOT / "research_extension" / "phase14_results_analysis"
PHASE14_DIR.mkdir(parents=True, exist_ok=True)

# Load Models & Validation Data
from src.models.etgnn import ETGNNClassifier as OrigETGNN
from src.models.time_encoding import encode_time as orig_encode_time

val_data = torch.load("data/processed/validation_graph.pt", map_location="cpu", weights_only=True)
val_mask = val_data['val_mask']
val_indices = torch.where(val_mask)[0].numpy()
val_labels = val_data['y'][val_mask].numpy()

# Load frozen historical baseline
m0 = OrigETGNN()
m0_ckpt = torch.load("results/checkpoints/etgnn.pt", map_location="cpu", weights_only=False)
m0.load_state_dict(m0_ckpt.get("state_dict", m0_ckpt))
m0.eval()

# Validation predictions
with torch.no_grad():
    enc_v = orig_encode_time(val_data['time_step'], 16)
    comb_v = torch.cat((val_data['x'], enc_v), dim=1)
    h_v = m0.conv1(comb_v, val_data['edge_index']).relu()
    out_v = m0.conv2(h_v, val_data['edge_index'])
    probs_v = F.softmax(out_v[val_mask], dim=1)[:, 1].numpy()

# Find validation optimal threshold tau*
best_tau = 0.5
best_f1 = -1.0
for tau in np.linspace(0.1, 0.9, 81):
    pred = (probs_v >= tau).astype(int)
    from sklearn.metrics import f1_score
    f1 = f1_score(val_labels, pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_tau = tau

print(f"Validation Tau*: {best_tau:.2f} (F1: {best_f1:.4f})")
val_preds = (probs_v >= best_tau).astype(int)

# Identify TP, FP, TN, FN
tp_indices = val_indices[(val_preds == 1) & (val_labels == 1)]
fp_indices = val_indices[(val_preds == 1) & (val_labels == 0)]
print(f"Validation TPs: {len(tp_indices)}, FPs: {len(fp_indices)}")

# 1. ATTRIBUTION & PERTURBATION ANALYSIS
# Measure edge necessity: drop each incoming edge, measure decrease in illicit probability
val_edges = val_data['edge_index'].numpy()
src_nodes, dst_nodes = val_edges[0], val_edges[1]

# Map incoming edges per node
from collections import defaultdict
in_edges = defaultdict(list)
for edge_id, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
    in_edges[d].append((edge_id, s))

# Evaluate sample of TPs and FPs for matched explanation audit
sample_tps = tp_indices[:50]
sample_fps = fp_indices[:50]

c1_c2_records = []

for cohort, nodes in [("TP", sample_tps), ("FP", sample_fps)]:
    for target_node in nodes:
        edges = in_edges[target_node]
        k = len(edges)
        if k == 0:
            continue
            
        orig_prob = probs_v[np.where(val_indices == target_node)[0][0]]
        
        # Test 1: Single Edge Removal (Necessity)
        necessity_drops = []
        for e_id, s_node in edges:
            mask = np.ones(val_edges.shape[1], dtype=bool)
            mask[e_id] = False
            sub_edges = torch.tensor(val_edges[:, mask], dtype=torch.long)
            
            with torch.no_grad():
                h_sub = m0.conv1(comb_v, sub_edges).relu()
                out_sub = m0.conv2(h_sub, sub_edges)
                p_sub = F.softmax(out_sub[target_node:target_node+1], dim=1)[0, 1].item()
                drop = orig_prob - p_sub
                necessity_drops.append((drop, e_id, s_node))
                
        # Top edge drop
        necessity_drops.sort(key=lambda x: x[0], reverse=True)
        top_drop = necessity_drops[0][0]
        
        # Test 2: Random Edge Removal Control
        rand_drop = np.mean([x[0] for x in necessity_drops])
        
        # Test 3: Sufficiency (Remove ALL edges EXCEPT top edge)
        top_e_id = necessity_drops[0][1]
        keep_mask = np.zeros(val_edges.shape[1], dtype=bool)
        keep_mask[top_e_id] = True
        # Keep non-target edges intact to preserve background graph
        keep_mask[dst_nodes != target_node] = True
        sub_edges_suff = torch.tensor(val_edges[:, keep_mask], dtype=torch.long)
        with torch.no_grad():
            h_suff = m0.conv1(comb_v, sub_edges_suff).relu()
            out_suff = m0.conv2(h_suff, sub_edges_suff)
            p_suff = F.softmax(out_suff[target_node:target_node+1], dim=1)[0, 1].item()
            sufficiency_score = p_suff / (orig_prob + 1e-8)

        # Test 4: Counterfactual Flip Search
        # Can dropping incoming edges flip prediction below tau*?
        all_target_e_ids = [x[1] for x in edges]
        cf_mask = np.ones(val_edges.shape[1], dtype=bool)
        for e_id in all_target_e_ids:
            cf_mask[e_id] = False
        sub_edges_cf = torch.tensor(val_edges[:, cf_mask], dtype=torch.long)
        with torch.no_grad():
            h_cf = m0.conv1(comb_v, sub_edges_cf).relu()
            out_cf = m0.conv2(h_cf, sub_edges_cf)
            p_cf = F.softmax(out_cf[target_node:target_node+1], dim=1)[0, 1].item()
            flipped = bool(p_cf < best_tau)

        c1_c2_records.append({
            "cohort": cohort,
            "node_id": target_node,
            "in_degree": k,
            "orig_prob": orig_prob,
            "top_edge_necessity_drop": top_drop,
            "random_edge_mean_drop": rand_drop,
            "sufficiency_ratio": sufficiency_score,
            "cf_all_edges_removed_prob": p_cf,
            "cf_flipped_to_licit": flipped,
            "delta_p_cf": orig_prob - p_cf
        })

df_xai = pd.DataFrame(c1_c2_records)
df_xai.to_csv(PHASE14_DIR / "PHASE14_C1_C2_INTERVENTION_RECORDS.csv", index=False)

# Summary of XAI Metrics by Cohort
xai_summary = df_xai.groupby("cohort").agg({
    "top_edge_necessity_drop": ["mean", "std"],
    "random_edge_mean_drop": ["mean", "std"],
    "sufficiency_ratio": ["mean", "std"],
    "cf_flipped_to_licit": ["mean"],
    "delta_p_cf": ["mean", "std"],
    "in_degree": ["mean"]
})
print("\n=== C1/C2 XAI INTERVENTION SUMMARY (TP vs FP) ===")
print(xai_summary)

# Test TP vs FP Difference in Top-Edge Necessity Drop
tp_drops = df_xai[df_xai["cohort"] == "TP"]["top_edge_necessity_drop"].to_numpy()
fp_drops = df_xai[df_xai["cohort"] == "FP"]["top_edge_necessity_drop"].to_numpy()
t_xai, p_xai = stats.ttest_ind(tp_drops, fp_drops, equal_var=False)
u_xai, p_mann = stats.mannwhitneyu(tp_drops, fp_drops)

print(f"\nTP vs FP Fidelity Difference: {tp_drops.mean() - fp_drops.mean():+.4f}")
print(f"Welch t-test: t = {t_xai:.3f}, p = {p_xai:.4f}")
print(f"Mann-Whitney U: U = {u_xai:.1f}, p = {p_mann:.4f}")

# Causal Gating Evaluation for RG-ETGNN
# Check if gate gamma correlates with edge drop Delta P
print("\n=== EVALUATING GATE ATTRIBUTION CAUSALITY (RG-ETGNN) ===")
from research_extension.phase14_results_analysis.diagnose_gates import ReliabilityAwareETGNNClassifier
m3_ckpt = torch.load("research_extension/phase13_advanced_investigation/checkpoints/M3_RG_ETGNN_CE_s42.pt", map_location="cpu", weights_only=False)
m3_model = ReliabilityAwareETGNNClassifier(enable_gate=True)
m3_model.load_state_dict(m3_ckpt["state_dict"])
m3_model.eval()

with torch.no_grad():
    out_m3, g_dict = m3_model(val_data['x'], val_data['edge_index'], val_data['time_step'], return_gates=True)
    g1 = g_dict['gates1'].numpy()

# Check correlation between Layer 1 gate value and edge necessity drop for sampled nodes
gate_intervention_pairs = []
for target_node in sample_tps[:20]:
    edges = in_edges[target_node]
    for e_id, s_node in edges:
        gate_val = g1[e_id] if e_id < len(g1) else 1.0
        # Compute edge drop
        mask = np.ones(val_edges.shape[1], dtype=bool)
        mask[e_id] = False
        sub_edges = torch.tensor(val_edges[:, mask], dtype=torch.long)
        with torch.no_grad():
            out_sub = m3_model(val_data['x'], sub_edges, val_data['time_step'])
            p_sub = F.softmax(out_sub[target_node:target_node+1], dim=1)[0, 1].item()
            p_orig = F.softmax(out_m3[target_node:target_node+1], dim=1)[0, 1].item()
            gate_intervention_pairs.append({"gate_value": gate_val, "necessity_drop": p_orig - p_sub})

df_gate_causal = pd.DataFrame(gate_intervention_pairs)
corr, p_corr = stats.pearsonr(df_gate_causal["gate_value"], df_gate_causal["necessity_drop"])
spear, p_spear = stats.spearmanr(df_gate_causal["gate_value"], df_gate_causal["necessity_drop"])
print(f"Gate Value vs Intervention Drop: Pearson r = {corr:.4f} (p={p_corr:.4f}), Spearman rho = {spear:.4f} (p={p_spear:.4f})")

print("\n[SUCCESS] Completed C1/C2 explainability evaluation.")

