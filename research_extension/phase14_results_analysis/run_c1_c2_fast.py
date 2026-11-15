import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch_geometric.utils import k_hop_subgraph

from src.models.etgnn import ETGNNClassifier as OrigETGNN
from src.models.time_encoding import encode_time as orig_encode_time

val_data = torch.load("data/processed/validation_graph.pt", map_location="cpu", weights_only=True)
val_mask = val_data['val_mask']
val_indices = torch.where(val_mask)[0].numpy()
val_labels = val_data['y'][val_mask].numpy()

m0 = OrigETGNN()
ckpt = torch.load("results/checkpoints/etgnn.pt", map_location="cpu", weights_only=False)
m0.load_state_dict(ckpt.get("state_dict", ckpt))
m0.eval()

enc_v = orig_encode_time(val_data['time_step'], 16)
comb_v = torch.cat((val_data['x'], enc_v), dim=1)

with torch.no_grad():
    h_v = m0.conv1(comb_v, val_data['edge_index']).relu()
    out_v = m0.conv2(h_v, val_data['edge_index'])
    probs_v = F.softmax(out_v[val_mask], dim=1)[:, 1].numpy()

# Find validation optimal threshold tau*
best_tau = 0.5
best_f1 = -1.0
from sklearn.metrics import f1_score
for tau in np.linspace(0.1, 0.9, 81):
    pred = (probs_v >= tau).astype(int)
    f1 = f1_score(val_labels, pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_tau = tau

val_preds = (probs_v >= best_tau).astype(int)
tp_indices = val_indices[(val_preds == 1) & (val_labels == 1)]
fp_indices = val_indices[(val_preds == 1) & (val_labels == 0)]
print(f"Validation Tau*: {best_tau:.2f} (F1: {best_f1:.4f}) | TPs: {len(tp_indices)}, FPs: {len(fp_indices)}")

# Degree filtering: select nodes with degree 1 <= k <= 10
val_edge_index = val_data['edge_index']
dst_nodes = val_edge_index[1].numpy()
deg_map = np.bincount(dst_nodes, minlength=val_data['x'].size(0))

tp_selected = [n for n in tp_indices if 1 <= deg_map[n] <= 10][:40]
fp_selected = [n for n in fp_indices if 1 <= deg_map[n] <= 10][:40]
print(f"Selected {len(tp_selected)} TPs and {len(fp_selected)} FPs with in-degree in [1, 10]")

records = []
t0 = time.time()

for cohort, nodes in [("TP", tp_selected), ("FP", fp_selected)]:
    for target_node in nodes:
        # Extract 2-hop ego subgraph
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=int(target_node), num_hops=2, edge_index=val_edge_index, relabel_nodes=True
        )
        sub_x = comb_v[subset]
        target_local_idx = mapping.item()

        # Full 2-hop baseline probability
        with torch.no_grad():
            h_sub = m0.conv1(sub_x, sub_edge_index).relu()
            out_sub = m0.conv2(h_sub, sub_edge_index)
            p_orig = F.softmax(out_sub[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()

        # Incoming edges directly to target_node
        in_mask = (sub_edge_index[1] == target_local_idx)
        in_edge_indices = torch.where(in_mask)[0].numpy()
        k = len(in_edge_indices)
        if k == 0:
            continue

        # Single edge removal (necessity)
        drops = []
        for e_idx in in_edge_indices:
            e_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool)
            e_mask[e_idx] = False
            sub_e_pert = sub_edge_index[:, e_mask]
            with torch.no_grad():
                h_pert = m0.conv1(sub_x, sub_e_pert).relu()
                out_pert = m0.conv2(h_pert, sub_e_pert)
                p_pert = F.softmax(out_pert[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()
                drops.append(p_orig - p_pert)

        top_drop = max(drops) if drops else 0.0
        mean_drop = np.mean(drops) if drops else 0.0

        # Remove ALL direct incoming edges (counterfactual total graph ablation)
        no_in_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool)
        no_in_mask[in_mask] = False
        sub_e_no_in = sub_edge_index[:, no_in_mask]
        with torch.no_grad():
            h_no_in = m0.conv1(sub_x, sub_e_no_in).relu()
            out_no_in = m0.conv2(h_no_in, sub_e_no_in)
            p_no_in = F.softmax(out_no_in[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()
            flipped = bool(p_no_in < best_tau)
            total_drop = p_orig - p_no_in

        records.append({
            "cohort": cohort, "node_id": target_node, "in_degree": k,
            "orig_prob": p_orig, "top_edge_necessity_drop": top_drop,
            "random_edge_mean_drop": mean_drop,
            "cf_all_edges_removed_prob": p_no_in, "cf_flipped": flipped,
            "total_graph_drop": total_drop
        })

print(f"Computed {len(records)} ego-subgraph interventions in {time.time()-t0:.2f}s!")
df_xai = pd.DataFrame(records)
PHASE14_DIR = Path("research_extension/phase14_results_analysis")
df_xai.to_csv(PHASE14_DIR / "PHASE14_C1_C2_INTERVENTION_RECORDS.csv", index=False)

summary = df_xai.groupby("cohort").agg({
    "top_edge_necessity_drop": ["mean", "std"],
    "random_edge_mean_drop": ["mean", "std"],
    "cf_flipped": ["mean"],
    "total_graph_drop": ["mean", "std"],
    "in_degree": ["mean"]
})
print("\n=== C1/C2 XAI INTERVENTION SUMMARY ===")
print(summary)

# Test TP vs FP Difference
tp_drops = df_xai[df_xai["cohort"] == "TP"]["top_edge_necessity_drop"].to_numpy()
fp_drops = df_xai[df_xai["cohort"] == "FP"]["top_edge_necessity_drop"].to_numpy()
t_stat, p_val = stats.ttest_ind(tp_drops, fp_drops, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(tp_drops, fp_drops)

print(f"\nTP vs FP Mean Difference: {tp_drops.mean() - fp_drops.mean():+.4f}")
print(f"Welch t-test: t = {t_stat:+.3f}, p = {p_val:.4f}")
print(f"Mann-Whitney U: U = {u_stat:.1f}, p = {u_pval:.4f}")

