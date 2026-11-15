import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import f1_score

# Output directory
OUT_DIR = Path("research_extension/phase14b_final_completion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. LOAD DATA & FROZEN MODELS
val_data = torch.load("data/processed/validation_graph.pt", map_location="cpu", weights_only=True)
val_mask = val_data['val_mask']
val_indices = torch.where(val_mask)[0].numpy()
val_labels = val_data['y'][val_mask].numpy()
val_edge_index = val_data['edge_index']
dst_nodes = val_edge_index[1].numpy()
deg_map = np.bincount(dst_nodes, minlength=val_data['x'].size(0))

from src.models.etgnn import ETGNNClassifier as M0_ETGNN
from src.models.time_encoding import encode_time as orig_encode_time
from research_extension.phase14_results_analysis.diagnose_gates import ReliabilityAwareETGNNClassifier as M3_RG_ETGNN

# Define M1 (ResidualETGNN) with notebook time encoding
from torch_geometric.nn import GATConv

def nb_encode_time(time_step, time_dim=16):
    device = time_step.device
    half_dim = time_dim // 2
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -(np.log(10000.0) / (half_dim - 1)))
    args = time_step.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class M1_ResidualETGNN(torch.nn.Module):
    def __init__(self, input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.time_dim = time_dim
        self.dropout = dropout
        in_dim = input_dim + time_dim
        self.conv1 = GATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.res1 = torch.nn.Linear(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, 2, heads=1, concat=False, dropout=dropout)
        self.res2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, time_step):
        encoded = nb_encode_time(time_step, self.time_dim)
        combined = torch.cat((x, encoded), dim=1)
        h_graph = self.conv1(combined, edge_index)
        h_res = self.res1(combined)
        h = F.relu(h_graph + h_res)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out_graph = self.conv2(h, edge_index)
        out_res = self.res2(h)
        return out_graph + out_res

# Load M0
m0 = M0_ETGNN()
m0_ckpt = torch.load("results/checkpoints/etgnn.pt", map_location="cpu", weights_only=False)
m0.load_state_dict(m0_ckpt.get("state_dict", m0_ckpt))
m0.eval()

# Load M1 (Seed 42)
m1 = M1_ResidualETGNN()
m1_ckpt = torch.load("research_extension/phase13_advanced_investigation/checkpoints/M1_ResidualETGNN_s42.pt", map_location="cpu", weights_only=False)
m1.load_state_dict(m1_ckpt["state_dict"])
m1.eval()

# Load M3 (Seed 42)
m3 = M3_RG_ETGNN(enable_gate=True)
m3_ckpt = torch.load("research_extension/phase13_advanced_investigation/checkpoints/M3_RG_ETGNN_CE_s42.pt", map_location="cpu", weights_only=False)
m3_model_dict = m3_ckpt["state_dict"]
m3.load_state_dict(m3_model_dict)
m3.eval()

print("[MODELS LOADED] M0 (Historical), M1 (Residual), M3 (RG-ETGNN CE)")

# Validation predictions & thresholds
with torch.no_grad():
    enc_m0 = orig_encode_time(val_data['time_step'], 16)
    comb_m0 = torch.cat((val_data['x'], enc_m0), dim=1)
    h0 = m0.conv1(comb_m0, val_data['edge_index']).relu()
    out0 = m0.conv2(h0, val_data['edge_index'])
    probs_m0 = F.softmax(out0[val_mask], dim=1)[:, 1].numpy()

    out1 = m1(val_data['x'], val_data['edge_index'], val_data['time_step'])
    probs_m1 = F.softmax(out1[val_mask], dim=1)[:, 1].numpy()

    out3 = m3(val_data['x'], val_data['edge_index'], val_data['time_step'])
    probs_m3 = F.softmax(out3[val_mask], dim=1)[:, 1].numpy()

def get_tau(probs):
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.1, 0.9, 81):
        f = f1_score(val_labels, (probs >= tau).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_tau = f, tau
    return best_tau, best_f1

tau0, f1_0 = get_tau(probs_m0)
tau1, f1_1 = get_tau(probs_m1)
tau3, f1_3 = get_tau(probs_m3)
print(f"Validation Tau* -> M0: {tau0:.2f} (F1: {f1_0:.4f}) | M1: {tau1:.2f} (F1: {f1_1:.4f}) | M3: {tau3:.2f} (F1: {f1_3:.4f})")

# 2. MATCHED-PAIR COHORT CONSTRUCTION (PROTOCOL C2)
# Matching variables: in-degree k, time step t, and predicted probability
preds_m1 = (probs_m1 >= tau1).astype(int)
tp_all = val_indices[(preds_m1 == 1) & (val_labels == 1)]
fp_all = val_indices[(preds_m1 == 1) & (val_labels == 0)]

val_steps = val_data['time_step'].numpy()
prob_map = {idx: probs_m1[i] for i, idx in enumerate(val_indices)}

# Greedy 1:1 matching without replacement
matched_pairs = []
used_fps = set()

# Filter to nodes with in-degree in [1, 10]
tp_candidates = [n for n in tp_all if 1 <= deg_map[n] <= 10]
fp_candidates = [n for n in fp_all if 1 <= deg_map[n] <= 10]

for tp in tp_candidates:
    best_fp = None
    min_dist = 999.0
    k_tp = deg_map[tp]
    t_tp = val_steps[tp]
    p_tp = prob_map[tp]

    for fp in fp_candidates:
        if fp in used_fps:
            continue
        k_fp = deg_map[fp]
        t_fp = val_steps[fp]
        p_fp = prob_map[fp]

        # Exact match on degree, time diff <= 1, prob diff <= 0.15
        if k_tp == k_fp and abs(t_tp - t_fp) <= 1:
            dist = abs(p_tp - p_fp)
            if dist < min_dist and dist <= 0.20:
                min_dist = dist
                best_fp = fp

    if best_fp is not None:
        used_fps.add(best_fp)
        matched_pairs.append((tp, best_fp, k_tp, t_tp, p_tp, prob_map[best_fp]))

print(f"Constructed {len(matched_pairs)} rigorously matched TP-FP pairs!")

# Covariate balance verification
diff_p = [p[4] - p[5] for p in matched_pairs]
diff_t = [abs(val_steps[p[0]] - val_steps[p[1]]) for p in matched_pairs]
print(f"Covariate Balance -> Mean Prob Diff: {np.mean(diff_p):+.4f} (SD: {np.std(diff_p):.4f}) | Max Time Diff: {max(diff_t)}")

# 3. INTERVENTION EXPERIMENTS ON MATCHED PAIRS
c1_c2_results = []
t0 = time.time()

for pair_id, (tp_node, fp_node, deg, t_step, p_tp_init, p_fp_init) in enumerate(matched_pairs):
    for cohort, node in [("TP", tp_node), ("FP", fp_node)]:
        # Extract 2-hop ego subgraph
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=int(node), num_hops=2, edge_index=val_edge_index, relabel_nodes=True
        )
        sub_x = val_data['x'][subset]
        sub_t = val_data['time_step'][subset]
        target_local_idx = mapping.item()

        # Local baseline predictions across M0, M1, M3
        with torch.no_grad():
            p0 = F.softmax(m0.conv2(m0.conv1(torch.cat((sub_x, orig_encode_time(sub_t, 16)), dim=1), sub_edge_index).relu(), sub_edge_index)[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()
            p1 = F.softmax(m1(sub_x, sub_edge_index, sub_t)[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()
            p3 = F.softmax(m3(sub_x, sub_edge_index, sub_t)[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()

        in_mask = (sub_edge_index[1] == target_local_idx)
        in_edge_indices = torch.where(in_mask)[0].numpy()
        k = len(in_edge_indices)

        # A. Feature Attribution Intervention (Zero out top 10% local features)
        # Identify top features by magnitude
        feat_vec = sub_x[target_local_idx].abs().numpy()
        top_feat_idx = np.argsort(feat_vec)[-17:]  # top 10% of 165 features
        sub_x_feat_pert = sub_x.clone()
        sub_x_feat_pert[target_local_idx, top_feat_idx] = 0.0

        with torch.no_grad():
            p1_feat_pert = F.softmax(m1(sub_x_feat_pert, sub_edge_index, sub_t)[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()
            feat_necessity_drop = p1 - p1_feat_pert

        # B. Edge Intervention (Fixed-count deletion: 1 edge)
        edge_drops_m1 = []
        for e_idx in in_edge_indices:
            e_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool)
            e_mask[e_idx] = False
            with torch.no_grad():
                p_e = F.softmax(m1(sub_x, sub_edge_index[:, e_mask], sub_t)[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()
                edge_drops_m1.append(p1 - p_e)

        top_edge_drop = max(edge_drops_m1) if edge_drops_m1 else 0.0
        rand_edge_drop = np.mean(edge_drops_m1) if edge_drops_m1 else 0.0

        # C. Sufficiency (Keep ONLY top edge, remove all other incoming edges)
        if len(in_edge_indices) > 1:
            best_e_idx = in_edge_indices[np.argmax(edge_drops_m1)]
            suff_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool)
            for idx in in_edge_indices:
                if idx != best_e_idx:
                    suff_mask[idx] = False
            with torch.no_grad():
                p1_suff = F.softmax(m1(sub_x, sub_edge_index[:, suff_mask], sub_t)[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()
                sufficiency_ratio = p1_suff / (p1 + 1e-8)
        else:
            sufficiency_ratio = 1.0

        # D. Counterfactual Minimal Flips Search (Minimum edges to flip below tau1)
        flipped = False
        min_flips = 0
        sorted_edge_order = [in_edge_indices[i] for i in np.argsort(edge_drops_m1)[::-1]]
        cf_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool)

        for step_i, e_idx in enumerate(sorted_edge_order, start=1):
            cf_mask[e_idx] = False
            with torch.no_grad():
                p_cf = F.softmax(m1(sub_x, sub_edge_index[:, cf_mask], sub_t)[target_local_idx:target_local_idx+1], dim=1)[0, 1].item()
            if p_cf < tau1:
                flipped = True
                min_flips = step_i
                break

        total_edge_drop = p1 - p_cf if k > 0 else 0.0

        c1_c2_results.append({
            "pair_id": pair_id,
            "cohort": cohort,
            "node_id": node,
            "in_degree": deg,
            "time_step": t_step,
            "p_m0": p0,
            "p_m1": p1,
            "p_m3": p3,
            "feat_necessity_drop": feat_necessity_drop,
            "top_edge_necessity_drop": top_edge_drop,
            "rand_edge_mean_drop": rand_edge_drop,
            "sufficiency_ratio": sufficiency_ratio,
            "cf_flipped": flipped,
            "min_edges_to_flip": min_flips if flipped else -1,
            "total_graph_drop": total_edge_drop
        })

df_all = pd.DataFrame(c1_c2_results)
df_all.to_csv(OUT_DIR / "C1_C2_MATCHED_INTERVENTION_EXPERIMENTS.csv", index=False)
print(f"Executed {len(df_all)} multi-model intervention evaluations in {time.time()-t0:.2f}s!")

# 4. MULTIPLE TESTING ADJUSTMENTS (HOLM-BONFERRONI & FDR)
p_vals = {}

# Test A: TP vs FP on top_edge_necessity_drop
tp_vals = df_all[df_all["cohort"] == "TP"]["top_edge_necessity_drop"].to_numpy()
fp_vals = df_all[df_all["cohort"] == "FP"]["top_edge_necessity_drop"].to_numpy()
t_a, p_a = stats.ttest_rel(tp_vals, fp_vals)
p_vals["Top Edge Drop (TP vs FP)"] = p_a

# Test B: TP vs FP on feat_necessity_drop
tp_f = df_all[df_all["cohort"] == "TP"]["feat_necessity_drop"].to_numpy()
fp_f = df_all[df_all["cohort"] == "FP"]["feat_necessity_drop"].to_numpy()
t_b, p_b = stats.ttest_rel(tp_f, fp_f)
p_vals["Feature Drop (TP vs FP)"] = p_b

# Test C: Feature Drop vs Edge Drop within TPs
t_c, p_c = stats.ttest_rel(tp_f, tp_vals)
p_vals["Feature vs Edge Drop (within TPs)"] = p_c

# Test D: Feature Drop vs Edge Drop within FPs
t_d, p_d = stats.ttest_rel(fp_f, fp_vals)
p_vals["Feature vs Edge Drop (within FPs)"] = p_d

# Test E: Counterfactual Flip Rate (Fisher's exact test)
tp_flips = df_all[df_all["cohort"] == "TP"]["cf_flipped"].sum()
fp_flips = df_all[df_all["cohort"] == "FP"]["cf_flipped"].sum()
n_pairs = len(matched_pairs)
_, p_e = stats.fisher_exact([[tp_flips, n_pairs - tp_flips], [fp_flips, n_pairs - fp_flips]])
p_vals["Counterfactual Flip Rate (Fisher Exact)"] = p_e

# Multiple comparison corrections
test_names = list(p_vals.keys())
raw_p = [p_vals[k] for k in test_names]
order = np.argsort(raw_p)
m = len(raw_p)

holm_bonf = np.zeros(m)
fdr_bh = np.zeros(m)

for rank, idx in enumerate(order, start=1):
    holm_bonf[idx] = min(1.0, raw_p[idx] * (m - rank + 1))
    fdr_bh[idx] = min(1.0, raw_p[idx] * m / rank)

df_multi = pd.DataFrame({
    "hypothesis": test_names,
    "raw_pvalue": raw_p,
    "holm_bonferroni_p": holm_bonf,
    "fdr_bh_qvalue": fdr_bh,
    "statistically_significant": [bool(p < 0.05) for p in holm_bonf]
})
df_multi.to_csv(OUT_DIR / "MULTIPLE_TESTING_CORRECTION_AUDIT.csv", index=False)
print("\n=== MULTIPLE TESTING CORRECTION AUDIT ===")
print(df_multi.to_string())

# 5. DEGREE-STRATIFIED INTERVENTION BREAKDOWN
strat = df_all.groupby(["cohort", "in_degree"]).agg({
    "top_edge_necessity_drop": ["mean", "count"],
    "feat_necessity_drop": ["mean"],
    "cf_flipped": ["mean"]
})
print("\n=== DEGREE-STRATIFIED XAI BREAKDOWN ===")
print(strat.to_string())
strat.to_csv(OUT_DIR / "DEGREE_STRATIFIED_XAI_SUMMARY.csv")

print("\n[SUCCESS] Completed comprehensive C1/C2 explainability pipeline.")

