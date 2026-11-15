"""
verify_full_integration_evidence.py
===================================
Comprehensive, authoritative verification script validating the integration of
ETGNN C1 + C2 results against frozen configuration and raw intervention records.

Checks:
1. Frozen configuration alignment:
   - 377 unique execution nodes, valid time steps (37..40), candidate edges >= 2.
   - C1 register: 240 transactions.
   - C2 registers: 420 pair records across 8 overlapping registers.
2. Duplicate key integrity:
   - node_index is 100% unique in summary and C1 datasets.
   - pair_id is 100% unique in C2 paired difference records.
3. Configured ID and register membership:
   - Exact one-to-one match between summary nodes and configured execution union.
   - Exact one-to-one match between C1 nodes and configured C1 register.
   - Exact match between all 420 configured C2 pairs and saved pair difference records.
4. Deletion budget rule integrity:
   - Every transaction verifies k = max(1, ceil(0.25 * E_cand)) across all 377 nodes.
5. Numerical completeness and structurally absent runtime values:
   - Structurally absent runtime_ms values for random comparator draws verified as expected.
   - Zero nulls, zero Infs across all core non-runtime numeric columns.
   - All probabilities verified within [0.0, 1.0].
6. C1 shared intervention integrity:
   - Shared ETGNN ranking and shared target class across both scoring models.
   - Exact identical deleted edge sets verified between ETGNN and Matched GAT for all 240 C1 nodes.
7. C2 own ranking and target class integrity:
   - Matched GAT uses own ranking and own target class in C2.
8. Reconstruct EVERY C2 pair difference from detailed intervention records:
   - All 420 pair differences reconstructed from scratch with explicit tolerance < 1e-10.
9. Exact summary statistic and cluster bootstrap recomputations:
   - C1 delta_adj and delta_raw statistics verified.
   - All 8 C2 bootstrap point estimates, standard errors, and 95% CIs verified (B=2,000, seed 20261020).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".").resolve()
CONFIG_PATH = PROJECT_ROOT / "configs" / "main_c1_c2_run_config.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "main_c1_c2_results"
DERIVED_DIR = RESULTS_DIR / "derived"

print("================================================================================")
print("  AUTHORITATIVE VERIFICATION OF C1 + C2 RESULTS INTEGRATION & INTEGRITY")
print("================================================================================")

# 1. Load frozen configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg_nodes = cfg["execution_node_union"]["nodes"]
assert len(cfg_nodes) == 377
df_cfg = pd.DataFrame(cfg_nodes)
assert df_cfg["node_index"].nunique() == 377
assert df_cfg["transaction_id"].nunique() == 377
assert set(df_cfg["time_step"].unique()).issubset({37, 38, 39, 40})
assert (df_cfg["candidate_edge_count"] >= 2).all()
print("[PASS] 1. Frozen configuration verified: 377 unique nodes in time steps 37..40, all E_cand >= 2.")

# Check C1 and C2 definitions in config
c1_cfg_nodes = [n["node_index"] for n in cfg["c1_sample_register"]["nodes"]]
assert len(c1_cfg_nodes) == 240
assert len(set(c1_cfg_nodes)) == 240
assert set(c1_cfg_nodes).issubset(set(df_cfg["node_index"]))
print("[PASS] 2. C1 sample register verified: 240 unique transactions, strict subset of 377 execution nodes.")

c2_regs = cfg["c2_pair_registers"]
c2_pair_counts = {}
all_cfg_pairs = []
for spec_key, spec in c2_regs.items():
    for model_key, model_dict in spec.items():
        for pair_type in ["tp_fp_pairs", "tn_fn_pairs"]:
            plist = model_dict[pair_type]
            name = f"{spec_key}_{model_key}_{pair_type}"
            c2_pair_counts[name] = len(plist)
            all_cfg_pairs.extend(plist)

total_pairs_cfg = len(all_cfg_pairs)
assert total_pairs_cfg == 420
print(f"[PASS] 3. C2 pair registers verified: 420 pair records across 8 registers in frozen config: {c2_pair_counts}")

# 2. Check resume state
resume_file = RESULTS_DIR / "main_resume_state.json"
assert resume_file.exists()
with open(resume_file, "r", encoding="utf-8") as f:
    r_state = json.load(f)
assert r_state["status"] == "COMPLETED"
assert r_state["completed_count"] == 377
assert len(r_state["completed_node_indices"]) == 377
assert set(r_state["completed_node_indices"]) == set(df_cfg["node_index"])
print("[PASS] 4. Main resume state verified: status=COMPLETED, completed_count=377.")

# 3. Check saved data files
summary_df = pd.read_csv(RESULTS_DIR / "main_benchmark_summary.csv")
raw_df = pd.read_csv(RESULTS_DIR / "raw_benchmark_records.csv")
det_df = pd.read_csv(RESULTS_DIR / "detailed_intervention_records.csv")
c1_df = pd.read_csv(RESULTS_DIR / "c1_statistical_results.csv")
pair_diff_df = pd.read_csv(RESULTS_DIR / "c2_individual_pair_differences.csv")
boot_df = pd.read_csv(RESULTS_DIR / "c2_bootstrap_results.csv")

assert len(summary_df) == 377
assert len(raw_df) == 13195  # 377 * 35
assert len(det_df) == 39585  # 377 * 105
assert len(c1_df) == 240
assert len(pair_diff_df) == 420
assert len(boot_df) == 8
print("[PASS] 5. Exact row counts verified: summary=377, raw=13195, detailed=39585, c1=240, pair_diff=420, boot=8.")

# 4. Check duplicate keys & exact ID membership
assert summary_df["node_index"].is_unique, "Duplicate node_index in summary_df!"
assert summary_df["node_index"].tolist() == [n["node_index"] for n in cfg_nodes]
assert c1_df["node_index"].is_unique, "Duplicate node_index in c1_df!"
assert set(c1_df["node_index"]) == set(c1_cfg_nodes)
assert pair_diff_df["pair_id"].is_unique, "Duplicate pair_id in pair_diff_df!"
assert set(pair_diff_df["pair_id"]) == set([p["pair_id"] for p in all_cfg_pairs])
print("[PASS] 6. Duplicate keys & exact configured ID membership verified: 100% unique keys and exact 1-to-1 mapping.")

# 5. Check deletion budgets across every single node
for idx, row in summary_df.iterrows():
    expected_k = max(1, int(np.ceil(0.25 * row["E_cand"])))
    assert row["k_budget"] == expected_k, f"Budget mismatch at node {row['node_index']}: expected {expected_k}, got {row['k_budget']}"
    assert abs(row["actual_k_over_E"] - (expected_k / row["E_cand"])) < 1e-3

ec_counts = summary_df["E_cand"].value_counts().to_dict()
assert ec_counts[2] == 317, f"Expected 317 nodes with E_cand=2, found {ec_counts.get(2)}"
assert (summary_df[summary_df["E_cand"] == 2]["k_budget"] == 1).all()
assert (summary_df[summary_df["E_cand"] == 2]["actual_k_over_E"] == 0.5).all()
print("[PASS] 7. Budget rule verified: k = max(1, ceil(0.25 * E_cand)) holds across 100% of 377 nodes (317 have E_cand=2, k=1, actual frac=50.0%).")

# 6. Check finite values, allowed structurally missing runtime fields, and probabilities
for df_name, df_obj in [("summary", summary_df), ("raw", raw_df), ("detailed", det_df), ("c1", c1_df), ("pair_diff", pair_diff_df)]:
    core_cols = [c for c in df_obj.select_dtypes(include=[np.number]).columns if not c.startswith("runtime")]
    assert not df_obj[core_cols].isnull().any().any(), f"Core column null found in {df_name}"
    assert not np.isinf(df_obj[core_cols].to_numpy()).any(), f"Inf found in {df_name}"

# Structurally absent runtime values for random comparator draws
assert raw_df["runtime_ms_etgnn"].isnull().sum() == 11310  # 377 * 30 random draws
assert raw_df["runtime_ms_gat"].isnull().sum() == 12064    # random draws + secondary diagnostics
assert det_df["runtime_ms"].isnull().sum() == 33930        # random comparator draws across models

prob_cols = ["orig_prob_target", "del_prob_target", "orig_p0", "orig_p1", "del_p0", "del_p1"]
for pc in prob_cols:
    assert ((det_df[pc] >= 0.0) & (det_df[pc] <= 1.0)).all(), f"Probability out of bounds in {pc}"
print("[PASS] 8. Numerical completeness verified: zero nulls/Infs in all core metrics, all probabilities in [0.0, 1.0], and structurally absent runtime fields accounted for.")

# 7. Check C1 Shared Intervention & Deleted-Edge Sets
cfg_node_map = {n["node_index"]: n for n in cfg_nodes}
c1_det = det_df[det_df["contrast"] == "C1"]
c1_ge = c1_det[c1_det["method"] == "GNNExplainer"]
assert (c1_ge["ranking_model_id"] == "ETGNN").all(), "GNNExplainer ranking in C1 not ETGNN!"
assert not (c1_det["ranking_model_id"] == "Matched_GAT").any()

# For each node in C1, target_class must match configured ETGNN target class identically
for nid, g in c1_det.groupby("node_index"):
    exp_tgt = cfg_node_map[nid]["c1_target_class_etgnn"]
    assert (g["target_class"] == exp_tgt).all(), f"Node {nid} has mismatched target class in C1"

# In C1, ETGNN and Matched GAT must receive the exact identical deleted edge set for every node and seed
for (nid, seed), g in c1_ge.groupby(["node_index", "seed"]):
    e_et = g[g["scoring_model_id"] == "ETGNN"]["deleted_global_edge_ids"].iloc[0]
    e_gat = g[g["scoring_model_id"] == "Matched_GAT"]["deleted_global_edge_ids"].iloc[0]
    assert e_et == e_gat, f"C1 deleted-edge mismatch at node {nid}, seed {seed}: {e_et} vs {e_gat}"
print("[PASS] 9. C1 shared intervention verified: shared ETGNN ranking, identical target class, and exact identical deleted-edge sets across all seeds.")

# 8. Check C2 Own Ranking & Target Class Integrity
c2_det = det_df[det_df["contrast"] == "C2"]
c2_gat_ge = c2_det[(c2_det["method"] == "GNNExplainer") & (c2_det["scoring_model_id"] == "Matched_GAT")]
assert (c2_gat_ge["ranking_model_id"] == "Matched_GAT").all()
for nid, g in c2_det.groupby("node_index"):
    exp_tgt = cfg_node_map[nid]["c2_target_class_gat"]
    assert (g["target_class"] == exp_tgt).all(), f"Node {nid} has mismatched target class in C2"
print("[PASS] 10. C2 own ranking verified: Matched GAT uses own GNNExplainer ranking and own configured target class.")

# 9. Reconstruct EVERY C2 Pair Difference from Detailed Intervention Records
adv_map = {}
# ETGNN advantages: from C1 detailed rows (ranking=ETGNN, scoring=ETGNN vs ranking=Random, scoring=ETGNN)
for nid, g in det_df[(det_df["contrast"] == "C1") & (det_df["scoring_model_id"] == "ETGNN")].groupby("node_index"):
    ge = g[g["method"] == "GNNExplainer"]["signed_drop"].mean()
    rnd = g[g["method"] == "Random"]["signed_drop"].mean()
    adv_map[(nid, "ETGNN")] = ge - rnd

# Matched GAT advantages: from C2 detailed rows (ranking=Matched_GAT, scoring=Matched_GAT vs ranking=Random, scoring=Matched_GAT)
for nid, g in det_df[(det_df["contrast"] == "C2") & (det_df["scoring_model_id"] == "Matched_GAT")].groupby("node_index"):
    ge = g[g["method"] == "GNNExplainer"]["signed_drop"].mean()
    rnd = g[g["method"] == "Random"]["signed_drop"].mean()
    adv_map[(nid, "Matched_GAT")] = ge - rnd

max_recon_diff = 0.0
for idx, row in pair_diff_df.iterrows():
    reg = row["register_name"]
    model = "ETGNN" if "ETGNN" in reg else "Matched_GAT"
    treat_nid = row["treat_node_index"]
    ctrl_nid = row["ctrl_node_index"]

    calc_treat_adv = adv_map[(treat_nid, model)]
    calc_ctrl_adv = adv_map[(ctrl_nid, model)]
    calc_diff = calc_treat_adv - calc_ctrl_adv

    diff_t = abs(calc_treat_adv - row["treat_advantage"])
    diff_c = abs(calc_ctrl_adv - row["ctrl_advantage"])
    diff_p = abs(calc_diff - row["pair_difference"])

    max_recon_diff = max(max_recon_diff, diff_t, diff_c, diff_p)
    assert diff_t < 1e-10, f"Treat advantage reconstruction error at {row['pair_id']}: {diff_t}"
    assert diff_c < 1e-10, f"Ctrl advantage reconstruction error at {row['pair_id']}: {diff_c}"
    assert diff_p < 1e-10, f"Pair difference reconstruction error at {row['pair_id']}: {diff_p}"

print(f"[PASS] 11. Full C2 reconstruction verified: all 420 pair differences reconstructed from detailed records (max diff = {max_recon_diff:.2e} < 1e-10).")

# 10. Check C1 Summary Recomputation
recomp_c1 = []
for nid, g in det_df[det_df["node_index"].isin(c1_cfg_nodes) & (det_df["contrast"] == "C1")].groupby("node_index"):
    ge_et = g[(g["ranking_model_id"] == "ETGNN") & (g["scoring_model_id"] == "ETGNN")]["signed_drop"].mean()
    rnd_et = g[(g["ranking_model_id"] == "Random") & (g["scoring_model_id"] == "ETGNN")]["signed_drop"].mean()
    adv_et = ge_et - rnd_et

    ge_gt = g[(g["ranking_model_id"] == "ETGNN") & (g["scoring_model_id"] == "Matched_GAT")]["signed_drop"].mean()
    rnd_gt = g[(g["ranking_model_id"] == "Random") & (g["scoring_model_id"] == "Matched_GAT")]["signed_drop"].mean()
    adv_gt = ge_gt - rnd_gt

    recomp_c1.append({
        "node_index": nid,
        "delta_adj": adv_et - adv_gt,
        "delta_raw": ge_et - ge_gt
    })
df_rc1 = pd.DataFrame(recomp_c1).sort_values("node_index").reset_index(drop=True)
df_c1_sorted = c1_df.sort_values("node_index").reset_index(drop=True)
assert np.allclose(df_rc1["delta_adj"], df_c1_sorted["delta_adj"], atol=1e-10)
assert np.allclose(df_rc1["delta_raw"], df_c1_sorted["delta_raw"], atol=1e-10)

mean_adj = float(df_rc1["delta_adj"].mean())
mean_raw = float(df_rc1["delta_raw"].mean())
assert abs(mean_adj - 0.002903) < 1e-5
assert abs(mean_raw - 0.038491) < 1e-5
print(f"[PASS] 12. C1 statistics recomputation verified: delta_adj mean = {mean_adj:.6f} (~0.002903), delta_raw mean = {mean_raw:.6f} (~0.038491).")

# 11. Check C2 Bootstrap Recomputation (Seed 20261020, B=2000)
for idx, r in boot_df.iterrows():
    reg_name = r["register"]
    sub_pairs = pair_diff_df[pair_diff_df["register_name"] == reg_name]
    diffs = sub_pairs["pair_difference"].to_numpy()
    n_p = len(diffs)

    rng = np.random.default_rng(20261020)
    boot_means = [np.mean(diffs[rng.choice(n_p, size=n_p, replace=True)]) for _ in range(2000)]

    recomp_mean = float(np.mean(diffs))
    recomp_se = float(np.std(boot_means, ddof=1))
    recomp_low = float(np.percentile(boot_means, 2.5))
    recomp_high = float(np.percentile(boot_means, 97.5))

    assert abs(recomp_mean - r["mean_diff"]) < 1e-10
    assert abs(recomp_se - r["bootstrap_se"]) < 1e-10
    assert abs(recomp_low - r["ci_95_low"]) < 1e-10
    assert abs(recomp_high - r["ci_95_high"]) < 1e-10

print("[PASS] 13. C2 cluster bootstrap recomputation verified: 100% exact match across all 8 registers with B=2,000, seed 20261020.")

# 12. Verify Key Reference Estimates Specifically:
# ETGNN primary FP-TP: mean 0.039343, CI [0.020234, 0.057410]
r_et_prim = boot_df[boot_df["register"] == "ETGNN Primary TP-FP"].iloc[0]
assert abs(r_et_prim["mean_diff"] - 0.039343) < 1e-5
assert abs(r_et_prim["ci_95_low"] - 0.020234) < 1e-5
assert abs(r_et_prim["ci_95_high"] - 0.057410) < 1e-5

# ETGNN degree sensitivity FP-TP: mean 0.043455, CI [0.019882, 0.069552]
r_et_deg = boot_df[boot_df["register"] == "ETGNN Degree-Sensitivity TP-FP"].iloc[0]
assert abs(r_et_deg["mean_diff"] - 0.043455) < 1e-5
assert abs(r_et_deg["ci_95_low"] - 0.019882) < 1e-5
assert abs(r_et_deg["ci_95_high"] - 0.069552) < 1e-5

# GAT primary FP-TP: mean 0.020315, CI [0.003917, 0.038684]
r_gat_prim = boot_df[boot_df["register"] == "Matched GAT Primary TP-FP"].iloc[0]
assert abs(r_gat_prim["mean_diff"] - 0.020315) < 1e-5
assert abs(r_gat_prim["ci_95_low"] - 0.003917) < 1e-5
assert abs(r_gat_prim["ci_95_high"] - 0.038684) < 1e-5

# GAT degree sensitivity FP-TP: mean +0.010547, CI [-0.013240, 0.032770]
r_gat_deg = boot_df[boot_df["register"] == "Matched GAT Degree-Sensitivity TP-FP"].iloc[0]
assert abs(r_gat_deg["mean_diff"] - 0.010547) < 1e-5
assert abs(r_gat_deg["ci_95_low"] - (-0.013240)) < 1e-5
assert abs(r_gat_deg["ci_95_high"] - 0.032770) < 1e-5

print("[PASS] 14. Reference estimates verified: ETGNN primary (+0.0393 [0.0202, 0.0574]), ETGNN degree (+0.0435 [0.0199, 0.0696]), GAT primary (+0.0203 [0.0039, 0.0387]), GAT degree (+0.0105 [-0.0132, 0.0328]).")
print("\n================================================================================")
print("  ALL VERIFICATION CHECKS PASSED WITH 100% MATHEMATICAL PRECISION.")
print("================================================================================")
