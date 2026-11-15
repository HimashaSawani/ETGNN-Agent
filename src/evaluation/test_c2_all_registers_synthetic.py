#!/usr/bin/env python3
"""
test_c2_all_registers_synthetic.py
==================================
Synthetic validation of Contrast 2 matched-pair cluster bootstrap across
all 8 primary and degree-sensitivity registers in configs/main_c1_c2_run_config.json.

Asserts:
1. Complete pair membership across all 8 registers against the 377 execution union.
2. Zero missing nodes in any register.
3. Individual pair differences are computed and saved with pair IDs.
4. Matched-pair cluster bootstrap (B=2,000) executes cleanly across all 8 registers.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

with open(PROJECT_ROOT / "configs" / "main_c1_c2_run_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

union_nodes = [n["node_index"] for n in config["execution_node_union"]["nodes"]]
assert len(union_nodes) == 377, f"Expected 377 union nodes, found {len(union_nodes)}"

# Generate deterministic synthetic advantages for all 377 nodes
rng = np.random.default_rng(20261020)
synth_adv_et = {nid: float(rng.normal(0.05, 0.15)) for nid in union_nodes}
synth_adv_gt = {nid: float(rng.normal(0.02, 0.15)) for nid in union_nodes}

c2_registers = [
    ("ETGNN Primary TP-FP", config["c2_pair_registers"]["primary_caliper_matches"]["etgnn_full"]["tp_fp_pairs"], "ETGNN"),
    ("ETGNN Primary TN-FN", config["c2_pair_registers"]["primary_caliper_matches"]["etgnn_full"]["tn_fn_pairs"], "ETGNN"),
    ("Matched GAT Primary TP-FP", config["c2_pair_registers"]["primary_caliper_matches"]["matched_gat"]["tp_fp_pairs"], "Matched_GAT"),
    ("Matched GAT Primary TN-FN", config["c2_pair_registers"]["primary_caliper_matches"]["matched_gat"]["tn_fn_pairs"], "Matched_GAT"),
    ("ETGNN Degree-Sensitivity TP-FP", config["c2_pair_registers"]["degree_sensitivity_matches"]["etgnn_full"]["tp_fp_pairs"], "ETGNN"),
    ("ETGNN Degree-Sensitivity TN-FN", config["c2_pair_registers"]["degree_sensitivity_matches"]["etgnn_full"]["tn_fn_pairs"], "ETGNN"),
    ("Matched GAT Degree-Sensitivity TP-FP", config["c2_pair_registers"]["degree_sensitivity_matches"]["matched_gat"]["tp_fp_pairs"], "Matched_GAT"),
    ("Matched GAT Degree-Sensitivity TN-FN", config["c2_pair_registers"]["degree_sensitivity_matches"]["matched_gat"]["tn_fn_pairs"], "Matched_GAT"),
]

individual_pair_diffs = []
c2_outcomes = []

print("=== TESTING ALL 8 PAIR REGISTERS ON 377-NODE EXECUTION UNION ===")
for reg_name, pairs_list, model_type in c2_registers:
    adv_dict = synth_adv_et if model_type == "ETGNN" else synth_adv_gt
    pair_diffs = []

    for p_item in pairs_list:
        pid = p_item["pair_id"]
        u = int(p_item["treat_node_index"])
        v = int(p_item["ctrl_node_index"])
        u_tx = int(p_item["treat_txId"])
        v_tx = int(p_item["ctrl_txId"])

        # Complete pair membership assertion: never silently skip!
        assert u in adv_dict, f"Register '{reg_name}': Treat node {u} missing from 377 union!"
        assert v in adv_dict, f"Register '{reg_name}': Control node {v} missing from 377 union!"

        adv_u = adv_dict[u]
        adv_v = adv_dict[v]
        diff = adv_u - adv_v
        pair_diffs.append(diff)

        individual_pair_diffs.append({
            "register_name": reg_name,
            "pair_id": pid,
            "treat_node_index": u,
            "treat_txId": u_tx,
            "ctrl_node_index": v,
            "ctrl_txId": v_tx,
            "treat_advantage": adv_u,
            "ctrl_advantage": adv_v,
            "pair_difference": diff,
        })

    pair_diffs = np.array(pair_diffs)
    n_p = len(pair_diffs)
    assert n_p == len(pairs_list), f"Register {reg_name} pair count mismatch ({n_p} vs {len(pairs_list)})"

    B = 2000
    boot_means = []
    b_rng = np.random.default_rng(20261020)
    for _ in range(B):
        boot_idx = b_rng.choice(n_p, size=n_p, replace=True)
        boot_means.append(np.mean(pair_diffs[boot_idx]))

    b_mean = float(np.mean(boot_means))
    b_se = float(np.std(boot_means, ddof=1))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    print(f"  {reg_name:38s} | N={n_p:>2d} pairs | MeanDiff={np.mean(pair_diffs):>8.5f} | SE={b_se:>7.5f} | 95% CI=[{ci_low:>8.5f}, {ci_high:>8.5f}]")
    c2_outcomes.append({
        "register": reg_name,
        "pairs_count": n_p,
        "mean_diff": np.mean(pair_diffs),
        "bootstrap_se": b_se,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    })

out_dir = PROJECT_ROOT / "experiments" / "synthetic_c2_validation"
out_dir.mkdir(parents=True, exist_ok=True)

df_diffs = pd.DataFrame(individual_pair_diffs)
df_diffs.to_csv(out_dir / "synthetic_c2_individual_pair_differences.csv", index=False)
df_out = pd.DataFrame(c2_outcomes)
df_out.to_csv(out_dir / "synthetic_c2_bootstrap_results.csv", index=False)

print(f"\nSynthetic C2 Validation PASSED: All 8 registers ({len(df_diffs)} total pairs) verified with 100% pair completeness.")
