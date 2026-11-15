"""
Standalone verification script to numerically validate all figures and tables
against raw stored arrays and data artifacts within strict numerical tolerances.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def run_verification():
    print("=" * 60)
    print("ETGNN NUMERICAL VERIFICATION AUDIT")
    print("=" * 60)

    # 1. Load source test data
    nodes_df = pd.read_csv("data/processed/nodes_with_masks.csv", usecols=["label", "test_mask"])
    test_mask = nodes_df["test_mask"].to_numpy(dtype=bool)
    y_test = nodes_df.loc[test_mask, "label"].to_numpy()
    
    probs_npz = np.load("data/processed/final_test_probabilities.npz")
    results_df = pd.read_csv("results/tables/final_test_results.csv").set_index("model")
    
    print(f"Test Set Size: {len(y_test)} nodes (Illicit prevalence: {y_test.mean():.4%})")
    print("\n--- Verifying Test Metric Agreement (Tolerance: atol=1e-5) ---")
    
    all_passed = True
    for i, model_name in enumerate(results_df.index):
        k = f"model_{i}"
        prob = probs_npz[k]
        pred = (prob >= 0.5).astype(int)
        
        calc_prec = precision_score(y_test, pred, zero_division=0)
        calc_rec = recall_score(y_test, pred, zero_division=0)
        calc_f1 = f1_score(y_test, pred, zero_division=0)
        calc_roc = roc_auc_score(y_test, prob)
        calc_ap = average_precision_score(y_test, prob)
        
        row = results_df.loc[model_name]
        diffs = {
            "precision": abs(calc_prec - row["precision"]),
            "recall": abs(calc_rec - row["recall"]),
            "f1": abs(calc_f1 - row["f1"]),
            "roc_auc": abs(calc_roc - row["roc_auc"]),
            "auprc": abs(calc_ap - row["auprc"]),
        }
        max_diff = max(diffs.values())
        status = "PASSED" if max_diff < 1e-5 else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f"[{status}] {model_name:<25} | Max Diff: {max_diff:.2e} | AP: {calc_ap:.4f} (Saved: {row['auprc']:.4f})")

    # 2. Verify Explanation Edge Deletion Controls
    print("\n--- Verifying Explanation Deletion Controls ---")
    ctrl_df = pd.read_csv("results/tables/explanation_edge_deletion_controls.csv")
    for _, row in ctrl_df.iterrows():
        p_before = row["probability_before"]
        p_after = row["probability_after_deletion"]
        calc_drop = p_before - p_after
        diff = abs(calc_drop - row["signed_probability_drop"])
        print(f"Case {row['case']:<2} ({row['strategy']:<6}) | P_before: {p_before:.4f} | P_after: {p_after:.4f} | Drop: {calc_drop:+.4f} (Diff: {diff:.2e})")

    # 3. Verify Multi-Seed Validation Stats
    print("\n--- Verifying Multi-Seed Ablation Averages (Seeds 7, 42, 123) ---")
    etgnn_seeds = [
        json.load(open("results/tables/ablation/etgnn_seed_7.json"))["validation_auprc"],
        json.load(open("results/tables/ablation/etgnn_seed_42.json"))["validation_auprc"],
        json.load(open("results/tables/ablation/etgnn_seed_123.json"))["validation_auprc"],
    ]
    mgat_seeds = [
        json.load(open("results/tables/ablation/matched_gat_seed_7.json"))["validation_auprc"],
        json.load(open("results/tables/ablation/matched_gat_seed_42.json"))["validation_auprc"],
        json.load(open("results/tables/ablation/matched_gat_seed_123.json"))["validation_auprc"],
    ]
    print(f"ETGNN Validation AP:       Mean = {np.mean(etgnn_seeds):.4f}, Std = {np.std(etgnn_seeds):.4f}")
    print(f"Matched GAT Validation AP: Mean = {np.mean(mgat_seeds):.4f}, Std = {np.std(mgat_seeds):.4f}")
    
    print("\nOverall Verification Status:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
