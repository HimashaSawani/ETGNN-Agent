#!/usr/bin/env python3
"""
generate_phase12_figures.py
===========================
Generates publication-grade figures for Phase 12:
  1. `figures/exp_a_temporal_stability_pre_vs_post.png`:
     - Panel A: Candidate-edge (E_cand) structural shift between Pre-Shift (steps 41-42) and Post-Shift (steps 43-49)
     - Panel B: Explanation stability & faithfulness metrics (Pre-Shift vs. Post-Shift with 95% bootstrap CIs)
  2. `figures/exp_b_error_mitigation_controls.png`:
     - Panel A: Held-out Test Average Precision (AP) and F1@0.5 across Single-Model & Dual-Model Mitigations vs. Matched Controls
     - Panel B: Held-out Test False Positive Counts (FP) and False Positive Rate (FPR) across Mitigations vs. Controls
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
P12_DIR = PROJECT_ROOT / "research_extension" / "phase12"
TABLES_DIR = P12_DIR / "tables"
FIGS_DIR = P12_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def plot_experiment_a():
    df_cen = pd.read_csv(TABLES_DIR / "EXP_A_CANDIDATE_EDGE_SHIFT_CENSUS.csv")
    df_sum = pd.read_csv(TABLES_DIR / "EXP_A_TEMPORAL_STABILITY_SUMMARY.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)

    # Panel A: E_cand distribution Pre vs Post (All vs Illicit)
    ax0 = axes[0]
    cohorts = [
        "Pre-Shift (41-42) - All",
        "Post-Shift (43-49) - All",
        "Pre-Shift (41-42) - Illicit (y=1)",
        "Post-Shift (43-49) - Illicit (y=1)",
    ]
    short_labels = ["Pre All\n(N=3,286)", "Post All\n(N=6,687)", "Pre Illicit\n(N=355)", "Post Illicit\n(N=169)"]
    sub_c = df_cen.set_index("Cohort").loc[cohorts]

    p0 = sub_c["E_cand_0_Pct"].values
    p1 = sub_c["E_cand_1_Pct"].values
    p2 = sub_c["E_cand_2_Pct"].values
    p3 = sub_c["E_cand_ge3_Pct"].values

    x = np.arange(len(cohorts))
    w = 0.55
    ax0.bar(x, p0, w, label="E_cand = 0 (Isolated / Unexplainable)", color="#d95f02", alpha=0.88)
    ax0.bar(x, p1, w, bottom=p0, label="E_cand = 1 (Trivial Single-Edge)", color="#fc8d62", alpha=0.88)
    ax0.bar(x, p2, w, bottom=p0 + p1, label="E_cand = 2 (Binary Bottleneck)", color="#66c2a5", alpha=0.88)
    ax0.bar(x, p3, w, bottom=p0 + p1 + p2, label="E_cand >= 3 (Multi-Edge Ranking)", color="#1b9e77", alpha=0.92)

    ax0.set_xticks(x)
    ax0.set_xticklabels(short_labels, fontsize=9)
    ax0.set_ylabel("Percentage of Transactions (%)", fontsize=10, fontweight="bold")
    ax0.set_title("(A) 2-Hop Candidate-Edge (E_cand) Shift Across Step 43", fontsize=10.5, fontweight="bold")
    ax0.set_ylim(0, 108)
    ax0.grid(axis="y", linestyle="--", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel B: Pre vs Post Stability Metrics (Stratified Cohort N=100 vs 100)
    ax1 = axes[1]
    s1 = df_sum[df_sum["Stratum"].str.startswith("1. Stratified")].set_index("Metric_Key")
    keys_plot = [
        "GE_Intra_Top1_Agree",
        "GE_Intra_TopK_Jaccard",
        "GE_Inter165_Top1_Agree",
        "LOO_Inter165_Top1_Agree",
        "LOO_Inter165_Spearman",
        "LOO_Inter93_Top1_Agree",
        "GE_Top1_Hits_LOO_Oracle",
    ]
    labels_plot = [
        "GE Intra-Seed\nTop-1 Agree",
        "GE Intra-Seed\nTop-k Jaccard",
        "GE Inter-Seed\n(165f) Top-1",
        "Exact LOO Inter\n(165f) Top-1",
        "Exact LOO Inter\n(165f) Spearman",
        "Exact LOO Inter\n(93f) Top-1",
        "GE vs. LOO\nOracle Hit Rate",
    ]
    sub_s = s1.loc[keys_plot]
    y_pos = np.arange(len(keys_plot))
    h = 0.35

    pre_m = sub_s["Pre_Mean"].values
    pre_err_l = pre_m - sub_s["Pre_CI95_Low"].values
    pre_err_h = sub_s["Pre_CI95_High"].values - pre_m

    post_m = sub_s["Post_Mean"].values
    post_err_l = post_m - sub_s["Post_CI95_Low"].values
    post_err_h = sub_s["Post_CI95_High"].values - post_m

    ax1.barh(y_pos + h / 2, pre_m, h, xerr=[pre_err_l, pre_err_h], capsize=3, label="Pre-Shift (Steps 41-42)", color="#2b5c8f", alpha=0.9)
    ax1.barh(y_pos - h / 2, post_m, h, xerr=[post_err_l, post_err_h], capsize=3, label="Post-Shift (Steps 43-49)", color="#e07a5f", alpha=0.9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels_plot, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Stability / Agreement Score (with 95% Bootstrap CI)", fontsize=10, fontweight="bold")
    ax1.set_title("(B) Multi-Edge (E_cand >= 2) Explanation Stability: Pre vs. Post Shift", fontsize=10.5, fontweight="bold")
    ax1.set_xlim(-0.05, 1.08)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    out_p = FIGS_DIR / "exp_a_temporal_stability_pre_vs_post.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"[Figure A] Saved {out_p}")


def plot_experiment_b():
    df_res = pd.read_csv(TABLES_DIR / "EXP_B_TEST_MITIGATION_RESULTS.csv")
    df_full = df_res[df_res["Period"].str.startswith("Full_Test")].set_index("Strategy_ID")

    sel_ids = [
        "01_Baseline_ETGNN_165f_Default",
        "03_M1a_SingleModel_GE_TopK_Pruned",
        "04_M1b_SingleModel_LOO_TopK_Pruned",
        "05_C1a_SingleModel_Random_TopK_Pruned",
        "06_M1c_SingleModel_SensGated_SelfLoop",
        "07_C1b_SingleModel_ConfGated_SelfLoop",
        "09_M1d_SingleModel_100pct_SelfLoop",
        "10_M2a_DualModel_SensEmpty_20pct",
        "12_C2a_DualModel_ConfUncert_20pct",
        "13_C2b_DualModel_PredRisk_20pct",
        "14_C2c_DualModel_Random_20pct",
        "15_M3a_PredPos_SensEmpty_Veto50pct",
        "18_C3b_PredPos_Random_Veto50pct",
        "20_C4_Ungated_LinearBlend_ValOpt",
        "21_C5_Standalone_ETGNN_Local_NoEdges",
    ]
    short_names = [
        "Baseline: Original ETGNN (165f)",
        "M1a: Single-Model GE Top-k Pruned",
        "M1b: Single-Model LOO Top-k Pruned",
        "C1a: Single-Model Random-k Pruned",
        "M1c: Single-Model Sens-Gated Self-Loop",
        "C1b: Single-Model Conf-Gated Self-Loop",
        "M1d: Single-Model 100% Self-Loop (E=0)",
        "M2a: Dual Sens-Gated Local (r=20%)",
        "C2a: Dual Conf-Gated Local (r=20%)",
        "C2b: Dual PredRisk-Gated Local (r=20%)",
        "C2c: Dual Random-Gated Local (r=20%)",
        "M3a: PredPos Sens-Veto 50% -> Local",
        "C3b: PredPos Random-Veto 50% -> Local",
        "C4: Ungated Val-Opt Linear Blend",
        "C5: Standalone 100% Local No-Edges",
    ]

    sub = df_full.loc[sel_ids]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), dpi=300)

    y = np.arange(len(sel_ids))
    h = 0.36

    # Panel A: AP and F1
    ax0 = axes[0]
    ax0.barh(y - h / 2, sub["AP_Primary"].values, h, label="Primary AP (average_precision_score)", color="#1f4e78", alpha=0.92)
    ax0.barh(y + h / 2, sub["F1"].values, h, label="F1-Score (@ tau)", color="#2ca02c", alpha=0.85)
    ax0.set_yticks(y)
    ax0.set_yticklabels(short_names, fontsize=8.5)
    ax0.invert_yaxis()
    ax0.set_xlabel("Score on Held-Out Test Split (Steps 41-49, N=9,973)", fontsize=9.5, fontweight="bold")
    ax0.set_title("(A) Held-Out Test AP & F1: Mitigations vs. Matched Controls", fontsize=10.5, fontweight="bold")
    ax0.grid(axis="x", linestyle="--", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Panel B: False Positives (FP) and True Positives (TP)
    ax1 = axes[1]
    ax1.barh(y - h / 2, sub["FP"].values, h, label="False Positives (FP, Lower is Better)", color="#d62728", alpha=0.85)
    ax1.barh(y + h / 2, sub["TP"].values, h, label="True Positives (TP out of 524 Illicit)", color="#ff7f0e", alpha=0.88)
    ax1.set_yticks(y)
    ax1.set_yticklabels([""] * len(sel_ids))
    ax1.invert_yaxis()
    ax1.set_xlabel("Transaction Count on Held-Out Test Split (N=9,973)", fontsize=9.5, fontweight="bold")
    ax1.set_title("(B) False Alarm Reduction (FP) vs. True Detections (TP)", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    out_p = FIGS_DIR / "exp_b_error_mitigation_controls.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"[Figure B] Saved {out_p}")


if __name__ == "__main__":
    plot_experiment_a()
    plot_experiment_b()

