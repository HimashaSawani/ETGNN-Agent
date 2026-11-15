"""
Comprehensive script to generate all publication-grade, high-resolution figures (PNG at 200 DPI + Vector PDF)
for the ETGNN Research Project, strictly aligned with empirical results, explicit uncertainty reporting,
and ablation studies.

Figures Generated:
1. Dataset Class Distribution (Licit, Illicit, Unknown)
2. Node Volume Across 49 Discrete Time Steps (Train, Val, Test Splits)
3. Labelled Class Composition (Illicit vs Licit) Across Time Steps
4. Receiver Operating Characteristic (ROC) Curves on Test Split (t=41..49)
5. Precision-Recall Curves with Average Precision (AP) on Test Split (t=41..49)
6. Model Performance Metric Comparison (F1 and AP) Across All Models
7. Core RQ2 Temporal Ablation: Static GAT vs Matched GAT vs ETGNN (Single Test Split)
8. Multi-Seed Uncertainty: Matched GAT vs ETGNN Validation AP (Mean ± Std over Seeds 7, 42, 123)
9. Topological and Feature Ablation: Message-Passing vs Tabular Baselines (Central Finding)
10. Normalized Confusion Matrices with Raw Counts (Random Forest, GAT, ETGNN)
"""

import gc
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset & EDA Figures
# -------------------------------------------------------------
def plot_eda_figures():
    print("Generating EDA Figures...")
    nodes_df = pd.read_csv("data/processed/nodes_with_masks.csv", usecols=["label", "time_step"])
    
    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    class_counts = nodes_df["label"].value_counts().sort_index()
    labels = ["Unknown (-1)", "Licit (0)", "Illicit (1)"]
    counts = [class_counts.get(-1, 0), class_counts.get(0, 0), class_counts.get(1, 0)]
    colors = ["#7f8c8d", "#27ae60", "#e74c3c"]
    
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", width=0.55, alpha=0.9)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,}\n({height/len(nodes_df)*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax.set_title("Elliptic Bitcoin Dataset: Node Class Distribution", pad=12)
    ax.set_ylabel("Number of Transactions (Nodes)")
    ax.set_ylim(0, max(counts) * 1.18)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=200)
    plt.savefig(FIG_DIR / "class_distribution.pdf")
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 2. Time-step Distribution
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ts_counts = nodes_df["time_step"].value_counts().sort_index()
    ax.bar(ts_counts.index, ts_counts.values, color="#2980b9", edgecolor="black", alpha=0.85, width=0.8)
    ax.axvspan(0.5, 32.5, color="#2ecc71", alpha=0.12, label="Train Split (t=1..32)")
    ax.axvspan(32.5, 40.5, color="#f39c12", alpha=0.15, label="Validation Split (t=33..40)")
    ax.axvspan(40.5, 49.5, color="#e74c3c", alpha=0.15, label="Test Split (t=41..49)")
    
    ax.set_title("Transaction Count Across 49 Discrete Time Steps", pad=10)
    ax.set_xlabel("Time Step (1 to 49)")
    ax.set_ylabel("Transaction Count")
    ax.set_xlim(0, 50)
    ax.set_xticks(range(1, 50, 4))
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "time_distribution.png", dpi=200)
    plt.savefig(FIG_DIR / "time_distribution.pdf")
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 3. Class Per Time Step (Stacked Bar for Labelled)
    fig, ax = plt.subplots(figsize=(10, 4))
    labelled_df = nodes_df[nodes_df["label"].isin([0, 1])]
    grouped = labelled_df.groupby(["time_step", "label"]).size().unstack(fill_value=0)
    
    p1 = ax.bar(grouped.index, grouped[0], label="Licit (0)", color="#27ae60", alpha=0.85, width=0.8)
    p2 = ax.bar(grouped.index, grouped[1], bottom=grouped[0], label="Illicit (1)", color="#e74c3c", alpha=0.85, width=0.8)
    
    ax.axvline(32.5, color="black", linestyle="--", linewidth=1.2, label="Train / Val Boundary")
    ax.axvline(40.5, color="black", linestyle="-.", linewidth=1.2, label="Val / Test Boundary")
    
    ax.set_title("Distribution of Supervised Labels (Licit vs. Illicit) Across Time Steps", pad=10)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Labelled Transaction Count")
    ax.set_xlim(0, 50)
    ax.set_xticks(range(1, 50, 4))
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_per_timestep.png", dpi=200)
    plt.savefig(FIG_DIR / "class_per_timestep.pdf")
    plt.close(fig)
    del fig, ax, nodes_df, labelled_df, grouped
    gc.collect()


# -------------------------------------------------------------
# 2. Performance Curves & Ablation Figures
# -------------------------------------------------------------
def plot_evaluation_figures():
    print("Generating Evaluation & Ablation Figures...")
    nodes_df = pd.read_csv("data/processed/nodes_with_masks.csv", usecols=["label", "test_mask"])
    test_mask = nodes_df["test_mask"].to_numpy(dtype=bool)
    y_test = nodes_df.loc[test_mask, "label"].to_numpy()

    probs_data = np.load("data/processed/final_test_probabilities.npz")
    results_df = pd.read_csv("results/tables/final_test_results.csv").set_index("model")
    model_keys = {name: f"model_{i}" for i, name in enumerate(results_df.index)}

    model_styles = {
        "Random Forest": {"color": "#1f77b4", "ls": "-", "label": "Random Forest"},
        "XGBoost": {"color": "#ff7f0e", "ls": "--", "label": "XGBoost"},
        "GCN": {"color": "#2ca02c", "ls": ":", "label": "GCN"},
        "GraphSAGE": {"color": "#9467bd", "ls": "-.", "label": "GraphSAGE"},
        "GAT": {"color": "#8c564b", "ls": ":", "label": "GAT"},
        "Matched GAT": {"color": "#e377c2", "ls": "--", "label": "Matched GAT"},
        "ETGNN": {"color": "#d62728", "ls": "-", "lw": 2.2, "label": "ETGNN (Proposed)"},
    }

    # 4. ROC Curves
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    for mname, style in model_styles.items():
        k = model_keys.get(mname)
        if k and k in probs_data.files:
            prob = probs_data[k]
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc_val = results_df.loc[mname, "roc_auc"] if mname in results_df.index else 0.0
            ax.plot(fpr, tpr, color=style["color"], linestyle=style.get("ls", "-"),
                    linewidth=style.get("lw", 1.6), label=f"{style['label']} (ROC-AUC={auc_val:.3f})")
            
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess")
    ax.set_title("Receiver Operating Characteristic (ROC) — Test Split (t=41..49)", pad=10)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=200)
    plt.savefig(FIG_DIR / "roc_curves.pdf")
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 5. Precision-Recall (PR) Curves with Average Precision (AP)
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    base_rate = (y_test == 1).mean()
    for mname, style in model_styles.items():
        k = model_keys.get(mname)
        if k and k in probs_data.files:
            prob = probs_data[k]
            prec, rec, _ = precision_recall_curve(y_test, prob)
            ap_val = results_df.loc[mname, "auprc"] if mname in results_df.index else 0.0
            ax.plot(rec, prec, color=style["color"], linestyle=style.get("ls", "-"),
                    linewidth=style.get("lw", 1.6), label=f"{style['label']} (AP={ap_val:.3f})")
            
    ax.axhline(base_rate, color="black", linestyle="--", alpha=0.4, label=f"Base Rate ({base_rate:.3f})")
    ax.set_title("Precision-Recall Curves (AP / PR-AUC) — Test Split (t=41..49)", pad=10)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_recall_curves.png", dpi=200)
    plt.savefig(FIG_DIR / "pr_curves.png", dpi=200)
    plt.savefig(FIG_DIR / "pr_curves.pdf")
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 6. Metric Bar Charts (F1 and AP)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    eval_models = ["Random Forest", "XGBoost", "GCN", "GraphSAGE", "GAT", "Matched GAT", "ETGNN"]
    f1_vals = [results_df.loc[m, "f1"] for m in eval_models]
    auprc_vals = [results_df.loc[m, "auprc"] for m in eval_models]
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#d62728"]

    bars1 = ax1.bar(eval_models, f1_vals, color=bar_colors, edgecolor="black", alpha=0.85, width=0.6)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_title("F1-Score (Minority Illicit Class)", pad=10)
    ax1.set_ylabel("F1-Score (Threshold=0.5)")
    ax1.set_ylim(0, max(f1_vals) * 1.18)
    ax1.set_xticks(range(len(eval_models)))
    ax1.set_xticklabels(eval_models, rotation=30, ha="right")

    bars2 = ax2.bar(eval_models, auprc_vals, color=bar_colors, edgecolor="black", alpha=0.85, width=0.6)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_title("Average Precision (AP / PR-AUC)", pad=10)
    ax2.set_ylabel("Average Precision (AP)")
    ax2.set_ylim(0, max(auprc_vals) * 1.18)
    ax2.set_xticks(range(len(eval_models)))
    ax2.set_xticklabels(eval_models, rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "f1_comparison.png", dpi=200)
    plt.savefig(FIG_DIR / "auprc_comparison.png", dpi=200)
    plt.savefig(FIG_DIR / "f1_auprc_comparison.pdf")
    plt.close(fig)
    del fig, ax1, ax2
    gc.collect()

    # 7. RQ2 Temporal Ablation (Single-Run Test Comparison)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ablation_models = ["GAT (Static)", "Matched GAT (Zero-Time)", "ETGNN (Sinusoidal Time)"]
    ablation_auprc = [results_df.loc["GAT", "auprc"], results_df.loc["Matched GAT", "auprc"], results_df.loc["ETGNN", "auprc"]]
    ablation_f1 = [results_df.loc["GAT", "f1"], results_df.loc["Matched GAT", "f1"], results_df.loc["ETGNN", "f1"]]
    
    x = np.arange(len(ablation_models))
    width = 0.35
    rects1 = ax.bar(x - width/2, ablation_auprc, width, label='AP (PR-AUC)', color='#3498db', edgecolor='black')
    rects2 = ax.bar(x + width/2, ablation_f1, width, label='F1-Score', color='#e67e22', edgecolor='black')
    
    for rects in [rects1, rects2]:
        for b in rects:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("RQ2 Temporal Ablation (Test Split t=41..49)", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_models)
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 0.45)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gat_vs_etgnn.png", dpi=200)
    plt.savefig(FIG_DIR / "gat_vs_etgnn.pdf")
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 8. Repeated-Run Multi-Seed Uncertainty (Seeds 7, 42, 123 on Validation)
    etgnn_seeds = [0.5096, 0.5488, 0.5293]
    mgat_seeds = [0.5572, 0.4836, 0.3934]
    
    fig, ax = plt.subplots(figsize=(6, 4.2))
    u_models = ["Matched GAT (Zero-Time)", "ETGNN (Sinusoidal Time)"]
    means = [np.mean(mgat_seeds), np.mean(etgnn_seeds)]
    stds = [np.std(mgat_seeds), np.std(etgnn_seeds)]
    
    x_u = np.arange(len(u_models))
    bars_u = ax.bar(x_u, means, yerr=stds, capsize=6, color=["#e377c2", "#d62728"], edgecolor="black", width=0.45, alpha=0.85)
    for b, m, s in zip(bars_u, means, stds):
        ax.annotate(f"{m:.4f} ± {s:.4f}", (b.get_x() + b.get_width()/2, m + s), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax.set_title("Validation Uncertainty: Multi-Seed Repeated Runs (Seeds 7, 42, 123)", pad=10)
    ax.set_xticks(x_u)
    ax.set_xticklabels(u_models)
    ax.set_ylabel("Validation AP (Mean ± Std)")
    ax.set_ylim(0, 0.70)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "repeated_ablation_uncertainty.png", dpi=200)
    plt.savefig(FIG_DIR / "repeated_ablation_uncertainty.pdf")
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 9. Prominent Feature & Edge Ablation Plot (Key Research Finding)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    abl_keys = [
        ("RF (Full 165 Feat)", results_df.loc["Random Forest", "auprc"]),
        ("RF Local Only (93 Feat)", results_df.loc["RF local only", "auprc"]),
        ("ETGNN Local No-Edges", results_df.loc["ETGNN local no edges", "auprc"]),
        ("ETGNN Local Only (93 Feat)", results_df.loc["ETGNN local only", "auprc"]),
        ("ETGNN Full (165 Feat + Edges)", results_df.loc["ETGNN", "auprc"]),
    ]
    abl_labels = [k[0] for k in abl_keys]
    abl_ap = [k[1] for k in abl_keys]
    abl_colors = ["#1f77b4", "#5dade2", "#27ae60", "#e67e22", "#d62728"]
    
    bars_abl = ax.bar(abl_labels, abl_ap, color=abl_colors, edgecolor="black", width=0.55, alpha=0.85)
    for b in bars_abl:
        h = b.get_height()
        ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax.set_title("Graph & Feature Ablations: Impact of Message Passing vs. Feature Isolation (Test Split)", pad=10)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_ylim(0, 0.75)
    ax.set_xticklabels(abl_labels, rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_graph_vs_features.png", dpi=200)
    plt.savefig(FIG_DIR / "ablation_graph_vs_features.pdf")
    plt.close(fig)
    del fig, ax
    gc.collect()

    # 10. Normalized Confusion Matrices with Raw Counts
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cm_models = [("Random Forest", "Random Forest"), ("GAT", "GAT (Static)"), ("ETGNN", "ETGNN (Proposed)")]
    for ax, (mkey, title) in zip(axes, cm_models):
        k = model_keys[mkey]
        prob = probs_data[k]
        pred = (prob >= 0.5).astype(int)
        cm_norm = confusion_matrix(y_test, pred, normalize="true")
        cm_raw = confusion_matrix(y_test, pred)
        
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_raw[i, j]:,}\n({cm_norm[i, j]:.1%})", ha="center", va="center",
                        color=text_color, fontweight="bold", fontsize=9.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Licit (0)", "Illicit (1)"])
        ax.set_yticklabels(["Licit (0)", "Illicit (1)"])
        ax.set_title(f"{title}\nCounts & Normalized % (Thresh=0.5)", pad=8)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrices.png", dpi=200)
    plt.savefig(FIG_DIR / "confusion_matrices.pdf")
    plt.close(fig)
    del fig, axes
    gc.collect()

if __name__ == "__main__":
    plot_eda_figures()
    plot_evaluation_figures()
    print("All evaluation & ablation figures generated successfully!")
