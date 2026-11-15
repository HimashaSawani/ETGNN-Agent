import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = Path("research_extension/phase14b_final_completion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. FINAL MODEL COMPARISON TABLE
df_tree = pd.read_csv(OUT_DIR / "TREE_BASELINES_TEST_EVALUATION.csv")
df_phase14 = pd.read_csv("research_extension/phase14_results_analysis/PHASE14_VERIFIED_MODEL_COMPARISON.csv")

final_rows = []

# Tree models
for _, r in df_tree.iterrows():
    m_name = r["model"]
    if m_name == "RandomForest_165feat":
        final_rows.append({
            "model_name": "Random Forest (165 features)",
            "model_paradigm": "Tree Ensemble (Tabular)",
            "parameter_count": "100 trees (depth 12)",
            "seeds": 1,
            "val_auprc": r["val_ap"],
            "test_auprc": r["test_ap"],
            "test_roc_auc": r["test_roc"],
            "test_f1": r["test_f1"],
            "test_precision": r["test_prec"],
            "test_recall": r["test_rec"],
            "ap_s0_k0": r["ap_s0"],
            "ap_s1_k12": r["ap_s1"],
            "ap_s2_k3plus": r["ap_s2"],
            "protocol_compatibility": "Direct (Strict Held-Out Split)"
        })
    elif m_name == "XGBoost_165feat":
        final_rows.append({
            "model_name": "XGBoost (165 features)",
            "model_paradigm": "Gradient Boosted Trees (Tabular)",
            "parameter_count": "300 trees (early stop)",
            "seeds": 1,
            "val_auprc": r["val_ap"],
            "test_auprc": r["test_ap"],
            "test_roc_auc": r["test_roc"],
            "test_f1": r["test_f1"],
            "test_precision": r["test_prec"],
            "test_recall": r["test_rec"],
            "ap_s0_k0": r["ap_s0"],
            "ap_s1_k12": r["ap_s1"],
            "ap_s2_k3plus": r["ap_s2"],
            "protocol_compatibility": "Direct (Strict Held-Out Split)"
        })

# Neural models from Phase 14
for _, r in df_phase14.iterrows():
    cfg = r["config"]
    final_rows.append({
        "model_name": cfg,
        "model_paradigm": r["model_type"],
        "parameter_count": f"{r['parameters']:,}",
        "seeds": r["seeds"],
        "val_auprc": r["val_ap_mean"],
        "test_auprc": r["test_ap_mean"],
        "test_roc_auc": r["test_roc_mean"],
        "test_f1": r["test_f1_mean"],
        "test_precision": r["test_precision"],
        "test_recall": r["test_recall"],
        "ap_s0_k0": r["ap_deg_0"],
        "ap_s1_k12": r["ap_deg_1_2"],
        "ap_s2_k3plus": r["ap_deg_3plus"],
        "protocol_compatibility": "Direct (Strict Held-Out Split)"
    })

df_final = pd.DataFrame(final_rows)
# Sort by test_auprc descending
df_final = df_final.sort_values(by="test_auprc", ascending=False)
df_final.to_csv(OUT_DIR / "PHASE14B_FINAL_MODEL_COMPARISON.csv", index=False)
print("Saved PHASE14B_FINAL_MODEL_COMPARISON.csv")

# 2. FIGURE 1: COMPLETE BENCHMARK COMPARISON
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
models_to_plot = [
    ("Random Forest", 0.6323, "#2B5C8F", "Tree Tabular"),
    ("XGBoost", 0.5472, "#4A7BB7", "Tree Tabular"),
    ("Feature-Only MLP", 0.3380, "#E76F51", "Neural Tabular"),
    ("Residual ETGNN (M1)", 0.3126, "#2A9D8F", "GNN + Skip"),
    ("M4 Control (No Gate)", 0.3127, "#588157", "GNN Control"),
    ("Baseline ETGNN (M0)", 0.2250, "#888888", "Baseline GNN"),
    ("RG-ETGNN CE (M3)", 0.2419, "#E63946", "Gated GNN"),
    ("RG-ETGNN Focal (M3)", 0.2273, "#D62828", "Gated GNN")
]

x_pos = np.arange(len(models_to_plot))
y_vals = [m[1] for m in models_to_plot]
colors = [m[2] for m in models_to_plot]

bars = ax.bar(x_pos, y_vals, color=colors, edgecolor='black', alpha=0.88, width=0.6)

for bar, val in zip(bars, y_vals):
    ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.015, f"{val:.4f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([m[0] for m in models_to_plot], rotation=25, ha='right', fontsize=10)
ax.set_ylabel("Held-Out Test AUPRC (Timesteps 41–49)", fontsize=12, fontweight='bold')
ax.set_title("Defensible Benchmark: Tree Ensembles, Tabular MLP, and Temporal GNNs on Elliptic Bitcoin Dataset", fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 0.72)

# Annotations separating regimes
ax.axvline(1.5, color='#444444', linestyle='--', alpha=0.5)
ax.axvline(2.5, color='#444444', linestyle='--', alpha=0.5)
ax.text(0.5, 0.68, "Tabular Trees\n(Robust to Shift)", ha='center', fontsize=10, fontweight='bold', color='#2B5C8F')
ax.text(2.0, 0.68, "Tabular MLP", ha='center', fontsize=10, fontweight='bold', color='#E76F51')
ax.text(5.0, 0.68, "Graph Neural Networks\n(Subject to Topology Collapse)", ha='center', fontsize=10, fontweight='bold', color='#2A9D8F')

plt.tight_layout()
fig_p1 = OUT_DIR / "phase14b_complete_benchmark_comparison.png"
plt.savefig(fig_p1)
plt.close()
print(f"Generated {fig_p1.name}")

# 3. FIGURE 2: C1 FEATURE VS EDGE PERTURBATION
df_xai = pd.read_csv(OUT_DIR / "C1_C2_MATCHED_INTERVENTION_EXPERIMENTS.csv")
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

tp_edge = df_xai[df_xai["cohort"] == "TP"]["top_edge_necessity_drop"].mean()
fp_edge = df_xai[df_xai["cohort"] == "FP"]["top_edge_necessity_drop"].mean()
tp_feat = df_xai[df_xai["cohort"] == "TP"]["feat_necessity_drop"].mean()
fp_feat = df_xai[df_xai["cohort"] == "FP"]["feat_necessity_drop"].mean()

ind = np.arange(2)
w = 0.35

b_feat = ax.bar(ind - w/2, [tp_feat, fp_feat], w, label='Tabular Feature Masking (Top 10%)', color='#E76F51', edgecolor='black', alpha=0.9)
b_edge = ax.bar(ind + w/2, [tp_edge, fp_edge], w, label='Graph Edge Removal (Top 1 Edge)', color='#2A9D8F', edgecolor='black', alpha=0.9)

for b in list(b_feat) + list(b_edge):
    h = b.get_height()
    va = 'bottom' if h >= 0 else 'top'
    y = h + 0.005 if h >= 0 else h - 0.015
    ax.text(b.get_x() + b.get_width()/2.0, y, f"{h:+.4f}", ha='center', va=va, fontsize=9.5, fontweight='bold')

ax.set_xticks(ind)
ax.set_xticklabels(["True Positives (TP, N=275)", "False Positives (FP, N=275)"], fontsize=11, fontweight='bold')
ax.set_ylabel("Mean Necessity Probability Drop (ΔP)", fontsize=11, fontweight='bold')
ax.set_title("C1 Fidelity: Tabular Feature Dominance vs Edge Noise Vulnerability", fontsize=12, fontweight='bold', pad=15)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylim(-0.03, 0.16)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
fig_p2 = OUT_DIR / "phase14b_c1_feature_vs_edge_intervention.png"
plt.savefig(fig_p2)
plt.close()
print(f"Generated {fig_p2.name}")

# 4. FIGURE 3: C2 COUNTERFACTUAL FLIP RATES
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
tp_flip = df_xai[df_xai["cohort"] == "TP"]["cf_flipped"].mean() * 100
fp_flip = df_xai[df_xai["cohort"] == "FP"]["cf_flipped"].mean() * 100

bars_cf = ax.bar([0, 1], [tp_flip, fp_flip], color=['#2A9D8F', '#E63946'], edgecolor='black', alpha=0.88, width=0.5)

for b, v in zip(bars_cf, [tp_flip, fp_flip]):
    ax.text(b.get_x() + b.get_width()/2.0, v + 0.5, f"{v:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(["True Positives (N=275)", "False Positives (N=275)"], fontsize=11, fontweight='bold')
ax.set_ylabel("Counterfactual Prediction Flip Rate (%)", fontsize=11, fontweight='bold')
ax.set_title("C2 Counterfactual Sensitivity: Edge Removal Exonerates False Positives", fontsize=12, fontweight='bold', pad=15)
ax.set_ylim(0, 16)
ax.text(0.5, 13.5, "Fisher's Exact Test: p = 0.0016\n(Holm-Bonferroni Adjusted p = 0.0032)", ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1))
plt.tight_layout()
fig_p3 = OUT_DIR / "phase14b_c2_counterfactual_flip_rates.png"
plt.savefig(fig_p3)
plt.close()
print(f"Generated {fig_p3.name}")

print("\n[SUCCESS] Generated all Phase 14B publication figures.")

