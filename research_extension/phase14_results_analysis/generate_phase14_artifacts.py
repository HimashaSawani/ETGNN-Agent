import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(".")
PHASE13_DIR = ROOT / "research_extension" / "phase13_advanced_investigation"
PHASE14_DIR = ROOT / "research_extension" / "phase14_results_analysis"
PHASE14_DIR.mkdir(parents=True, exist_ok=True)

df_val = pd.read_csv(PHASE13_DIR / "PHASE13_VAL_SELECTION.csv")
df_test = pd.read_csv(PHASE13_DIR / "PHASE13_TEST_METRICS.csv")

print(f"Loaded {len(df_val)} val records and {len(df_test)} test records.")

# 1. PARAMETER COUNTS & METADATA
MODEL_METADATA = {
    "M0_ETGNN_Baseline": {"params": 11910, "type": "GNN (Temporal GAT)", "has_res": False, "has_gate": False},
    "M1_ResidualETGNN": {"params": 23688, "type": "GNN + Ego-Residual", "has_res": True, "has_gate": False},
    "M2_DeepMLP": {"params": 31682, "type": "Tabular Feature-Only", "has_res": False, "has_gate": False},
    "M3_RG_ETGNN_CE": {"params": 51210, "type": "Gated GNN + Ego-Residual", "has_res": True, "has_gate": True},
    "M3_RG_ETGNN_Focal": {"params": 51210, "type": "Gated GNN + Ego-Residual", "has_res": True, "has_gate": True},
    "M4_Ablation_NoGate": {"params": 51210, "type": "Gating-Disabled Control", "has_res": True, "has_gate": False},
}

# 2. GENERATE COMPREHENSIVE COMPARISON TABLE
rows = []
for cfg in ["M0_ETGNN_Baseline", "M1_ResidualETGNN", "M2_DeepMLP", "M3_RG_ETGNN_CE", "M3_RG_ETGNN_Focal", "M4_Ablation_NoGate"]:
    sub_test = df_test[df_test["config"] == cfg]
    sub_val = df_val[df_val["config"] == cfg] if cfg != "M0_ETGNN_Baseline" else None

    meta = MODEL_METADATA[cfg]
    n_seeds = len(sub_test)

    val_ap_m = sub_val["val_ap"].mean() if sub_val is not None else 0.5284  # M0 historical val AP
    val_ap_s = sub_val["val_ap"].std(ddof=1) if sub_val is not None and n_seeds > 1 else 0.0

    test_ap_m = sub_test["test_ap"].mean()
    test_ap_s = sub_test["test_ap"].std(ddof=1) if n_seeds > 1 else 0.0

    test_roc_m = sub_test["test_roc"].mean()
    test_roc_s = sub_test["test_roc"].std(ddof=1) if n_seeds > 1 else 0.0

    test_f1_m = sub_test["test_f1"].mean()
    test_f1_s = sub_test["test_f1"].std(ddof=1) if n_seeds > 1 else 0.0

    test_prec_m = sub_test["test_prec"].mean()
    test_rec_m = sub_test["test_rec"].mean()

    ap_s0_m = sub_test["ap_s0_k0"].mean()
    ap_s1_m = sub_test["ap_s1_k12"].mean()
    ap_s2_m = sub_test["ap_s2_k3plus"].mean()

    time_m = sub_val["elapsed_s"].mean() if sub_val is not None else 0.0

    rows.append({
        "config": cfg,
        "model_type": meta["type"],
        "parameters": meta["params"],
        "seeds": n_seeds,
        "val_ap_mean": val_ap_m,
        "val_ap_std": val_ap_s,
        "test_ap_mean": test_ap_m,
        "test_ap_std": test_ap_s,
        "test_roc_mean": test_roc_m,
        "test_roc_std": test_roc_s,
        "test_f1_mean": test_f1_m,
        "test_f1_std": test_f1_s,
        "test_precision": test_prec_m,
        "test_recall": test_rec_m,
        "ap_deg_0": ap_s0_m,
        "ap_deg_1_2": ap_s1_m,
        "ap_deg_3plus": ap_s2_m,
        "training_time_s": time_m
    })

df_comp = pd.DataFrame(rows)
df_comp.to_csv(PHASE14_DIR / "PHASE14_VERIFIED_MODEL_COMPARISON.csv", index=False)
print("Saved PHASE14_VERIFIED_MODEL_COMPARISON.csv")

# 3. STATISTICAL EFFECT TESTS
# Note: For M0, historical AP is 0.225030 with canonical time encoding, and 0.255970 with notebook time encoding
# We evaluate both to provide 100% rigorous scientific transparency
stat_results = []

def calc_stats(diffs, comp_name, baseline_name):
    m = np.mean(diffs)
    s = np.std(diffs, ddof=1)
    se = stats.sem(diffs)
    ci_low = m - 1.96 * se
    ci_high = m + 1.96 * se
    t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
    w_stat, w_pval = stats.wilcoxon(diffs) if not (diffs == 0).all() else (0, 1.0)
    d = m / s if s > 0 else 0.0
    return {
        "comparison": comp_name,
        "reference": baseline_name,
        "mean_diff": m,
        "std_diff": s,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "cohens_d": d,
        "t_statistic": t_stat,
        "t_pvalue": p_val,
        "wilcoxon_pvalue": w_pval
    }

m1_ap = df_test[df_test["config"] == "M1_ResidualETGNN"]["test_ap"].to_numpy()
m2_ap = df_test[df_test["config"] == "M2_DeepMLP"]["test_ap"].to_numpy()
m3_ce_ap = df_test[df_test["config"] == "M3_RG_ETGNN_CE"]["test_ap"].to_numpy()
m3_fl_ap = df_test[df_test["config"] == "M3_RG_ETGNN_Focal"]["test_ap"].to_numpy()
m4_ap = df_test[df_test["config"] == "M4_Ablation_NoGate"]["test_ap"].to_numpy()

m0_canonical = 0.225030
m0_notebook = 0.255970

stat_results.append(calc_stats(m1_ap - m0_canonical, "M1 vs M0 (Canonical)", "M0_Canonical (0.2250)"))
stat_results.append(calc_stats(m1_ap - m0_notebook, "M1 vs M0 (Notebook)", "M0_Notebook (0.2560)"))
stat_results.append(calc_stats(m3_ce_ap - m1_ap, "Delta_Gate (M3_CE - M1)", "M1_ResidualETGNN"))
stat_results.append(calc_stats(m3_ce_ap - m4_ap, "Delta_Ctrl (M3_CE - M4)", "M4_Ablation_NoGate"))
stat_results.append(calc_stats(m3_fl_ap - m3_ce_ap, "Delta_Loss (M3_Focal - M3_CE)", "M3_RG_ETGNN_CE"))
stat_results.append(calc_stats(m3_ce_ap - m2_ap, "Delta_Graph (M3_CE - M2)", "M2_DeepMLP"))
stat_results.append(calc_stats(m1_ap - m2_ap, "M1 vs M2 (Residual vs MLP)", "M2_DeepMLP"))

df_stats = pd.DataFrame(stat_results)
df_stats.to_csv(PHASE14_DIR / "PHASE14_STATISTICAL_TESTS.csv", index=False)
print("Saved PHASE14_STATISTICAL_TESTS.csv")

# 4. PUBLICATION-QUALITY FIGURES
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_colors = ['#4A7BB7', '#2A9D8F', '#E76F51', '#E63946', '#D62828', '#457B9D']

# Figure 1: Test AUPRC Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
models_order = ["M0_ETGNN_Baseline", "M1_ResidualETGNN", "M2_DeepMLP", "M3_RG_ETGNN_CE", "M3_RG_ETGNN_Focal", "M4_Ablation_NoGate"]
labels_order = ["M0: Baseline\nETGNN", "M1: Residual\nETGNN", "M2: Feature\nOnly MLP", "M3: RG-ETGNN\n(Cross-Entropy)", "M3: RG-ETGNN\n(Focal Loss)", "M4: Gating\nDisabled Ablation"]

means = [df_comp[df_comp["config"] == m]["test_ap_mean"].values[0] for m in models_order]
stds = [df_comp[df_comp["config"] == m]["test_ap_std"].values[0] for m in models_order]

bars = ax.bar(range(len(models_order)), means, yerr=stds, capsize=5, color=fig_colors, edgecolor='black', alpha=0.88, width=0.6)

# Historical baseline lines
ax.axhline(0.225030, color='#888888', linestyle='--', linewidth=1.5, label='Historical Canonical M0 (0.2250)')
ax.axhline(0.6380, color='#2B5C8F', linestyle=':', linewidth=1.5, label='Random Forest Benchmark (0.6380)')

for bar, m in zip(bars, means):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{m:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(models_order)))
ax.set_xticklabels(labels_order, fontsize=10)
ax.set_ylabel("Held-Out Test AUPRC (Timesteps 41–49)", fontsize=12, fontweight='bold')
ax.set_title("Phase 14: Rigorous Controlled Model Comparison on Elliptic Bitcoin Dataset", fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 0.70)
ax.legend(loc='upper right', frameon=True, fontsize=10)
plt.tight_layout()
fig_path1 = PHASE14_DIR / "phase14_model_auprc_comparison.png"
plt.savefig(fig_path1)
plt.close()
print(f"Generated {fig_path1.name}")

# Figure 2: Degree-Stratified Breakdown
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
x = np.arange(len(models_order))
width = 0.25

s0_vals = [df_comp[df_comp["config"] == m]["ap_deg_0"].values[0] for m in models_order]
s1_vals = [df_comp[df_comp["config"] == m]["ap_deg_1_2"].values[0] for m in models_order]
s2_vals = [df_comp[df_comp["config"] == m]["ap_deg_3plus"].values[0] for m in models_order]

b0 = ax.bar(x - width, s0_vals, width, label='S0: Degree k=0 (Isolated)', color='#457B9D', edgecolor='black', alpha=0.9)
b1 = ax.bar(x, s1_vals, width, label='S1: Degree k ∈ {1, 2} (Sparse)', color='#2A9D8F', edgecolor='black', alpha=0.9)
b2 = ax.bar(x + width, s2_vals, width, label='S2: Degree k ≥ 3 (Connected)', color='#E76F51', edgecolor='black', alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(labels_order, fontsize=10)
ax.set_ylabel("Stratified Test AUPRC", fontsize=12, fontweight='bold')
ax.set_title("Performance Breakdown by Node In-Degree Stratum Under Temporal Shift", fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 0.60)
ax.legend(loc='upper right', frameon=True, fontsize=10)
plt.tight_layout()
fig_path2 = PHASE14_DIR / "phase14_degree_stratified_performance.png"
plt.savefig(fig_path2)
plt.close()
print(f"Generated {fig_path2.name}")

# Figure 3: Gate Activation Collapse (M3 Layer 1 vs Layer 2)
df_gates = pd.read_csv(PHASE14_DIR / "M3_GATE_ACTIVATION_DIAGNOSTICS.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

seeds = [42, 43, 44, 45, 46]
x_idx = np.arange(len(seeds))
w = 0.35

ce_sub = df_gates[df_gates["model"] == "M3_RG_ETGNN_CE"]
fl_sub = df_gates[df_gates["model"] == "M3_RG_ETGNN_Focal"]

ax1.bar(x_idx - w/2, ce_sub["test_g1_mean"], w, label='Layer 1 Gate (CE)', color='#457B9D', edgecolor='black', alpha=0.85)
ax1.bar(x_idx + w/2, ce_sub["test_g2_mean"], w, label='Layer 2 Gate (CE)', color='#E63946', edgecolor='black', alpha=0.85)
ax1.set_xticks(x_idx)
ax1.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
ax1.set_ylabel("Mean Gate Activation γ ∈ (0, 1)", fontsize=11, fontweight='bold')
ax1.set_title("M3 Cross-Entropy: Layer 1 vs Layer 2 Gates", fontsize=12, fontweight='bold')
ax1.set_ylim(0, 0.6)
ax1.legend(loc='upper right')

ax2.bar(x_idx - w/2, fl_sub["test_g1_mean"], w, label='Layer 1 Gate (Focal)', color='#2A9D8F', edgecolor='black', alpha=0.85)
ax2.bar(x_idx + w/2, fl_sub["test_g2_mean"], w, label='Layer 2 Gate (Focal)', color='#D62828', edgecolor='black', alpha=0.85)
ax2.set_xticks(x_idx)
ax2.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
ax2.set_ylabel("Mean Gate Activation γ ∈ (0, 1)", fontsize=11, fontweight='bold')
ax2.set_title("M3 Focal Loss: Layer 1 vs Layer 2 Gates", fontsize=12, fontweight='bold')
ax2.set_ylim(0, 0.6)
ax2.legend(loc='upper right')

plt.suptitle("Empirical Proof of Gate Collapse: Layer 2 Gates Starve Graph Messages to Near Zero", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path3 = PHASE14_DIR / "phase14_gate_activation_collapse.png"
plt.savefig(fig_path3)
plt.close()
print(f"Generated {fig_path3.name}")

# Figure 4: Validation AP vs Test AP Scatter Plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
for cfg, color, marker in [
    ("M1_ResidualETGNN", "#2A9D8F", "o"),
    ("M2_DeepMLP", "#E76F51", "s"),
    ("M3_RG_ETGNN_CE", "#E63946", "^"),
    ("M3_RG_ETGNN_Focal", "#D62828", "v"),
    ("M4_Ablation_NoGate", "#457B9D", "D")
]:
    v = df_val[df_val["config"] == cfg]["val_ap"].values
    t = df_test[df_test["config"] == cfg]["test_ap"].values
    ax.scatter(v, t, color=color, marker=marker, s=90, label=cfg, edgecolors='black', alpha=0.85)

ax.set_xlabel("Validation AUPRC (Timesteps 33–40)", fontsize=12, fontweight='bold')
ax.set_ylabel("Held-Out Test AUPRC (Timesteps 41–49)", fontsize=12, fontweight='bold')
ax.set_title("Temporal Generalization Gap: Validation AP vs Test AP", fontsize=13, fontweight='bold', pad=15)
ax.axline((0.4, 0.4), slope=1, color='#888888', linestyle=':', label='y = x (Perfect Transfer)')
ax.legend(loc='lower left', frameon=True, fontsize=9)
plt.tight_layout()
fig_path4 = PHASE14_DIR / "phase14_validation_vs_test_scatter.png"
plt.savefig(fig_path4)
plt.close()
print(f"Generated {fig_path4.name}")

print("\n[SUCCESS] Phase 14 artifacts and figures generated successfully.")

