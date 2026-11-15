"""
generate_c1_c2_publication_artifacts.py
=======================================
Generates publication-grade figures (PNG at 200 DPI + vector PDF) and CSV tables
for the main C1 and C2 benchmark results on the Elliptic transaction network.

Methodological Notes:
- Contrast 1 (C1): Evaluates responses to shared ETGNN-ranked deletion sets.
  delta_adj = Adv_ETGNN - Adv_Matched_GAT, where Adv is GNNExplainer drop minus
  30-draw random comparator drop. Standard deviation across transactions is 0.096180.
- Contrast 2 (C2): Evaluates paired difference:
  Delta_pair = A(incorrect) - A(correct).
  For FP-TP pairs: Delta_pair = A(FP) - A(TP).
  A positive value indicates that explanation-minus-random deletion response
  was larger for false positives than true positives.
- Budget rule: k = max(1, ceil(0.25 * E_candidate)) on eligible edges in the
  defined computation neighbourhood.

Outputs:
Figures:
1. results/figures/c1_response_distributions (PNG + PDF)
2. results/figures/c2_paired_differences_ci (PNG + PDF)
3. results/figures/c2_primary_vs_degree_sensitivity (PNG + PDF)
4. results/figures/candidate_edges_deletion_budget (PNG + PDF)

Tables:
1. results/tables/c1_response_summary.csv
2. results/tables/c2_registers_bootstrap_summary.csv
3. results/tables/c2_sensitivity_comparison.csv
4. results/tables/candidate_edge_budget_summary.csv
"""

import os
from pathlib import Path
import json

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Styling configuration
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results" / "main_c1_c2_results"
DERIVED_DIR = RESULTS_DIR / "derived"

# Load verified data
c1_df = pd.read_csv(DERIVED_DIR / "c1_statistical_results_derived.csv")
c2_boot_df = pd.read_csv(RESULTS_DIR / "c2_bootstrap_results.csv")
c2_pairs_df = pd.read_csv(RESULTS_DIR / "c2_individual_pair_differences.csv")
summary_df = pd.read_csv(DERIVED_DIR / "main_benchmark_summary_derived.csv")


def generate_figure1_and_table1():
    print("Generating Figure 1: C1 Response Distributions...")
    
    delta_adj = c1_df["delta_adj"].values
    delta_raw = c1_df["delta_raw"].values
    
    mean_adj = float(np.mean(delta_adj))
    median_adj = float(np.median(delta_adj))
    std_adj = float(np.std(delta_adj, ddof=1))
    iqr_adj = float(np.percentile(delta_adj, 75) - np.percentile(delta_adj, 25))
    
    mean_raw = float(np.mean(delta_raw))
    median_raw = float(np.median(delta_raw))
    std_raw = float(np.std(delta_raw, ddof=1))
    iqr_raw = float(np.percentile(delta_raw, 75) - np.percentile(delta_raw, 25))

    # --- Plotting Figure 1 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Panel A: Histograms for delta_adj and delta_raw
    bins = np.linspace(-0.60, 0.65, 51)
    counts_adj, _ = np.histogram(delta_adj, bins=bins)
    assert counts_adj.sum() == 240, f"Histogram bin range excluded points! Expected 240, got {counts_adj.sum()}"
    
    ax1.hist(delta_adj, bins=bins, color="#2980b9", alpha=0.65, edgecolor="black", 
             label=f"Adjusted Diff (Δ_adj)\nMean = {mean_adj:+.4f}, SD = {std_adj:.4f}")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Zero Reference (No Difference)")
    ax1.axvline(mean_adj, color="#c0392b", linestyle="-", linewidth=1.8, label=f"Mean Δ_adj = {mean_adj:+.4f}")
    
    ax1.set_title("(A) C1: Shared Deletions Adjusted Diff (Δ_adj)", pad=10)
    ax1.set_xlabel("Random-Adjusted Response Difference (Δ_adj)\n[Full observed range covered; N = 240, SD = ±0.0962]")
    ax1.set_ylabel("Transaction Count (N = 240)")
    ax1.set_xlim(-0.62, 0.68)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Box plots comparing Raw vs Adjusted across ETGNN categories
    categories = ["Overall", "TP", "FP", "TN", "FN"]
    box_data_adj = [delta_adj]
    box_data_raw = [delta_raw]
    for cat in ["TP", "FP", "TN", "FN"]:
        sub = c1_df[c1_df["etgnn_category"] == cat]
        box_data_adj.append(sub["delta_adj"].values)
        box_data_raw.append(sub["delta_raw"].values)

    positions = np.arange(len(categories))
    width = 0.35

    bp1 = ax2.boxplot(box_data_adj, positions=positions - width/2, widths=width * 0.85, 
                      patch_artist=True, boxprops=dict(facecolor="#3498db", alpha=0.7),
                      medianprops=dict(color="black", linewidth=1.5),
                      showmeans=True, meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="#e74c3c", markersize=5))
    bp2 = ax2.boxplot(box_data_raw, positions=positions + width/2, widths=width * 0.85, 
                      patch_artist=True, boxprops=dict(facecolor="#95a5a6", alpha=0.6),
                      medianprops=dict(color="black", linewidth=1.5),
                      showmeans=True, meanprops=dict(marker="s", markeredgecolor="black", markerfacecolor="#2c3e50", markersize=5))

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f"{cat}\n(N={len(d)})" for cat, d in zip(categories, box_data_adj)])
    ax2.set_title("(B) Deletion Responses by ETGNN Prediction Category", pad=10)
    ax2.set_ylabel("Deletion Response Difference")
    ax2.legend([bp1["boxes"][0], bp2["boxes"][0], bp1["means"][0], bp2["means"][0]],
               ["Adjusted (Δ_adj)", "Raw (Δ_raw)", "Adjusted Mean", "Raw Mean"],
               loc="upper right", framealpha=0.9, fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "c1_response_distributions.png", dpi=200)
    plt.savefig(FIG_DIR / "c1_response_distributions.pdf")
    plt.close(fig)

    # --- Table 1: C1 Statistical Summary ---
    summary_rows = []
    for label, sub_df in [("Overall (N=240)", c1_df)] + [(f"{cat} (N={len(c1_df[c1_df['etgnn_category']==cat])})", c1_df[c1_df['etgnn_category']==cat]) for cat in ["TP", "FP", "TN", "FN"]]:
        summary_rows.append({
            "Cohort": label,
            "N": len(sub_df),
            "delta_adj_mean": sub_df["delta_adj"].mean(),
            "delta_adj_sample_sd": sub_df["delta_adj"].std(ddof=1),
            "delta_adj_median": sub_df["delta_adj"].median(),
            "delta_adj_iqr": sub_df["delta_adj"].quantile(0.75) - sub_df["delta_adj"].quantile(0.25),
            "delta_raw_mean": sub_df["delta_raw"].mean(),
            "delta_raw_sample_sd": sub_df["delta_raw"].std(ddof=1),
            "delta_raw_median": sub_df["delta_raw"].median(),
            "adv_etgnn_mean": sub_df["adv_etgnn"].mean(),
            "adv_gat_common_mean": sub_df["adv_gat_common"].mean(),
            "note": "Sample SD is transaction variance across cohort; not standard error of mean."
        })
    df_t1 = pd.DataFrame(summary_rows)
    df_t1.to_csv(TABLE_DIR / "c1_response_summary.csv", index=False)
    print("  Saved Figure 1 and Table 1.")


def generate_figure2_and_table2():
    print("Generating Figure 2: C2 Paired Differences & Confidence Intervals (All 8 Registers)...")
    
    # Order registers: ETGNN registers first, then Matched GAT registers
    reg_order = [
        "ETGNN Primary TP-FP",
        "ETGNN Degree-Sensitivity TP-FP",
        "ETGNN Primary TN-FN",
        "ETGNN Degree-Sensitivity TN-FN",
        "Matched GAT Primary TP-FP",
        "Matched GAT Degree-Sensitivity TP-FP",
        "Matched GAT Primary TN-FN",
        "Matched GAT Degree-Sensitivity TN-FN",
    ]
    
    df = c2_boot_df.set_index("register").loc[reg_order].reset_index()
    
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    
    y_pos = np.arange(len(df))
    colors = ["#2980b9" if "ETGNN" in r else "#d35400" for r in df["register"]]
    
    # Plot zero reference line
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label="Zero Difference: A(incorrect) = A(correct)")
    
    for i, row in df.iterrows():
        y = len(df) - 1 - i
        mean_val = row["mean_diff"]
        low_val = row["ci_95_low"]
        high_val = row["ci_95_high"]
        n_pairs = row["pairs_count"]
        col = colors[i]
        
        # Error bar
        ax.errorbar(mean_val, y, xerr=[[mean_val - low_val], [high_val - mean_val]],
                    fmt="o", color=col, ecolor=col, elinewidth=2.2, capsize=5, capthick=1.8, markersize=7)
        
        # Label with exact CI values
        ci_str = f"{mean_val:+.4f} [{low_val:+.4f}, {high_val:+.4f}] (N={n_pairs})"
        ax.text(high_val + 0.005, y, ci_str, va="center", ha="left", fontsize=8.5, fontweight="semibold", color="#2c3e50")

    labels_reversed = [f"{r}" for r in df["register"][::-1]]
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(labels_reversed, fontweight="medium")
    ax.set_xlabel("Mean Paired Difference: Δ_pair = A(incorrect) − A(correct)\n[Positive => Advantage larger for errors (FP or FN); 95% Marginal Percentile Bootstrap CI, B = 2,000]")
    ax.set_title("Contrast 2: Paired Differences & 95% Marginal Bootstrap CIs (All 8 Registers)\n[Δ_pair = A(FP) − A(TP) for TP-FP; Δ_pair = A(FN) − A(TN) for TN-FN]", pad=28)
    ax.set_xlim(-0.16, 0.28)
    ax.set_ylim(-0.8, 7.8)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    
    custom_lines = [
        Line2D([0], [0], color="#2980b9", lw=2.5, marker="o", label="ETGNN Registers"),
        Line2D([0], [0], color="#d35400", lw=2.5, marker="o", label="Matched GAT Registers"),
        Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="Zero Difference (A_err = A_corr)"),
    ]
    # Place legend above plot to ensure 100% clean clearance from all points, error bars, and labels
    ax.legend(handles=custom_lines, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, framealpha=0.95, fontsize=8.8)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "c2_paired_differences_ci.png", dpi=200)
    plt.savefig(FIG_DIR / "c2_paired_differences_ci.pdf")
    plt.close(fig)

    # --- Table 2: Full C2 Bootstrap Summary Table ---
    df_t2 = df.copy()
    df_t2["model"] = ["ETGNN" if "ETGNN" in r else "Matched GAT" for r in df_t2["register"]]
    df_t2["contrast_formula"] = ["A(FP) - A(TP)" if "TP-FP" in r else "A(FN) - A(TN)" for r in df_t2["register"]]
    df_t2["match_type"] = ["Degree-Sensitivity" if "Degree-Sensitivity" in r else "Primary Caliper" for r in df_t2["register"]]
    df_t2["ci_crosses_zero"] = (df_t2["ci_95_low"] <= 0) & (df_t2["ci_95_high"] >= 0)
    df_t2["interpretation"] = [
        "Advantage larger for FP than TP (CI excludes 0)" if not row.ci_crosses_zero and "TP-FP" in row.register and row.mean_diff > 0
        else "CI spans zero; sensitive to matching / imprecise"
        for _, row in df_t2.iterrows()
    ]
    df_t2.to_csv(TABLE_DIR / "c2_registers_bootstrap_summary.csv", index=False)
    print("  Saved Figure 2 and Table 2.")


def generate_figure3_and_table3():
    print("Generating Figure 3: Primary vs Degree-Sensitivity Results...")
    
    comparisons = [
        ("ETGNN", "ETGNN Primary TP-FP", "ETGNN Degree-Sensitivity TP-FP", "#1f77b4"),
        ("Matched GAT", "Matched GAT Primary TP-FP", "Matched GAT Degree-Sensitivity TP-FP", "#d35400"),
        ("ETGNN", "ETGNN Primary TN-FN", "ETGNN Degree-Sensitivity TN-FN", "#2ca02c"),
        ("Matched GAT", "Matched GAT Primary TN-FN", "Matched GAT Degree-Sensitivity TN-FN", "#c0392b"),
    ]
    
    # Use independent categorical axes (sharey=False) to prevent cross-panel label contamination
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=False)
    boot_map = c2_boot_df.set_index("register").to_dict(orient="index")
    table3_rows = []
    
    panel_specs = [
        (axes[0], "(A) Illicit Domain: Δ_pair = A(FP) − A(TP)", comparisons[:2], [-0.035, 0.125]),
        (axes[1], "(B) Licit/Imprecise Domain: Δ_pair = A(FN) − A(TN)", comparisons[2:], [-0.17, 0.25])
    ]
    
    for ax, title, comps, xlims in panel_specs:
        y_indices = [1.1, 0.35]
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        
        for idx, (model_label, prim_key, deg_key, color) in enumerate(comps):
            y_base = y_indices[idx]
            prim_row = boot_map[prim_key]
            deg_row = boot_map[deg_key]
            
            # Primary point
            ax.errorbar(prim_row["mean_diff"], y_base + 0.12, 
                        xerr=[[prim_row["mean_diff"] - prim_row["ci_95_low"]], [prim_row["ci_95_high"] - prim_row["mean_diff"]]],
                        fmt="o", color=color, ecolor=color, elinewidth=2.2, capsize=4, markersize=8,
                        label=f"{model_label} Primary (N={prim_row['pairs_count']})")
            
            # Degree sensitivity point
            ax.errorbar(deg_row["mean_diff"], y_base - 0.12, 
                        xerr=[[deg_row["mean_diff"] - deg_row["ci_95_low"]], [deg_row["ci_95_high"] - deg_row["mean_diff"]]],
                        fmt="^", color=color, ecolor=color, elinewidth=2.0, capsize=4, markersize=8, alpha=0.8,
                        label=f"{model_label} Deg. Sens. (N={deg_row['pairs_count']})")
            
            # Connect with dotted line
            ax.plot([prim_row["mean_diff"], deg_row["mean_diff"]], [y_base + 0.12, y_base - 0.12], 
                    color=color, linestyle=":", alpha=0.6, linewidth=1.2)
            
            shift = deg_row["mean_diff"] - prim_row["mean_diff"]
            # Position annotation carefully in non-conflicting space
            x_annot = max(prim_row["ci_95_high"], deg_row["ci_95_high"]) + 0.006
            ax.annotate(f"Shift: {shift:+.4f}\nPrim: [{prim_row['ci_95_low']:+.3f}, {prim_row['ci_95_high']:+.3f}]\nDeg: [{deg_row['ci_95_low']:+.3f}, {deg_row['ci_95_high']:+.3f}]",
                        xy=((prim_row["mean_diff"] + deg_row["mean_diff"])/2, y_base),
                        xytext=(x_annot, y_base),
                        textcoords="data",
                        fontsize=7.3, va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.88, edgecolor=color))
            
            table3_rows.append({
                "Comparison": f"{model_label}: " + ("FP vs TP" if "TP-FP" in prim_key else "FN vs TN"),
                "Formula": "A(FP) - A(TP)" if "TP-FP" in prim_key else "A(FN) - A(TN)",
                "Primary_Register": prim_key,
                "Primary_N": prim_row["pairs_count"],
                "Primary_Mean": prim_row["mean_diff"],
                "Primary_CI_Low": prim_row["ci_95_low"],
                "Primary_CI_High": prim_row["ci_95_high"],
                "Degree_Register": deg_key,
                "Degree_N": deg_row["pairs_count"],
                "Degree_Mean": deg_row["mean_diff"],
                "Degree_CI_Low": deg_row["ci_95_low"],
                "Degree_CI_High": deg_row["ci_95_high"],
                "Mean_Shift": shift,
                "Primary_Crosses_Zero": prim_row["ci_95_low"] <= 0 <= prim_row["ci_95_high"],
                "Degree_Crosses_Zero": deg_row["ci_95_low"] <= 0 <= deg_row["ci_95_high"],
                "Sample_Overlap_Note": "Primary and Degree sensitivity registers represent different, overlapping matched samples."
            })

        ax.set_title(title, pad=10, fontsize=10.5)
        ax.set_xlabel("Mean Paired Difference [95% Marginal CI]\n(Positive => Larger advantage on errors)", fontsize=9)
        ax.set_yticks(y_indices)
        ax.set_yticklabels(["ETGNN", "Matched GAT"], fontweight="semibold", fontsize=9.5)
        ax.set_ylim(-0.05, 1.7)
        ax.set_xlim(xlims)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.legend(loc="upper left" if "FP" in title else "upper right", fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "c2_primary_vs_degree_sensitivity.png", dpi=200)
    plt.savefig(FIG_DIR / "c2_primary_vs_degree_sensitivity.pdf")
    plt.close(fig)

    df_t3 = pd.DataFrame(table3_rows)
    df_t3.to_csv(TABLE_DIR / "c2_sensitivity_comparison.csv", index=False)
    print("  Saved Figure 3 and Table 3.")


def generate_figure4_and_table4():
    print("Generating Figure 4: Candidate-Edge Counts & Actual Deletion Fractions...")
    
    # 377 transactions
    e_counts = summary_df["E_cand"].value_counts().sort_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    
    # Panel A: Distribution of candidate edges
    edge_vals = e_counts.index.values
    counts = e_counts.values
    pcts = counts / len(summary_df) * 100
    
    colors = ["#c0392b" if e == 2 else "#2980b9" for e in edge_vals]
    bars = ax1.bar([str(e) for e in edge_vals], counts, color=colors, edgecolor="black", width=0.55, alpha=0.85)
    
    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        ax1.annotate(f"{h:,}\n({pct:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax1.set_title("(A) Eligible Edges in Computation Neighbourhood ($E_{cand}$)", pad=12)
    ax1.set_xlabel("Number of Eligible Edges in Computation Neighbourhood ($E_{cand}$)")
    ax1.set_ylabel("Transaction Count (Total = 377)")
    ax1.set_ylim(0, max(counts) * 1.18)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    
    ax1.annotate("317 of 377 transactions (84.1%)\nhave exactly 2 eligible edges!",
                 xy=(0, 317), xytext=(0.8, 250),
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#c0392b"))

    # Panel B: Actual Deletion Fraction vs Nominal Rule k = max(1, ceil(0.25 * E_cand))
    edge_table = summary_df.groupby("E_cand").agg(
        count=("node_index", "count"),
        k=("k_budget", "first"),
        actual_frac=("actual_k_over_E", "first")
    ).reset_index()
    
    edge_table["nominal_frac"] = 0.25
    edge_table["inflation_ratio"] = edge_table["actual_frac"] / edge_table["nominal_frac"]
    edge_table["pct_population"] = edge_table["count"] / len(summary_df) * 100

    x_labels = [f"E={row.E_cand}\n(k={row.k})" for _, row in edge_table.iterrows()]
    bar_cols = ["#e74c3c" if row.actual_frac > 0.25 else "#27ae60" for _, row in edge_table.iterrows()]
    
    b2 = ax2.bar(x_labels, edge_table["actual_frac"] * 100, color=bar_cols, edgecolor="black", width=0.55, alpha=0.85)
    ax2.axhline(25, color="black", linestyle="--", linewidth=1.5, label="Nominal Budget Target (α = 25%)")
    
    for bar, _, row in zip(b2, x_labels, edge_table.itertuples()):
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%\n({row.count} tx)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_title("(B) Actual Deletion Fraction: k / $E_{cand}$\n[Rule: k = max(1, ceil(0.25 × $E_{cand}$))]", pad=12)
    ax2.set_xlabel("Eligible Neighbourhood Edges ($E_{cand}$) & Deleted Edges ($k$)")
    ax2.set_ylabel("Actual Deletion Fraction (%)")
    ax2.set_ylim(0, 60)
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "candidate_edges_deletion_budget.png", dpi=200)
    plt.savefig(FIG_DIR / "candidate_edges_deletion_budget.pdf")
    plt.close(fig)

    edge_table.to_csv(TABLE_DIR / "candidate_edge_budget_summary.csv", index=False)
    print("  Saved Figure 4 and Table 4.")


if __name__ == "__main__":
    generate_figure1_and_table1()
    generate_figure2_and_table2()
    generate_figure3_and_table3()
    generate_figure4_and_table4()
    print("\nALL PUBLICATION FIGURES AND TABLES SUCCESSFULLY CREATED.")
