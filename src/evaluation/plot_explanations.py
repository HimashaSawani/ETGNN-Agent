import gc
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load node mappings for exact txId resolution
mapping_df = pd.read_csv("data/processed/node_mapping.csv").set_index("node_index")

# -------------------------------------------------------------
# 1. Subgraph Attribution Visualizations (TP, FP, FN, TN)
# -------------------------------------------------------------
cases = [
    ("connected_tp.json", "True Positive (TP) — Correctly Identified Fraud", "explanation_tp_subgraph", 1),
    ("connected_fp.json", "False Positive (FP) — Licit Flagged as Fraud", "explanation_fp_subgraph", 0),
    ("connected_fn.json", "False Negative (FN) — Missed Fraud (Predicted Licit)", "explanation_fn_subgraph", 1),
    ("connected_tn.json", "True Negative (TN) — Correctly Identified Licit", "explanation_tn_subgraph", 0),
]

for filename, title, out_name, true_label in cases:
    path = Path("results/explanations") / filename
    if not path.exists():
        continue
    with open(path, "r") as f:
        data = json.load(f)

    center_node = data["node_index"]
    tx_id = mapping_df.loc[center_node, "txId"] if center_node in mapping_df.index else "N/A"
    time_step = mapping_df.loc[center_node, "time_step"] if center_node in mapping_df.index else "N/A"
    pred_class = data.get("predicted_class", 1 if data["case"] in ["TP", "FP"] else 0)
    orig_prob = data.get("original_predicted_class_probability", 0.0)

    top_edges = data.get("top_edges", [])
    
    G = nx.DiGraph()
    center_color = "#e74c3c" if pred_class == 1 else "#27ae60"
    G.add_node(center_node, role="Target Tx", color=center_color)
    
    edge_labels = {}
    for edge_info in top_edges:
        src = edge_info["source_node_index"]
        dst = edge_info["target_node_index"]
        mask_val = edge_info["mask_value"]
        if src != center_node:
            G.add_node(src, role="Neighbor Tx", color="#3498db")
        if dst != center_node:
            G.add_node(dst, role="Neighbor Tx", color="#3498db")
        G.add_edge(src, dst, weight=mask_val)
        edge_labels[(src, dst)] = f"{mask_val:.2f}"

    fig, (ax_graph, ax_feat) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.2, 1]})
    
    pos = nx.spring_layout(G, seed=42, k=0.8)
    node_colors = [G.nodes[n].get("color", "#95a5a6") for n in G.nodes]
    node_sizes = [1200 if n == center_node else 650 for n in G.nodes]
    
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=8, font_weight="bold")
    
    weights = [max(1.0, G[u][v]["weight"] * 3.5) for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax_graph, width=weights, edge_color="#2c3e50", arrowsize=12, arrowstyle="-|>")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax_graph, font_size=8)
    
    metadata_str = f"Target Node: {center_node} | TxID: {tx_id} | Time Step: {time_step}\nTrue: {'Illicit (1)' if true_label==1 else 'Licit (0)'} | Pred: {'Illicit (1)' if pred_class==1 else 'Licit (0)'} (P(c)={orig_prob:.2%})"
    ax_graph.set_title(f"Influential Subgraph Attribution\n{metadata_str}", pad=8, fontsize=9.5)
    ax_graph.axis("off")

    top_feat_indices = data.get("top_feature_columns_zero_based", [])[:8]
    top_feat_masks = data.get("top_feature_mask_values", [])[:8]
    feat_labels = [f"Feature {idx}" for idx in top_feat_indices]
    
    y_pos = np.arange(len(feat_labels))
    ax_feat.barh(y_pos, top_feat_masks, color="#9b59b6", edgecolor="black", alpha=0.85)
    ax_feat.set_yticks(y_pos)
    ax_feat.set_yticklabels(feat_labels)
    ax_feat.invert_yaxis()
    ax_feat.set_xlabel("Attribution Mask Weight", fontsize=9)
    ax_feat.set_title("Top Influential Node Features (Center Node)", pad=6, fontsize=10)
    
    fig.suptitle(f"[Validation Case Study (t=34)] {title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{out_name}.png", dpi=200)
    plt.savefig(FIG_DIR / f"{out_name}.pdf")
    plt.close(fig)
    del fig, ax_graph, ax_feat, G
    gc.collect()

# -------------------------------------------------------------
# 2. Explanation Controls Comparison: Top vs Random vs Bottom Deletion
# -------------------------------------------------------------
controls_path = Path("results/tables/explanation_edge_deletion_controls.csv")
if controls_path.exists():
    ctrl_df = pd.read_csv(controls_path)
    # Compare strategies across cases
    cases_order = ["TP", "FP", "FN", "TN"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    
    strategies = ["top", "random", "bottom"]
    strat_colors = {"top": "#e74c3c", "random": "#95a5a6", "bottom": "#3498db"}
    strat_labels = {"top": "Top-5 Deleted", "random": "Random-5 Deleted", "bottom": "Bottom-5 Deleted"}
    
    x = np.arange(len(cases_order))
    width = 0.25
    
    for idx, strat in enumerate(strategies):
        sub = ctrl_df[ctrl_df["strategy"] == strat].set_index("case").reindex(cases_order)
        # Plot signed probability drop: prob_before - prob_after
        drops = sub["signed_probability_drop"].values
        rects = ax.bar(x + (idx - 1) * width, drops, width, label=strat_labels[strat],
                       color=strat_colors[strat], edgecolor="black", alpha=0.85)
        for b in rects:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            ax.annotate(f"{h:+.2f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 2 if h >= 0 else -6),
                         textcoords="offset points", ha="center", va=va, fontsize=7.5)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Explanation Control Evaluation: Probability Drop Under Edge Deletion", pad=10, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Case: {c}" for c in cases_order], fontsize=9.5)
    ax.set_ylabel("Signed Probability Drop (Before - After)", fontsize=9)
    ax.legend(frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "explanation_deletion_controls.png", dpi=200)
    plt.savefig(FIG_DIR / "explanation_deletion_controls.pdf")
    plt.close(fig)
    del fig, ax, ctrl_df
    gc.collect()

# -------------------------------------------------------------
# 3. Feature vs Edge Mask Sparsity Analysis
# -------------------------------------------------------------
sparsity_path = Path("results/tables/explanation_mask_sparsity.csv")
if sparsity_path.exists():
    sp_df = pd.read_csv(sparsity_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    cases = ["TP", "FP", "FN", "TN"]
    thresholds = [0.3, 0.5, 0.7]
    
    for case in cases:
        c_df = sp_df[(sp_df["case"] == case) & (sp_df["mask_type"] == "center_features")]
        ax1.plot(c_df["threshold"], c_df["sparsity"], marker="o", label=f"Case: {case}")
        
        e_df = sp_df[(sp_df["case"] == case) & (sp_df["mask_type"] == "edges")]
        ax2.plot(e_df["threshold"], e_df["sparsity"], marker="s", label=f"Case: {case}")
        
    ax1.set_title("Node Feature Mask Sparsity", pad=8, fontsize=10.5)
    ax1.set_xlabel("Mask Threshold")
    ax1.set_ylabel("Sparsity Fraction (Entries < Threshold)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=True, fontsize=8)
    
    ax2.set_title("Edge Mask Sparsity", pad=8, fontsize=10.5)
    ax2.set_xlabel("Mask Threshold")
    ax2.set_ylabel("Sparsity Fraction (Entries < Threshold)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "explanation_mask_sparsity.png", dpi=200)
    plt.savefig(FIG_DIR / "explanation_mask_sparsity.pdf")
    plt.close(fig)
    del fig, ax1, ax2, sp_df
    gc.collect()

print("All explanation figures and controls generated successfully.")
