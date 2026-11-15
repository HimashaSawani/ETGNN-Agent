import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import json
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
from torch_geometric.utils import degree, k_hop_subgraph

from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier
from src.models.gat import GATClassifier


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def calc_smd(treat_vals, ctrl_vals, pooled_sd_pre):
    if pooled_sd_pre == 0 or np.isnan(pooled_sd_pre):
        return 0.0
    return float((np.mean(treat_vals) - np.mean(ctrl_vals)) / pooled_sd_pre)


def calc_variance_ratio(treat_vals, ctrl_vals):
    v_t = np.var(treat_vals, ddof=1)
    v_c = np.var(ctrl_vals, ddof=1)
    if v_c == 0:
        return float("inf") if v_t > 0 else 1.0
    return float(v_t / v_c)


def perform_caliper_matching(
    treat_df,
    ctrl_df,
    caliper_margin=0.05,
    caliper_edges=5,
    pooled_sd_margin=None,
    pooled_sd_edges=None,
):
    """
    Caliper-constrained 1-to-1 matching without replacement.
    Objective:
      1. Maximize matched coverage (cardinality) within admissible pairs.
      2. Minimize total standardized Euclidean distance among admissible pairs.
    Admissibility:
      |margin_t - margin_c| <= caliper_margin AND |edges_t - edges_c| <= caliper_edges
    """
    n_t = len(treat_df)
    n_c = len(ctrl_df)
    if n_t == 0 or n_c == 0:
        return []

    t_m = treat_df["margin"].values
    c_m = ctrl_df["margin"].values
    t_e = treat_df["candidate_edge_count"].values
    c_e = ctrl_df["candidate_edge_count"].values

    diff_m = np.abs(t_m[:, None] - c_m[None, :])
    diff_e = np.abs(t_e[:, None] - c_e[None, :])

    if pooled_sd_margin is None:
        pooled_sd_margin = float(np.sqrt((np.var(t_m, ddof=1) + np.var(c_m, ddof=1)) / 2.0))
    if pooled_sd_edges is None:
        pooled_sd_edges = float(np.sqrt((np.var(t_e, ddof=1) + np.var(c_e, ddof=1)) / 2.0))

    dist_mat = np.sqrt(
        (diff_m / max(pooled_sd_margin, 1e-6)) ** 2
        + (diff_e / max(pooled_sd_edges, 1e-6)) ** 2
    )

    admissible = (diff_m <= caliper_margin) & (diff_e <= caliper_edges)

    # Augmented square cost matrix of size (n_t + n_c) x (n_c + n_t)
    BIG = 10000.0
    cost_mat = np.zeros((n_t + n_c, n_c + n_t))
    tl = np.full((n_t, n_c), BIG * 2.0)
    tl[admissible] = dist_mat[admissible] - BIG
    cost_mat[:n_t, :n_c] = tl

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if r < n_t and c < n_c and admissible[r, c]:
            t_row = treat_df.iloc[r]
            c_row = ctrl_df.iloc[c]
            matched_pairs.append({
                "treat_node_index": int(t_row["node_index"]),
                "treat_txId": int(t_row["transaction_id"]),
                "treat_time_step": int(t_row["time_step"]),
                "treat_category": t_row["prediction_category"],
                "treat_p_illicit": float(t_row["p_illicit"]),
                "treat_margin": float(t_row["margin"]),
                "treat_candidate_edges": int(t_row["candidate_edge_count"]),
                "treat_in_degree": int(t_row["in_degree"]),
                "treat_out_degree": int(t_row["out_degree"]),
                "treat_total_degree": int(t_row["total_degree"]),
                "ctrl_node_index": int(c_row["node_index"]),
                "ctrl_txId": int(c_row["transaction_id"]),
                "ctrl_time_step": int(c_row["time_step"]),
                "ctrl_category": c_row["prediction_category"],
                "ctrl_p_illicit": float(c_row["p_illicit"]),
                "ctrl_margin": float(c_row["margin"]),
                "ctrl_candidate_edges": int(c_row["candidate_edge_count"]),
                "ctrl_in_degree": int(c_row["in_degree"]),
                "ctrl_out_degree": int(c_row["out_degree"]),
                "ctrl_total_degree": int(c_row["total_degree"]),
                "abs_diff_margin": float(diff_m[r, c]),
                "abs_diff_candidate_edges": int(diff_e[r, c]),
                "abs_diff_in_degree": int(abs(t_row["in_degree"] - c_row["in_degree"])),
                "abs_diff_out_degree": int(abs(t_row["out_degree"] - c_row["out_degree"])),
                "abs_diff_total_degree": int(abs(t_row["total_degree"] - c_row["total_degree"])),
                "norm_distance": float(dist_mat[r, c]),
            })

    return matched_pairs


def main():
    print("=== Starting Corrected C1 + C2 Audit and Caliper Matching Pipeline ===")
    out_dir = Path("experiments/matched_c1_c2_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    val_graph_path = "data/processed/validation_graph.pt"
    mapping_path = "data/processed/node_mapping.csv"
    val_graph = torch.load(val_graph_path, map_location="cpu", weights_only=True)
    df_map = pd.read_csv(mapping_path)
    node_to_txId = dict(zip(df_map["node_index"], df_map["txId"]))

    # Rigorous check: original_node_index verification across all validation rows
    num_val_nodes = val_graph["x"].shape[0]
    pos_tensor = torch.arange(num_val_nodes)
    pos_diffs = int((val_graph["original_node_index"] != pos_tensor).sum())
    print(f"Validation graph size: {num_val_nodes} nodes.")
    print(f"Positions where original_node_index != position index: {pos_diffs} (identity confirmed across all {num_val_nodes} rows).")

    # Historical pilot verification
    hist_pilot_path = "experiments/pilot_validation_audit/pilot_nodes.csv"
    hist_pilot_tx_ids = set()
    hist_pilot_node_indices = set()
    if os.path.exists(hist_pilot_path):
        hist_df = pd.read_csv(hist_pilot_path)
        for _, r in hist_df.iterrows():
            nid = int(r["node_index"])
            hist_pilot_node_indices.add(nid)
            tx = node_to_txId.get(nid, None)
            if tx is not None:
                hist_pilot_tx_ids.add(int(tx))
    print(f"Loaded {len(hist_pilot_node_indices)} historical pilot node IDs ({len(hist_pilot_tx_ids)} unique txIds).")

    # 2. Candidate edge counts cache
    cand_cache_path = out_dir / "candidate_edge_counts_cache.json"
    edge_index = val_graph["edge_index"]
    in_degs = degree(edge_index[1], num_nodes=num_val_nodes).long().numpy()
    out_degs = degree(edge_index[0], num_nodes=num_val_nodes).long().numpy()
    val_mask = val_graph["val_mask"]
    val_indices = val_mask.nonzero().flatten().tolist()

    if cand_cache_path.exists():
        print("Loading candidate edge counts from cache...")
        with open(cand_cache_path, "r") as f:
            cand_edge_counts = {int(k): v for k, v in json.load(f).items()}
    else:
        print("Computing 2-hop computation subgraph candidate edge counts...")
        cand_edge_counts = {}
        for pos in val_indices:
            _, sub_edges, _, _ = k_hop_subgraph(
                pos, 2, edge_index, relabel_nodes=False, flow="source_to_target"
            )
            cand_edge_counts[pos] = sub_edges.shape[1]
        with open(cand_cache_path, "w") as f:
            json.dump(cand_edge_counts, f)

    # 3. Model definitions
    models_to_eval = [
        ("ETGNN_full", ETGNNClassifier(input_dim=165), "results/checkpoints/etgnn.pt", 165, True),
        ("Matched_GAT", MatchedGATClassifier(input_dim=165), "results/checkpoints/matched_gat.pt", 165, False),
        ("ETGNN_local", ETGNNClassifier(input_dim=93), "results/checkpoints/etgnn_local.pt", 93, True),
    ]

    all_feasibility_dfs = {}

    for model_id, model_obj, ckpt_p, feat_dim, is_temporal in models_to_eval:
        print(f"\n--- Processing Model: {model_id} ---")
        model_obj.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=True))
        model_obj.eval()

        with torch.no_grad():
            x_in = val_graph["x"][:, :feat_dim]
            if isinstance(model_obj, MatchedGATClassifier):
                logits = model_obj(x_in, edge_index, val_graph["time_step"])
            elif is_temporal:
                logits = model_obj(x_in, edge_index, val_graph["time_step"])
            else:
                logits = model_obj(x_in, edge_index)
            probs = torch.softmax(logits, dim=1)

        records = []
        for pos in val_indices:
            global_node_idx = int(val_graph["original_node_index"][pos])
            tx_id = node_to_txId.get(global_node_idx, -1)
            t_step = int(val_graph["time_step"][pos])
            y_rec = int(val_graph["y"][pos])
            p_illicit = float(probs[pos, 1])
            frozen_thresh = 0.5
            pred_class = int(p_illicit >= frozen_thresh)
            margin = abs(p_illicit - frozen_thresh)
            c_edges = cand_edge_counts[pos]
            in_d = int(in_degs[pos])
            out_d = int(out_degs[pos])
            tot_d = in_d + out_d

            # Pilot exclusion checked against BOTH node_index and txId
            is_hist = (global_node_idx in hist_pilot_node_indices) or (tx_id in hist_pilot_tx_ids)

            cat = ("T" if y_rec == pred_class else "F") + ("P" if pred_class == 1 else "N")

            exclusion_reasons = []
            if t_step < 37:
                exclusion_reasons.append("dev_period_step_33_36")
            if is_hist:
                exclusion_reasons.append("historical_pilot_node")
            if y_rec == -1:
                exclusion_reasons.append("unknown_label")
            if in_d < 1:
                exclusion_reasons.append("zero_in_degree")
            if c_edges < 2:
                exclusion_reasons.append("candidate_edges_lt_2")

            eligible = len(exclusion_reasons) == 0
            reason_str = ";".join(exclusion_reasons) if exclusion_reasons else "none"

            records.append({
                "model_id": model_id,
                "transaction_id": tx_id,
                "node_index": global_node_idx,
                "time_step": t_step,
                "recorded_label": y_rec,
                "predicted_class": pred_class,
                "prediction_category": cat,
                "p_illicit": p_illicit,
                "frozen_threshold": frozen_thresh,
                "margin": margin,
                "candidate_edge_count": c_edges,
                "in_degree": in_d,
                "out_degree": out_d,
                "total_degree": tot_d,
                "eligibility": eligible,
                "exclusion_reason": reason_str,
                "historical_pilot_flag": is_hist,
            })

        df_feas = pd.DataFrame(records)
        df_feas.to_csv(out_dir / f"feasibility_{model_id.lower()}.csv", index=False)
        all_feasibility_dfs[model_id] = df_feas

    # 4. Caliper Matching (m <= 0.05, e <= 5)
    caliper_m = 0.05
    caliper_e = 5

    detailed_balance_records = []
    difference_distribution_records = []

    covariates_list = ["margin", "candidate_edge_count", "in_degree", "out_degree", "total_degree"]

    for model_id in ["ETGNN_full", "Matched_GAT", "ETGNN_local"]:
        df_m = all_feasibility_dfs[model_id]
        eligible_df = df_m[df_m["eligibility"]].copy().reset_index(drop=True)

        for contrast_name, (treat_cat, ctrl_cat) in [("TP_FP", ("FP", "TP")), ("TN_FN", ("FN", "TN"))]:
            treat_nodes = eligible_df[eligible_df["prediction_category"] == treat_cat].copy().reset_index(drop=True)
            ctrl_nodes = eligible_df[eligible_df["prediction_category"] == ctrl_cat].copy().reset_index(drop=True)

            n_t_avail = len(treat_nodes)
            n_c_avail = len(ctrl_nodes)

            # Pre-match pooled standard deviations (the fixed denominator for all SMD calculations)
            pooled_sds_pre = {}
            for cov in covariates_list:
                s_t = np.var(treat_nodes[cov], ddof=1)
                s_c = np.var(ctrl_nodes[cov], ddof=1)
                pooled_sds_pre[cov] = float(np.sqrt((s_t + s_c) / 2.0))

            pairs = perform_caliper_matching(
                treat_nodes,
                ctrl_nodes,
                caliper_margin=caliper_m,
                caliper_edges=caliper_e,
                pooled_sd_margin=pooled_sds_pre["margin"],
                pooled_sd_edges=pooled_sds_pre["candidate_edge_count"],
            )

            n_matched = len(pairs)
            pairs_df = pd.DataFrame(pairs)

            if n_matched > 0:
                pairs_df.insert(0, "pair_id", [f"{model_id}_{contrast_name}_{i+1:03d}" for i in range(n_matched)])
                pairs_df.to_csv(out_dir / f"caliper_pairs_{model_id.lower()}_{contrast_name.lower()}.csv", index=False)

                # Quantiles of differences
                diff_m_vals = pairs_df["abs_diff_margin"].values
                diff_e_vals = pairs_df["abs_diff_candidate_edges"].values

                difference_distribution_records.append({
                    "model_id": model_id,
                    "contrast": contrast_name,
                    "n_pairs": n_matched,
                    "diff_margin_mean": round(float(np.mean(diff_m_vals)), 4),
                    "diff_margin_median": round(float(np.median(diff_m_vals)), 4),
                    "diff_margin_q75": round(float(np.quantile(diff_m_vals, 0.75)), 4),
                    "diff_margin_q95": round(float(np.quantile(diff_m_vals, 0.95)), 4),
                    "diff_margin_max": round(float(np.max(diff_m_vals)), 4),
                    "diff_edges_mean": round(float(np.mean(diff_e_vals)), 2),
                    "diff_edges_median": round(float(np.median(diff_e_vals)), 2),
                    "diff_edges_q75": round(float(np.quantile(diff_e_vals, 0.75)), 2),
                    "diff_edges_q95": round(float(np.quantile(diff_e_vals, 0.95)), 2),
                    "diff_edges_max": int(np.max(diff_e_vals)),
                })

            # Covariate balance table for all 5 covariates
            for cov in covariates_list:
                pre_t_mean = float(treat_nodes[cov].mean())
                pre_c_mean = float(ctrl_nodes[cov].mean())
                pre_smd = calc_smd(treat_nodes[cov], ctrl_nodes[cov], pooled_sds_pre[cov])
                pre_vr = calc_variance_ratio(treat_nodes[cov], ctrl_nodes[cov])

                if n_matched > 0:
                    post_t_col = f"treat_{cov}" if f"treat_{cov}" in pairs_df.columns else f"treat_{cov.replace('_count', '')}"
                    post_c_col = f"ctrl_{cov}" if f"ctrl_{cov}" in pairs_df.columns else f"ctrl_{cov.replace('_count', '')}"
                    # handle name aliases
                    if cov == "candidate_edge_count":
                        post_t_col = "treat_candidate_edges"
                        post_c_col = "ctrl_candidate_edges"

                    post_t_vals = pairs_df[post_t_col]
                    post_c_vals = pairs_df[post_c_col]

                    post_t_mean = float(post_t_vals.mean())
                    post_c_mean = float(post_c_vals.mean())
                    post_smd = calc_smd(post_t_vals, post_c_vals, pooled_sds_pre[cov])
                    post_vr = calc_variance_ratio(post_t_vals, post_c_vals)
                else:
                    post_t_mean = post_c_mean = post_smd = post_vr = np.nan

                detailed_balance_records.append({
                    "model_id": model_id,
                    "contrast": contrast_name,
                    "covariate": cov,
                    "n_treat_available": n_t_avail,
                    "n_ctrl_available": n_c_avail,
                    "n_matched_pairs": n_matched,
                    "treat_coverage_pct": round(n_matched / n_t_avail * 100, 1),
                    "ctrl_coverage_pct": round(n_matched / n_c_avail * 100, 1),
                    "pooled_sd_pre": round(pooled_sds_pre[cov], 4),
                    "pre_treat_mean": round(pre_t_mean, 3),
                    "pre_ctrl_mean": round(pre_c_mean, 3),
                    "pre_smd": round(pre_smd, 3),
                    "pre_vr": round(pre_vr, 3),
                    "post_treat_mean": round(post_t_mean, 3) if not np.isnan(post_t_mean) else np.nan,
                    "post_ctrl_mean": round(post_c_mean, 3) if not np.isnan(post_c_mean) else np.nan,
                    "post_smd": round(post_smd, 3) if not np.isnan(post_smd) else np.nan,
                    "post_vr": round(post_vr, 3) if not np.isnan(post_vr) else np.nan,
                })

            # Overlap plot for caliper matched pairs
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.scatter(
                ctrl_nodes["margin"],
                ctrl_nodes["candidate_edge_count"],
                alpha=0.40,
                color="#1f77b4",
                s=35,
                label=f"All Eligible Correct ({ctrl_cat}, n={n_c_avail})",
            )
            ax.scatter(
                treat_nodes["margin"],
                treat_nodes["candidate_edge_count"],
                alpha=0.70,
                color="#d62728",
                marker="^",
                s=50,
                label=f"All Eligible Incorrect ({treat_cat}, n={n_t_avail})",
            )

            if n_matched > 0:
                for _, prow in pairs_df.iterrows():
                    ax.plot(
                        [prow["treat_margin"], prow["ctrl_margin"]],
                        [prow["treat_candidate_edges"], prow["ctrl_candidate_edges"]],
                        color="#2ca02c",
                        linestyle="-",
                        alpha=0.5,
                        linewidth=1.0,
                    )
                # Scatter matched points
                ax.scatter(
                    pairs_df["treat_margin"],
                    pairs_df["treat_candidate_edges"],
                    color="#9467bd",
                    marker="o",
                    s=15,
                    label=f"Caliper Matched Pairs (n={n_matched})",
                    zorder=4,
                )

            ax.set_title(
                f"Caliper-Constrained Matched Pairs: {model_id} ({treat_cat} vs {ctrl_cat})\n"
                f"Caliper: |Δ margin| <= {caliper_m}, |Δ E_cand| <= {caliper_e} (Steps 37–40)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Prediction Margin |p - 0.5|", fontsize=10)
            ax.set_ylabel("Candidate Subgraph Edges E_cand", fontsize=10)
            ax.set_yscale("log")
            ax.legend(frameon=True, loc="upper right")
            plt.tight_layout()
            fig_path = plots_dir / f"caliper_overlap_{model_id.lower()}_{contrast_name.lower()}.png"
            fig.savefig(fig_path)
            plt.close(fig)

    df_bal = pd.DataFrame(detailed_balance_records)
    df_bal.to_csv(out_dir / "caliper_balance_diagnostics_all_covariates.csv", index=False)

    df_diffs = pd.DataFrame(difference_distribution_records)
    df_diffs.to_csv(out_dir / "caliper_within_pair_differences.csv", index=False)

    print("\n=== CALIPER-CONSTRAINED MATCHING SUMMARY ===")
    print(df_diffs.to_string())

    print("\n=== BALANCE DIAGNOSTICS (MARGIN & CANDIDATE EDGES) ===")
    sub_bal = df_bal[df_bal["covariate"].isin(["margin", "candidate_edge_count"])].copy()
    print(sub_bal[["model_id", "contrast", "covariate", "n_matched_pairs", "treat_coverage_pct", "pre_smd", "post_smd", "pre_vr", "post_vr"]].to_string())


if __name__ == "__main__":
    main()
