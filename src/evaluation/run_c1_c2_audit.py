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


def standardized_mean_diff(x_treat, x_ctrl):
    m_t, m_c = np.mean(x_treat), np.mean(x_ctrl)
    s_t, s_c = np.var(x_treat, ddof=1), np.var(x_ctrl, ddof=1)
    pooled_s = np.sqrt((s_t + s_c) / 2.0)
    if pooled_s == 0:
        return 0.0
    return float((m_t - m_c) / pooled_s)


def variance_ratio(x_treat, x_ctrl):
    s_t, s_c = np.var(x_treat, ddof=1), np.var(x_ctrl, ddof=1)
    if s_c == 0:
        return float("inf") if s_t > 0 else 1.0
    return float(s_t / s_c)


def perform_matching(
    treat_df,
    ctrl_df,
    covariates=["margin", "candidate_edge_count"],
    order_by="margin_asc",
    caliper=None,
):
    """
    1-to-1 matching without replacement.
    treat_df: incorrect nodes (e.g. FP for TP-FP, FN for TN-FN)
    ctrl_df: correct nodes (e.g. TP for TP-FP, TN for TN-FN)
    covariates: list of covariate names to match on
    order_by: 'margin_asc', 'margin_desc', 'edges_asc', 'optimal'
    """
    if len(treat_df) == 0 or len(ctrl_df) == 0:
        return [], pd.DataFrame()

    # Normalize covariates across pooled sample
    pooled = pd.concat([treat_df[covariates], ctrl_df[covariates]])
    means = pooled.mean()
    stds = pooled.std().replace(0, 1.0)

    t_norm = ((treat_df[covariates] - means) / stds).values
    c_norm = ((ctrl_df[covariates] - means) / stds).values

    # Cost matrix: Euclidean distance in standardized space
    dist_matrix = cdist(t_norm, c_norm, metric="euclidean")

    if order_by == "optimal":
        # Global minimum distance matching
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            d = dist_matrix[r, c]
            if caliper is not None and d > caliper:
                continue
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
                "ctrl_node_index": int(c_row["node_index"]),
                "ctrl_txId": int(c_row["transaction_id"]),
                "ctrl_time_step": int(c_row["time_step"]),
                "ctrl_category": c_row["prediction_category"],
                "ctrl_p_illicit": float(c_row["p_illicit"]),
                "ctrl_margin": float(c_row["margin"]),
                "ctrl_candidate_edges": int(c_row["candidate_edge_count"]),
                "ctrl_in_degree": int(c_row["in_degree"]),
                "norm_distance": float(d),
            })
        return matched_pairs, dist_matrix

    # Greedy matching with specific order
    t_df_copy = treat_df.copy().reset_index(drop=True)
    if order_by == "margin_asc":
        order_indices = t_df_copy["margin"].argsort().values
    elif order_by == "margin_desc":
        order_indices = (-t_df_copy["margin"]).argsort().values
    elif order_by == "edges_asc":
        order_indices = t_df_copy["candidate_edge_count"].argsort().values
    elif order_by == "random":
        rng = np.random.default_rng(42)
        order_indices = rng.permutation(len(t_df_copy))
    else:
        order_indices = np.arange(len(t_df_copy))

    matched_pairs = []
    used_ctrl_indices = set()

    for r in order_indices:
        dists = dist_matrix[r].copy()
        # mask already used
        for u in used_ctrl_indices:
            dists[u] = np.inf
        best_c = np.argmin(dists)
        best_d = dists[best_c]
        if np.isinf(best_d):
            break
        if caliper is not None and best_d > caliper:
            continue
        used_ctrl_indices.add(best_c)
        t_row = t_df_copy.iloc[r]
        c_row = ctrl_df.iloc[best_c]
        matched_pairs.append({
            "treat_node_index": int(t_row["node_index"]),
            "treat_txId": int(t_row["transaction_id"]),
            "treat_time_step": int(t_row["time_step"]),
            "treat_category": t_row["prediction_category"],
            "treat_p_illicit": float(t_row["p_illicit"]),
            "treat_margin": float(t_row["margin"]),
            "treat_candidate_edges": int(t_row["candidate_edge_count"]),
            "treat_in_degree": int(t_row["in_degree"]),
            "ctrl_node_index": int(c_row["node_index"]),
            "ctrl_txId": int(c_row["transaction_id"]),
            "ctrl_time_step": int(c_row["time_step"]),
            "ctrl_category": c_row["prediction_category"],
            "ctrl_p_illicit": float(c_row["p_illicit"]),
            "ctrl_margin": float(c_row["margin"]),
            "ctrl_candidate_edges": int(c_row["candidate_edge_count"]),
            "ctrl_in_degree": int(c_row["in_degree"]),
            "norm_distance": float(best_d),
        })

    return matched_pairs, dist_matrix


def main():
    print("=== Starting C1 + C2 Audit and Feasibility Pipeline ===")
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

    # 2. Historical pilot nodes
    hist_pilot_path = "experiments/pilot_validation_audit/pilot_nodes.csv"
    hist_pilot_nodes = set()
    if os.path.exists(hist_pilot_path):
        hist_df = pd.read_csv(hist_pilot_path)
        hist_pilot_nodes = set(hist_df["node_index"].astype(int).tolist())
    print(f"Loaded {len(hist_pilot_nodes)} historical pilot node IDs.")

    # 3. Precompute degrees & candidate edges for validation nodes
    edge_index = val_graph["edge_index"]
    num_nodes = val_graph["x"].shape[0]
    in_degs = degree(edge_index[1], num_nodes=num_nodes).long().numpy()
    out_degs = degree(edge_index[0], num_nodes=num_nodes).long().numpy()

    val_mask = val_graph["val_mask"]
    val_indices = val_mask.nonzero().flatten().tolist()
    print(f"Total validation nodes to inspect: {len(val_indices)}")

    cand_cache_path = out_dir / "candidate_edge_counts_cache.json"
    if cand_cache_path.exists():
        print("Loading candidate edge counts from cache...")
        with open(cand_cache_path, "r") as f:
            cand_edge_counts = {int(k): v for k, v in json.load(f).items()}
    else:
        print("Computing 2-hop computation subgraph candidate edge counts...")
        t0 = time.perf_counter()
        cand_edge_counts = {}
        for pos in val_indices:
            _, sub_edges, _, _ = k_hop_subgraph(
                pos, 2, edge_index, relabel_nodes=False, flow="source_to_target"
            )
            cand_edge_counts[pos] = sub_edges.shape[1]
        t1 = time.perf_counter()
        print(f"Computed candidate edge counts in {t1 - t0:.2f} s.")
        with open(cand_cache_path, "w") as f:
            json.dump(cand_edge_counts, f)


    # 4. Model definitions to evaluate
    models_to_eval = [
        ("ETGNN_local", ETGNNClassifier(input_dim=93), "results/checkpoints/etgnn_local.pt", 93, True),
        ("ETGNN_full", ETGNNClassifier(input_dim=165), "results/checkpoints/etgnn.pt", 165, True),
        ("Matched_GAT", MatchedGATClassifier(input_dim=165), "results/checkpoints/matched_gat.pt", 165, False),
    ]

    all_feasibility_dfs = {}

    for model_id, model_obj, ckpt_p, feat_dim, is_temporal in models_to_eval:
        print(f"\n--- Evaluating Model: {model_id} ---")
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
            is_hist = bool(global_node_idx in hist_pilot_nodes)

            # Prediction category
            cat = ("T" if y_rec == pred_class else "F") + ("P" if pred_class == 1 else "N")

            # Eligibility determination
            # Main evaluation protocol: steps 37-40, not historical pilot, candidate_edge_count >= 2, in_degree >= 1
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
        print(f"Saved feasibility_{model_id.lower()}.csv ({len(df_feas)} rows)")
        print(f"Eligible main follow-up nodes (steps 37-40): {df_feas['eligibility'].sum()}")
        print("Breakdown of categories among eligible nodes:")
        print(df_feas[df_feas["eligibility"]]["prediction_category"].value_counts().to_dict())

    # 5. Matching analysis for ETGNN_local, ETGNN_full, and Matched_GAT
    matching_summary = []
    balance_tables = {}

    for model_id in ["ETGNN_local", "ETGNN_full", "Matched_GAT"]:
        df_m = all_feasibility_dfs[model_id]
        # Restrict to eligible nodes in main follow-up period (steps 37-40, not historical pilot, cand_edges >= 2)
        eligible_df = df_m[df_m["eligibility"]].copy()

        for contrast_name, (treat_cat, ctrl_cat) in [("TP_FP", ("FP", "TP")), ("TN_FN", ("FN", "TN"))]:
            treat_nodes = eligible_df[eligible_df["prediction_category"] == treat_cat].copy()
            ctrl_nodes = eligible_df[eligible_df["prediction_category"] == ctrl_cat].copy()

            n_treat = len(treat_nodes)
            n_ctrl = len(ctrl_nodes)

            # Test order sensitivity: optimal, margin_asc, margin_desc, edges_asc, random
            orders = ["optimal", "margin_asc", "margin_desc", "edges_asc", "random"]
            order_results = {}

            for ord_name in orders:
                pairs, dist_mat = perform_matching(
                    treat_nodes, ctrl_nodes, covariates=["margin", "candidate_edge_count"], order_by=ord_name
                )
                order_results[ord_name] = pairs

            # Select 'optimal' as primary deterministic matched set
            primary_pairs = order_results["optimal"]
            n_matched = len(primary_pairs)
            n_discarded_treat = n_treat - n_matched
            n_discarded_ctrl = n_ctrl - n_matched
            coverage = n_matched / n_treat if n_treat > 0 else 0.0

            # Save primary matched pairs
            pairs_df = pd.DataFrame(primary_pairs)
            if len(pairs_df) > 0:
                pairs_df.insert(0, "pair_id", [f"{model_id}_{contrast_name}_{i+1:03d}" for i in range(len(pairs_df))])
                pairs_df.to_csv(out_dir / f"matched_pairs_{model_id.lower()}_{contrast_name.lower()}.csv", index=False)

            # Balance diagnostics: pre-match vs post-match
            # Pre-match
            pre_smd_margin = standardized_mean_diff(treat_nodes["margin"], ctrl_nodes["margin"])
            pre_vr_margin = variance_ratio(treat_nodes["margin"], ctrl_nodes["margin"])
            pre_smd_edges = standardized_mean_diff(treat_nodes["candidate_edge_count"], ctrl_nodes["candidate_edge_count"])
            pre_vr_edges = variance_ratio(treat_nodes["candidate_edge_count"], ctrl_nodes["candidate_edge_count"])

            if n_matched > 0:
                post_t_margin = pairs_df["treat_margin"]
                post_c_margin = pairs_df["ctrl_margin"]
                post_t_edges = pairs_df["treat_candidate_edges"]
                post_c_edges = pairs_df["ctrl_candidate_edges"]

                post_smd_margin = standardized_mean_diff(post_t_margin, post_c_margin)
                post_vr_margin = variance_ratio(post_t_margin, post_c_margin)
                post_smd_edges = standardized_mean_diff(post_t_edges, post_c_edges)
                post_vr_edges = variance_ratio(post_t_edges, post_c_edges)
                mean_dist = float(pairs_df["norm_distance"].mean())
                max_dist = float(pairs_df["norm_distance"].max())
            else:
                post_smd_margin = post_vr_margin = post_smd_edges = post_vr_edges = mean_dist = max_dist = np.nan

            balance_row = {
                "model_id": model_id,
                "contrast": contrast_name,
                "treat_category": treat_cat,
                "ctrl_category": ctrl_cat,
                "n_treat_available": n_treat,
                "n_ctrl_available": n_ctrl,
                "n_matched_pairs": n_matched,
                "coverage_pct": round(coverage * 100, 1),
                "discarded_treat": n_discarded_treat,
                "discarded_ctrl": n_discarded_ctrl,
                "pre_smd_margin": round(pre_smd_margin, 3),
                "post_smd_margin": round(post_smd_margin, 3) if not np.isnan(post_smd_margin) else np.nan,
                "pre_vr_margin": round(pre_vr_margin, 3),
                "post_vr_margin": round(post_vr_margin, 3) if not np.isnan(post_vr_margin) else np.nan,
                "pre_smd_edges": round(pre_smd_edges, 3),
                "post_smd_edges": round(post_smd_edges, 3) if not np.isnan(post_smd_edges) else np.nan,
                "pre_vr_edges": round(pre_vr_edges, 3),
                "post_vr_edges": round(post_vr_edges, 3) if not np.isnan(post_vr_edges) else np.nan,
                "mean_norm_distance": round(mean_dist, 3) if not np.isnan(mean_dist) else np.nan,
                "max_norm_distance": round(max_dist, 3) if not np.isnan(max_dist) else np.nan,
            }
            matching_summary.append(balance_row)

            # Order sensitivity summary
            order_comp = []
            for ord_k, ord_pairs in order_results.items():
                if len(ord_pairs) > 0:
                    odf = pd.DataFrame(ord_pairs)
                    smd_m = standardized_mean_diff(odf["treat_margin"], odf["ctrl_margin"])
                    smd_e = standardized_mean_diff(odf["treat_candidate_edges"], odf["ctrl_candidate_edges"])
                    m_dist = float(odf["norm_distance"].mean())
                    order_comp.append({
                        "order_method": ord_k,
                        "n_pairs": len(ord_pairs),
                        "mean_distance": round(m_dist, 3),
                        "post_smd_margin": round(smd_m, 3),
                        "post_smd_edges": round(smd_e, 3),
                    })
            pd.DataFrame(order_comp).to_csv(
                out_dir / f"order_sensitivity_{model_id.lower()}_{contrast_name.lower()}.csv", index=False
            )

            # 6. Generate Overlap Plots
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.scatter(
                ctrl_nodes["margin"],
                ctrl_nodes["candidate_edge_count"],
                alpha=0.45,
                color="#1f77b4",
                s=40,
                label=f"Correct ({ctrl_cat}, n={n_ctrl})",
            )
            ax.scatter(
                treat_nodes["margin"],
                treat_nodes["candidate_edge_count"],
                alpha=0.75,
                color="#d62728",
                marker="^",
                s=55,
                label=f"Incorrect ({treat_cat}, n={n_treat})",
            )

            # If matched pairs exist, draw connecting lines for matched pairs
            if n_matched > 0:
                for _, prow in pairs_df.iterrows():
                    ax.plot(
                        [prow["treat_margin"], prow["ctrl_margin"]],
                        [prow["treat_candidate_edges"], prow["ctrl_candidate_edges"]],
                        color="gray",
                        linestyle="--",
                        alpha=0.4,
                        linewidth=0.8,
                    )

            ax.set_title(
                f"Covariate Overlap & Matched Pairs: {model_id} ({treat_cat} vs {ctrl_cat})\n"
                "Main Follow-up Steps 37–40 (Excl. Pilot, Cand Edges >= 2)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Prediction Margin $|p - 0.5|$", fontsize=10)
            ax.set_ylabel("Candidate Subgraph Edges $E_{cand}$", fontsize=10)
            ax.set_yscale("log")
            ax.legend(frameon=True, loc="upper right")
            plt.tight_layout()
            fig_path = plots_dir / f"overlap_{model_id.lower()}_{contrast_name.lower()}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"Generated overlap plot: {fig_path.name}")

    summary_df = pd.DataFrame(matching_summary)
    summary_df.to_csv(out_dir / "balance_diagnostics_summary.csv", index=False)
    print("\n=== Matching and Balance Summary ===")
    print(summary_df.to_string())

    print("\n=== Audit Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
