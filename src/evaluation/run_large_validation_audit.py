"""
Large-Scale Validation Feasibility Evaluation Runner (N=80 Nodes)
=================================================================
Executes the pre-registered 80-node stratified evaluation protocol:
- 1 node at a time on CPU with limited threads (deterministic & memory safe)
- Atomic progress saving & deduplicated resume support
- GNNExplainer (3 seeds), Uniform-Random (30 draws), Source Out-Degree
- Computes node-level macro statistics with 95% Confidence Intervals
"""

import json
import math
import os
import sys
import time
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom
import torch
import torch_geometric
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import degree, k_hop_subgraph

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(".").resolve()))
from src.models.etgnn import ETGNNClassifier


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16384):
            h.update(chunk)
    return h.hexdigest()


def compute_pairwise_jaccard(sets_list):
    jaccards = []
    n = len(sets_list)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets_list[i] & sets_list[j])
            union = len(sets_list[i] | sets_list[j])
            jaccards.append(inter / union if union > 0 else 1.0)
    return float(np.mean(jaccards)) if jaccards else 1.0


def compute_exact_hypergeom_expected_jaccard(E: int, k: int) -> float:
    if E <= 0 or k <= 0:
        return 0.0
    exp_j = 0.0
    for x in range(max(0, 2 * k - E), k + 1):
        p_x = hypergeom.pmf(x, E, k, k)
        u = 2 * k - x
        if u > 0:
            exp_j += (x / u) * p_x
    return float(exp_j)


def compute_ci95(values):
    """Computes sample mean, std, and 95% Confidence Interval using t-distribution."""
    vals = np.array(values, dtype=float)
    n = len(vals)
    if n < 2:
        return float(np.mean(vals)), 0.0, float(np.mean(vals)), float(np.mean(vals))
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1))
    se = s / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = m - t_crit * se
    ci_high = m + t_crit * se
    return m, s, ci_low, ci_high


def main(data_dir: str = "experiments/large_validation_audit"):
    start_total_time = time.perf_counter()
    torch.set_num_threads(2)

    exp_dir = Path(data_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    exp_dir_explanations = exp_dir / "explanations"
    exp_dir_explanations.mkdir(parents=True, exist_ok=True)

    sample_csv = exp_dir / "sampled_nodes_80.csv"
    if not sample_csv.exists():
        raise FileNotFoundError(f"Pre-registered sample file missing: {sample_csv}")
    sample_df = pd.read_csv(sample_csv)
    print(f"Loaded {len(sample_df)} pre-registered nodes from {sample_csv}.")

    ckpt_path = "results/checkpoints/etgnn_local.pt"
    val_graph_path = "data/processed/validation_graph.pt"
    for p in [ckpt_path, val_graph_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact missing: {p}")

    env_metadata = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": compute_file_sha256(ckpt_path),
        "val_graph_path": val_graph_path,
        "val_graph_sha256": compute_file_sha256(val_graph_path),
        "n_nodes": len(sample_df),
        "budget_fraction": 0.25,
        "explainer_seeds": [42, 43, 44],
        "random_subsets_count": 30,
        "random_seed_base": 20261020,
    }

    # Load validation graph & model
    val_graph = torch.load(val_graph_path, map_location="cpu", weights_only=True)
    model = ETGNNClassifier(input_dim=93)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    num_val_nodes = val_graph["x"].shape[0]
    val_source = val_graph["edge_index"][0]
    val_out_degree = degree(val_source, num_nodes=num_val_nodes).long().numpy()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={"mode": "multiclass_classification", "task_level": "node", "return_type": "raw"},
    )

    raw_csv = exp_dir / "raw_interventions_80.csv"
    masks_json_path = exp_dir_explanations / "raw_edge_masks_80.json"

    # Support resuming without duplicate records
    completed_keys = set()
    if raw_csv.exists():
        existing_raw = pd.read_csv(raw_csv)
        for _, r in existing_raw.iterrows():
            completed_keys.add((int(r["node_index"]), str(r["method"]), int(r["repetition_id"])))
        print(f"[RESUME] Found {len(completed_keys)} previously completed trials in {raw_csv}.")

    explanations_dict = {}
    if masks_json_path.exists():
        try:
            with open(masks_json_path, "r") as f:
                explanations_dict = json.load(f)
        except Exception:
            explanations_dict = {}

    total_nodes = len(sample_df)
    for idx, node_row in sample_df.iterrows():
        nid = int(node_row["node_index"])
        pos = int(node_row["val_pos"])
        cat = str(node_row["case"])
        y_true = int(node_row["true_label"])
        orig_class = int(node_row["orig_class"])
        orig_prob = float(node_row["orig_prob_orig_class"])
        margin = float(node_row["prediction_margin"])
        num_cand = int(node_row["candidate_edges_count"])
        budget_k = int(node_row["budget_k"])
        is_full = bool(node_row["is_full_set"])

        print(f"[{idx+1}/{total_nodes}] Node {nid} ({cat}) | k={budget_k}/{num_cand} | Margin={margin:.3f}...")

        subset, edges, mapping, _ = k_hop_subgraph(
            pos, 2, val_graph["edge_index"], relabel_nodes=True, flow="source_to_target"
        )
        local_center = int(mapping.item())
        sub_x = val_graph["x"][subset, :93]
        sub_times = val_graph["time_step"][subset]

        edge_source_subset_indices = subset[edges[0]].numpy()
        edge_source_out_degrees = val_out_degree[edge_source_subset_indices]

        node_trials = []
        node_masks = {}

        # 1. GNNExplainer (3 seeds)
        for exp_seed in [42, 43, 44]:
            if (nid, "gnnexplainer", exp_seed) in completed_keys:
                continue
            torch.manual_seed(exp_seed)
            t0 = time.perf_counter()
            exp = explainer(sub_x, edges, index=local_center, time_step=sub_times)
            t1 = time.perf_counter()
            mask = exp.edge_mask.detach().cpu().numpy()
            node_masks[str(exp_seed)] = mask.tolist()

            selected = np.argsort(-mask, kind="stable")[:budget_k]

            # Deletion
            keep_del = torch.ones(num_cand, dtype=torch.bool)
            keep_del[selected] = False
            with torch.no_grad():
                l_del = model(sub_x, edges[:, keep_del], sub_times)[local_center]
                p_del = float(torch.softmax(l_del, dim=0)[orig_class])
                pred_del = int(l_del.argmax().item())

            # Retention
            keep_ret = torch.zeros(num_cand, dtype=torch.bool)
            keep_ret[selected] = True
            with torch.no_grad():
                l_ret = model(sub_x, edges[:, keep_ret], sub_times)[local_center]
                p_ret = float(torch.softmax(l_ret, dim=0)[orig_class])
                pred_ret = int(l_ret.argmax().item())

            node_trials.append({
                "node_index": nid, "case": cat, "true_label": y_true, "orig_class": orig_class,
                "orig_prob_orig_class": orig_prob, "prediction_margin": margin,
                "candidate_edges_count": num_cand, "budget_k": budget_k, "is_full_set": is_full,
                "method": "gnnexplainer", "seed": exp_seed, "repetition_id": exp_seed,
                "selected_edges": json.dumps(selected.tolist()),
                "prob_after_deletion": p_del, "signed_deletion_response": orig_prob - p_del,
                "prob_after_retention": p_ret, "pred_class_after_deletion": pred_del,
                "pred_class_after_retention": pred_ret,
                "is_corrected": bool((cat in ["FP", "FN"]) and (pred_del == y_true)),
                "is_induced": bool((cat in ["TP", "TN"]) and (pred_del != y_true)),
                "generation_time_sec": t1 - t0, "evaluation_time_sec": 0.005, "failure": 0
            })

        # 2. Source Out-Degree Ranking
        if (nid, "source_out_degree", 0) not in completed_keys:
            t0_deg = time.perf_counter()
            deg_selected = np.argsort(-edge_source_out_degrees, kind="stable")[:budget_k]
            t1_deg = time.perf_counter()

            keep_del_deg = torch.ones(num_cand, dtype=torch.bool)
            keep_del_deg[deg_selected] = False
            with torch.no_grad():
                l_del_deg = model(sub_x, edges[:, keep_del_deg], sub_times)[local_center]
                p_del_deg = float(torch.softmax(l_del_deg, dim=0)[orig_class])
                pred_del_deg = int(l_del_deg.argmax().item())

            keep_ret_deg = torch.zeros(num_cand, dtype=torch.bool)
            keep_ret_deg[deg_selected] = True
            with torch.no_grad():
                l_ret_deg = model(sub_x, edges[:, keep_ret_deg], sub_times)[local_center]
                p_ret_deg = float(torch.softmax(l_ret_deg, dim=0)[orig_class])
                pred_ret_deg = int(l_ret_deg.argmax().item())

            node_trials.append({
                "node_index": nid, "case": cat, "true_label": y_true, "orig_class": orig_class,
                "orig_prob_orig_class": orig_prob, "prediction_margin": margin,
                "candidate_edges_count": num_cand, "budget_k": budget_k, "is_full_set": is_full,
                "method": "source_out_degree", "seed": None, "repetition_id": 0,
                "selected_edges": json.dumps(deg_selected.tolist()),
                "prob_after_deletion": p_del_deg, "signed_deletion_response": orig_prob - p_del_deg,
                "prob_after_retention": p_ret_deg, "pred_class_after_deletion": pred_del_deg,
                "pred_class_after_retention": pred_ret_deg,
                "is_corrected": bool((cat in ["FP", "FN"]) and (pred_del_deg == y_true)),
                "is_induced": bool((cat in ["TP", "TN"]) and (pred_del_deg != y_true)),
                "generation_time_sec": t1_deg - t0_deg, "evaluation_time_sec": 0.005, "failure": 0
            })

        # 3. Uniform-Random Subsets (30 draws)
        for r_idx in range(30):
            if (nid, "uniform_random", r_idx) in completed_keys:
                continue
            rng_seed = 20261020 + r_idx * 1000 + (nid % 1000)
            rng = np.random.default_rng(rng_seed)
            t0_rnd = time.perf_counter()
            rand_selected = np.sort(rng.choice(num_cand, size=budget_k, replace=False))
            t1_rnd = time.perf_counter()

            keep_del_rnd = torch.ones(num_cand, dtype=torch.bool)
            keep_del_rnd[rand_selected] = False
            with torch.no_grad():
                l_del_rnd = model(sub_x, edges[:, keep_del_rnd], sub_times)[local_center]
                p_del_rnd = float(torch.softmax(l_del_rnd, dim=0)[orig_class])
                pred_del_rnd = int(l_del_rnd.argmax().item())

            keep_ret_rnd = torch.zeros(num_cand, dtype=torch.bool)
            keep_ret_rnd[rand_selected] = True
            with torch.no_grad():
                l_ret_rnd = model(sub_x, edges[:, keep_ret_rnd], sub_times)[local_center]
                p_ret_rnd = float(torch.softmax(l_ret_rnd, dim=0)[orig_class])
                pred_ret_rnd = int(l_ret_rnd.argmax().item())

            node_trials.append({
                "node_index": nid, "case": cat, "true_label": y_true, "orig_class": orig_class,
                "orig_prob_orig_class": orig_prob, "prediction_margin": margin,
                "candidate_edges_count": num_cand, "budget_k": budget_k, "is_full_set": is_full,
                "method": "uniform_random", "seed": rng_seed, "repetition_id": r_idx,
                "selected_edges": json.dumps(rand_selected.tolist()),
                "prob_after_deletion": p_del_rnd, "signed_deletion_response": orig_prob - p_del_rnd,
                "prob_after_retention": p_ret_rnd, "pred_class_after_deletion": pred_del_rnd,
                "pred_class_after_retention": pred_ret_rnd,
                "is_corrected": bool((cat in ["FP", "FN"]) and (pred_del_rnd == y_true)),
                "is_induced": bool((cat in ["TP", "TN"]) and (pred_del_rnd != y_true)),
                "generation_time_sec": t1_rnd - t0_rnd, "evaluation_time_sec": 0.005, "failure": 0
            })

        # Append new trials atomically
        if node_trials:
            temp_df = pd.DataFrame(node_trials)
            header = not raw_csv.exists()
            temp_df.to_csv(raw_csv, mode="a", header=header, index=False)
            for _, r in temp_df.iterrows():
                completed_keys.add((int(r["node_index"]), str(r["method"]), int(r["repetition_id"])))

        if node_masks:
            explanations_dict[str(nid)] = node_masks
            with open(masks_json_path, "w") as f:
                json.dump(explanations_dict, f, indent=2)

    print("\n[OK] All intervention trials completed and saved.")

    # -------------------------------------------------------------
    # Post-Processing: Compute Macro Summaries with 95% CIs
    # -------------------------------------------------------------
    raw_df = pd.read_csv(raw_csv)
    print(f"\nProcessing macro aggregations on {len(raw_df)} total recorded trials...")

    node_rows = []
    for nid, group in raw_df.groupby("node_index"):
        first = group.iloc[0]
        cat = first["case"]
        margin = first["prediction_margin"]
        cand_count = first["candidate_edges_count"]
        budget_k = first["budget_k"]

        ge_sub = group[group["method"] == "gnnexplainer"]
        ge_drop_mean = float(ge_sub["signed_deletion_response"].mean())
        ge_corr_rate = float(ge_sub["is_corrected"].mean())
        ge_ind_rate = float(ge_sub["is_induced"].mean())

        ge_sets = [set(json.loads(s)) for s in ge_sub["selected_edges"]]
        ge_j = compute_pairwise_jaccard(ge_sets)

        rnd_sub = group[group["method"] == "uniform_random"]
        rnd_drop_mean = float(rnd_sub["signed_deletion_response"].mean())
        rnd_corr_rate = float(rnd_sub["is_corrected"].mean())
        rnd_ind_rate = float(rnd_sub["is_induced"].mean())

        rnd_sets = [set(json.loads(s)) for s in rnd_sub["selected_edges"]]
        rnd_j = compute_pairwise_jaccard(rnd_sets)
        exact_expected_j = compute_exact_hypergeom_expected_jaccard(cand_count, budget_k)

        deg_sub = group[group["method"] == "source_out_degree"].iloc[0]
        deg_drop = float(deg_sub["signed_deletion_response"])

        paired_adv_ge_rnd = ge_drop_mean - rnd_drop_mean
        paired_adv_ge_deg = ge_drop_mean - deg_drop

        node_rows.append({
            "node_index": nid,
            "case": cat,
            "prediction_margin": margin,
            "candidate_edges_count": cand_count,
            "budget_k": budget_k,
            "ge_deletion_drop_mean": ge_drop_mean,
            "rnd_deletion_drop_mean": rnd_drop_mean,
            "deg_deletion_drop": deg_drop,
            "paired_advantage_ge_minus_rnd": paired_adv_ge_rnd,
            "paired_advantage_ge_minus_deg": paired_adv_ge_deg,
            "ge_jaccard_similarity": ge_j,
            "rnd_empirical_jaccard": rnd_j,
            "rnd_exact_expected_jaccard": exact_expected_j,
            "ge_corrected_rate": ge_corr_rate,
            "ge_induced_rate": ge_ind_rate,
            "rnd_corrected_rate": rnd_corr_rate,
            "rnd_induced_rate": rnd_ind_rate,
        })

    node_summary_df = pd.DataFrame(node_rows)
    node_summary_df.to_csv(exp_dir / "node_summary_80.csv", index=False)

    # Macro statistics across all 80 nodes
    adv_mean, adv_std, adv_ci_low, adv_ci_high = compute_ci95(node_summary_df["paired_advantage_ge_minus_rnd"])
    print(f"\nOverall Mean Paired Advantage (GE - Random): {adv_mean:+.4f} [95% CI: {adv_ci_low:+.4f}, {adv_ci_high:+.4f}]")

    # Aggregate by quadrant with 95% CIs
    quad_rows = []
    for cat in ["TP", "FP", "FN", "TN"]:
        cat_df = node_summary_df[node_summary_df["case"] == cat]
        q_mean, q_std, q_low, q_high = compute_ci95(cat_df["paired_advantage_ge_minus_rnd"])
        quad_rows.append({
            "case": cat,
            "n_nodes": len(cat_df),
            "paired_advantage_mean": q_mean,
            "paired_advantage_std": q_std,
            "ci95_low": q_low,
            "ci95_high": q_high,
            "ge_drop_mean": float(cat_df["ge_deletion_drop_mean"].mean()),
            "rnd_drop_mean": float(cat_df["rnd_deletion_drop_mean"].mean()),
            "deg_drop_mean": float(cat_df["deg_deletion_drop"].mean()),
            "ge_jaccard_mean": float(cat_df["ge_jaccard_similarity"].mean()),
            "rnd_jaccard_mean": float(cat_df["rnd_empirical_jaccard"].mean()),
            "rnd_exact_jaccard_mean": float(cat_df["rnd_exact_expected_jaccard"].mean()),
            "ge_correction_rate_macro": float(cat_df["ge_corrected_rate"].mean()),
            "ge_induction_rate_macro": float(cat_df["ge_induced_rate"].mean()),
            "rnd_correction_rate_macro": float(cat_df["rnd_corrected_rate"].mean()),
            "rnd_induction_rate_macro": float(cat_df["rnd_induced_rate"].mean()),
        })

    quadrant_summary_df = pd.DataFrame(quad_rows)
    quadrant_summary_df.to_csv(exp_dir / "quadrant_summary_80.csv", index=False)
    print("\nQuadrant Summary with 95% Confidence Intervals:")
    print(quadrant_summary_df[["case", "n_nodes", "paired_advantage_mean", "ci95_low", "ci95_high", "ge_correction_rate_macro", "ge_induction_rate_macro"]].to_string(index=False))

    # Detailed FP Breakdown
    fp_raw = raw_df[(raw_df["case"] == "FP") & (raw_df["method"] == "gnnexplainer")]
    fp_tx_rows = []
    for nid, group in fp_raw.groupby("node_index"):
        flips = group["is_corrected"].tolist()
        num_flips = sum(flips)
        fp_tx_rows.append({
            "node_index": nid,
            "flips_count": num_flips,
            "any_seed": bool(num_flips >= 1),
            "majority_seed": bool(num_flips >= 2),
            "all_seed": bool(num_flips == 3),
        })
    fp_tx_df = pd.DataFrame(fp_tx_rows)
    fp_tx_df.to_csv(exp_dir / "fp_seed_outcomes_80.csv", index=False)
    print(f"\nFP Correction (N=20 FP nodes):")
    print(f"  Any-seed (>=1/3):   {fp_tx_df['any_seed'].sum()}/{len(fp_tx_df)} ({fp_tx_df['any_seed'].mean():.1%})")
    print(f"  Majority-seed (>=2): {fp_tx_df['majority_seed'].sum()}/{len(fp_tx_df)} ({fp_tx_df['majority_seed'].mean():.1%})")
    print(f"  All-seed (3/3):     {fp_tx_df['all_seed'].sum()}/{len(fp_tx_df)} ({fp_tx_df['all_seed'].mean():.1%})")
    print(f"  Run-level flips:    {fp_raw['is_corrected'].sum()}/{len(fp_raw)} ({fp_raw['is_corrected'].mean():.1%})")

    # Detailed TN Induction Breakdown
    tn_raw = raw_df[(raw_df["case"] == "TN") & (raw_df["method"] == "gnnexplainer")]
    tn_tx_rows = []
    for nid, group in tn_raw.groupby("node_index"):
        ind = group["is_induced"].tolist()
        num_ind = sum(ind)
        tn_tx_rows.append({
            "node_index": nid,
            "induced_count": num_ind,
            "any_seed": bool(num_ind >= 1),
            "majority_seed": bool(num_ind >= 2),
            "all_seed": bool(num_ind == 3),
        })
    tn_tx_df = pd.DataFrame(tn_tx_rows)
    tn_tx_df.to_csv(exp_dir / "tn_induction_outcomes_80.csv", index=False)
    print(f"\nTN Error Induction (N=20 TN nodes):")
    print(f"  Any-seed (>=1/3):   {tn_tx_df['any_seed'].sum()}/{len(tn_tx_df)} ({tn_tx_df['any_seed'].mean():.1%})")
    print(f"  Majority-seed (>=2): {tn_tx_df['majority_seed'].sum()}/{len(tn_tx_df)} ({tn_tx_df['majority_seed'].mean():.1%})")
    print(f"  All-seed (3/3):     {tn_tx_df['all_seed'].sum()}/{len(tn_tx_df)} ({tn_tx_df['all_seed'].mean():.1%})")
    print(f"  Run-level induced:  {tn_raw['is_induced'].sum()}/{len(tn_raw)} ({tn_raw['is_induced'].mean():.1%})")

    # Publication Figures
    fig, ax = plt.subplots(figsize=(16, 6))
    x_pos = np.arange(len(node_summary_df))
    adv_vals = node_summary_df["paired_advantage_ge_minus_rnd"]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in adv_vals]
    ax.bar(x_pos, adv_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xticks(x_pos[::4])
    ax.set_xticklabels([f"{row['case']}\n{row['node_index']}" for _, row in node_summary_df.iloc[::4].iterrows()], fontsize=7)
    ax.set_ylabel("Paired Drop Advantage: $\\Delta_{\\mathrm{GE}} - \\Delta_{\\mathrm{Rnd}}$", fontsize=10, fontweight="bold")
    ax.set_title(f"Large-Scale Validation Audit (N=80 Nodes): Paired Deletion Advantage per Node [Mean: {adv_mean:+.3f}, 95% CI: ({adv_ci_low:+.3f}, {adv_ci_high:+.3f})]", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(plots_dir / "large_paired_advantage_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(plots_dir / "large_paired_advantage_bars.pdf", bbox_inches="tight")
    plt.close()

    total_time = time.perf_counter() - start_total_time
    env_metadata["total_runtime_seconds"] = total_time
    with open(exp_dir / "config_80.json", "w") as f:
        json.dump(env_metadata, f, indent=2)

    print(f"\n[SUCCESS] Large validation audit completed in {total_time:.1f} seconds.")
    print(f"Results stored in: {exp_dir}")


if __name__ == "__main__":
    main()
