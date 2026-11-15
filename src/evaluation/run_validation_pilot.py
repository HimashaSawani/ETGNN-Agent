import json
import os
import sys
import time
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import degree, k_hop_subgraph

from src.models.etgnn import ETGNNClassifier


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    start_total_time = time.perf_counter()
    torch.set_num_threads(2)

    # 1. Setup paths and directories
    exp_dir = Path("experiments/pilot_validation_audit")
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    exp_dir_explanations = exp_dir / "explanations"
    exp_dir_explanations.mkdir(parents=True, exist_ok=True)

    ckpt_path = "results/checkpoints/etgnn_local.pt"
    val_graph_path = "data/processed/validation_graph.pt"
    mapping_path = "data/processed/node_mapping.csv"
    nodes_mask_path = "data/processed/nodes_with_masks.csv"

    # Verify required artifacts
    for p in [ckpt_path, val_graph_path, mapping_path, nodes_mask_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact missing: {p}")

    env_metadata = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": compute_file_sha256(ckpt_path),
        "checkpoint_size_bytes": os.path.getsize(ckpt_path),
        "val_graph_path": val_graph_path,
        "val_graph_sha256": compute_file_sha256(val_graph_path),
        "val_graph_size_bytes": os.path.getsize(val_graph_path),
        "budget_fraction": 0.25,
        "budget_rounding": "ceil",
        "explainer_seeds": [42, 43, 44],
        "random_subsets_count": 30,
        "random_seed_base": 20261019,
        "time_steps_range": [33, 40],
        "model_architecture": "ETGNNClassifier(input_dim=93, time_dim=16, hidden_dim=64, dropout=0.3)",
    }

    # 2. Load model and validation data
    val_graph = torch.load(val_graph_path, map_location="cpu", weights_only=True)
    model = ETGNNClassifier(input_dim=93)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    # Precompute global out-degrees for source nodes in validation graph
    val_source = val_graph["edge_index"][0]
    val_target = val_graph["edge_index"][1]
    num_val_nodes = val_graph["x"].shape[0]
    val_out_degree = degree(val_source, num_nodes=num_val_nodes).long().numpy()

    # Inference on full validation graph
    with torch.no_grad():
        full_logits = model(val_graph["x"][:, :93], val_graph["edge_index"], val_graph["time_step"])
        full_probs = torch.softmax(full_logits, dim=1)

    # 3. Node Selection: Up to 5 TP, 5 FP, 5 FN, 5 TN
    incoming = degree(val_target, num_nodes=num_val_nodes).long()
    val_mask = val_graph["val_mask"]
    val_pos_all = val_mask.nonzero().flatten().tolist()

    categories = {"TP": [], "FP": [], "FN": [], "TN": []}
    for pos in val_pos_all:
        y_true = int(val_graph["y"][pos])
        if y_true == -1:
            continue  # Exclude unknown labels
        prob1 = float(full_probs[pos, 1])
        y_pred = int(prob1 >= 0.5)
        in_deg = int(incoming[pos])
        cat = ("T" if y_true == y_pred else "F") + ("P" if y_pred == 1 else "N")
        if in_deg >= 2:
            categories[cat].append({
                "val_pos": pos,
                "in_degree": in_deg,
                "prob_illicit": prob1,
                "y_true": y_true,
                "y_pred": y_pred,
            })

    # Sort candidates by in_degree descending
    for cat in categories:
        categories[cat].sort(key=lambda x: x["in_degree"], reverse=True)

    pilot_nodes = []
    for cat in ["TP", "FP", "FN", "TN"]:
        picked_count = 0
        for cand in categories[cat]:
            pos = cand["val_pos"]
            # Extract 2-hop computation subgraph (directed flow: source_to_target)
            subset, edges, mapping, _ = k_hop_subgraph(
                pos, 2, val_graph["edge_index"], relabel_nodes=True, flow="source_to_target"
            )
            local_center = int(mapping.item())
            sub_x = val_graph["x"][subset, :93]
            sub_times = val_graph["time_step"][subset]

            # Verify inference equivalence
            with torch.no_grad():
                sub_logits = model(sub_x, edges, sub_times)
                sub_probs = torch.softmax(sub_logits, dim=1)

            diff = float(torch.abs(full_probs[pos] - sub_probs[local_center]).max())
            if diff > 1e-5:
                print(f"Warning: Inference mismatch for {cat} at pos {pos}: diff={diff:.2e}. Skipping.")
                continue

            global_id = int(val_graph["original_node_index"][pos])
            time_step = int(val_graph["time_step"][pos])
            p_illicit = cand["prob_illicit"]
            y_true = cand["y_true"]
            orig_class = cand["y_pred"]
            orig_prob_orig_class = float(full_probs[pos, orig_class])
            pred_margin = abs(p_illicit - 0.5)
            num_cand_edges = edges.shape[1]
            budget_k = max(1, int(np.ceil(0.25 * num_cand_edges)))
            is_full_set = bool(budget_k == num_cand_edges)

            pilot_nodes.append({
                "case": cat,
                "val_pos": pos,
                "node_index": global_id,
                "time_step": time_step,
                "true_label": y_true,
                "orig_class": orig_class,
                "orig_prob_illicit": p_illicit,
                "orig_prob_orig_class": orig_prob_orig_class,
                "prediction_margin": pred_margin,
                "incoming_degree": cand["in_degree"],
                "subgraph_nodes": len(subset),
                "candidate_edges_count": num_cand_edges,
                "budget_k": budget_k,
                "is_full_set": is_full_set,
                "inference_diff": diff,
            })
            picked_count += 1
            if picked_count >= 5:
                break

    pilot_nodes_df = pd.DataFrame(pilot_nodes)
    pilot_nodes_df.to_csv(exp_dir / "pilot_nodes.csv", index=False)
    print(f"Selected {len(pilot_nodes_df)} pilot nodes across 4 categories.")

    # 4. GNNExplainer Setup
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    raw_interventions = []
    candidate_edges_records = []
    explanations_dict = {}

    # 5. Process each node
    for node_idx, node_row in pilot_nodes_df.iterrows():
        pos = int(node_row["val_pos"])
        global_id = int(node_row["node_index"])
        cat = node_row["case"]
        orig_class = int(node_row["orig_class"])
        orig_prob = float(node_row["orig_prob_orig_class"])
        y_true = int(node_row["true_label"])
        budget_k = int(node_row["budget_k"])
        num_cand_edges = int(node_row["candidate_edges_count"])
        is_full_set = bool(node_row["is_full_set"])

        subset, edges, mapping, _ = k_hop_subgraph(
            pos, 2, val_graph["edge_index"], relabel_nodes=True, flow="source_to_target"
        )
        local_center = int(mapping.item())
        sub_x = val_graph["x"][subset, :93]
        sub_times = val_graph["time_step"][subset]

        # Record candidate edges with canonical order
        source_global_nodes = val_graph["original_node_index"][subset[edges[0]]].numpy()
        target_global_nodes = val_graph["original_node_index"][subset[edges[1]]].numpy()
        edge_source_subset_indices = subset[edges[0]].numpy()
        edge_source_out_degrees = val_out_degree[edge_source_subset_indices]

        for e_idx in range(num_cand_edges):
            candidate_edges_records.append({
                "node_index": global_id,
                "canonical_edge_idx": e_idx,
                "source_local": int(edges[0, e_idx]),
                "target_local": int(edges[1, e_idx]),
                "source_global": int(source_global_nodes[e_idx]),
                "target_global": int(target_global_nodes[e_idx]),
                "source_out_degree": int(edge_source_out_degrees[e_idx]),
            })

        # Precompute Single-Edge Occlusion Drops for candidate edges
        t0_occ_gen = time.perf_counter()
        occ_drops = np.zeros(num_cand_edges, dtype=float)
        for e_idx in range(num_cand_edges):
            keep_occ = torch.ones(num_cand_edges, dtype=torch.bool)
            keep_occ[e_idx] = False
            with torch.no_grad():
                l_occ = model(sub_x, edges[:, keep_occ], sub_times)[local_center]
                p_occ = float(torch.softmax(l_occ, dim=0)[orig_class])
            occ_drops[e_idx] = orig_prob - p_occ
        t1_occ_gen = time.perf_counter()
        occ_gen_time = t1_occ_gen - t0_occ_gen

        # Rank candidate edges for Occlusion: descending by individual drop, stable tie-break
        occ_ranking = np.argsort(-occ_drops, kind="stable")
        occ_selected = occ_ranking[:budget_k]

        # Rank candidate edges for Degree: descending by source out-degree, stable tie-break
        t0_deg_gen = time.perf_counter()
        deg_ranking = np.argsort(-edge_source_out_degrees, kind="stable")
        deg_selected = deg_ranking[:budget_k]
        t1_deg_gen = time.perf_counter()
        deg_gen_time = t1_deg_gen - t0_deg_gen

        # GNNExplainer masks across 3 seeds
        gnnexplainer_masks = {}
        for exp_seed in [42, 43, 44]:
            torch.manual_seed(exp_seed)
            t0_exp = time.perf_counter()
            exp = explainer(sub_x, edges, index=local_center, time_step=sub_times)
            t1_exp = time.perf_counter()
            exp_gen_time = t1_exp - t0_exp
            edge_mask = exp.edge_mask.detach().cpu().numpy()
            gnnexplainer_masks[exp_seed] = edge_mask

            # Rank by mask value descending, stable tie-break
            exp_ranking = np.argsort(-edge_mask, kind="stable")
            exp_selected = exp_ranking[:budget_k]

            # Evaluate Deletion and Retention
            t0_eval = time.perf_counter()
            # Deletion
            keep_del = torch.ones(num_cand_edges, dtype=torch.bool)
            keep_del[exp_selected] = False
            with torch.no_grad():
                l_del = model(sub_x, edges[:, keep_del], sub_times)[local_center]
                p_del = float(torch.softmax(l_del, dim=0)[orig_class])
                pred_del_class = int(l_del.argmax().item())

            # Retention
            keep_ret = torch.zeros(num_cand_edges, dtype=torch.bool)
            keep_ret[exp_selected] = True
            with torch.no_grad():
                l_ret = model(sub_x, edges[:, keep_ret], sub_times)[local_center]
                p_ret = float(torch.softmax(l_ret, dim=0)[orig_class])
                pred_ret_class = int(l_ret.argmax().item())
            t1_eval = time.perf_counter()
            eval_time = t1_eval - t0_eval

            signed_drop = orig_prob - p_del
            is_corrected = bool((cat in ["FP", "FN"]) and (pred_del_class == y_true))
            is_induced = bool((cat in ["TP", "TN"]) and (pred_del_class != y_true))

            raw_interventions.append({
                "node_index": global_id,
                "case": cat,
                "true_label": y_true,
                "orig_class": orig_class,
                "orig_prob_orig_class": orig_prob,
                "prediction_margin": node_row["prediction_margin"],
                "candidate_edges_count": num_cand_edges,
                "budget_k": budget_k,
                "is_full_set": is_full_set,
                "method": "gnnexplainer",
                "seed": exp_seed,
                "repetition_id": exp_seed,
                "selected_edges": json.dumps(exp_selected.tolist()),
                "prob_after_deletion": p_del,
                "signed_deletion_response": signed_drop,
                "prob_after_retention": p_ret,
                "pred_class_after_deletion": pred_del_class,
                "pred_class_after_retention": pred_ret_class,
                "is_corrected": is_corrected,
                "is_induced": is_induced,
                "generation_time_sec": exp_gen_time,
                "evaluation_time_sec": eval_time,
                "failure": 0,
            })

        # Save raw GNNExplainer masks
        explanations_dict[global_id] = {
            seed: mask.tolist() for seed, mask in gnnexplainer_masks.items()
        }

        # Evaluate Degree Selection
        t0_deg_eval = time.perf_counter()
        keep_del_deg = torch.ones(num_cand_edges, dtype=torch.bool)
        keep_del_deg[deg_selected] = False
        with torch.no_grad():
            l_del_deg = model(sub_x, edges[:, keep_del_deg], sub_times)[local_center]
            p_del_deg = float(torch.softmax(l_del_deg, dim=0)[orig_class])
            pred_del_deg_class = int(l_del_deg.argmax().item())

        keep_ret_deg = torch.zeros(num_cand_edges, dtype=torch.bool)
        keep_ret_deg[deg_selected] = True
        with torch.no_grad():
            l_ret_deg = model(sub_x, edges[:, keep_ret_deg], sub_times)[local_center]
            p_ret_deg = float(torch.softmax(l_ret_deg, dim=0)[orig_class])
            pred_ret_deg_class = int(l_ret_deg.argmax().item())
        t1_deg_eval = time.perf_counter()
        deg_eval_time = t1_deg_eval - t0_deg_eval

        raw_interventions.append({
            "node_index": global_id,
            "case": cat,
            "true_label": y_true,
            "orig_class": orig_class,
            "orig_prob_orig_class": orig_prob,
            "prediction_margin": node_row["prediction_margin"],
            "candidate_edges_count": num_cand_edges,
            "budget_k": budget_k,
            "is_full_set": is_full_set,
            "method": "source_out_degree",
            "seed": None,
            "repetition_id": 0,
            "selected_edges": json.dumps(deg_selected.tolist()),
            "prob_after_deletion": p_del_deg,
            "signed_deletion_response": orig_prob - p_del_deg,
            "prob_after_retention": p_ret_deg,
            "pred_class_after_deletion": pred_del_deg_class,
            "pred_class_after_retention": pred_ret_deg_class,
            "is_corrected": bool((cat in ["FP", "FN"]) and (pred_del_deg_class == y_true)),
            "is_induced": bool((cat in ["TP", "TN"]) and (pred_del_deg_class != y_true)),
            "generation_time_sec": deg_gen_time,
            "evaluation_time_sec": deg_eval_time,
            "failure": 0,
        })

        # Evaluate Occlusion Selection
        t0_occ_eval = time.perf_counter()
        keep_del_occ = torch.ones(num_cand_edges, dtype=torch.bool)
        keep_del_occ[occ_selected] = False
        with torch.no_grad():
            l_del_occ = model(sub_x, edges[:, keep_del_occ], sub_times)[local_center]
            p_del_occ = float(torch.softmax(l_del_occ, dim=0)[orig_class])
            pred_del_occ_class = int(l_del_occ.argmax().item())

        keep_ret_occ = torch.zeros(num_cand_edges, dtype=torch.bool)
        keep_ret_occ[occ_selected] = True
        with torch.no_grad():
            l_ret_occ = model(sub_x, edges[:, keep_ret_occ], sub_times)[local_center]
            p_ret_occ = float(torch.softmax(l_ret_occ, dim=0)[orig_class])
            pred_ret_occ_class = int(l_ret_occ.argmax().item())
        t1_occ_eval = time.perf_counter()
        occ_eval_time = t1_occ_eval - t0_occ_eval

        raw_interventions.append({
            "node_index": global_id,
            "case": cat,
            "true_label": y_true,
            "orig_class": orig_class,
            "orig_prob_orig_class": orig_prob,
            "prediction_margin": node_row["prediction_margin"],
            "candidate_edges_count": num_cand_edges,
            "budget_k": budget_k,
            "is_full_set": is_full_set,
            "method": "single_edge_occlusion",
            "seed": None,
            "repetition_id": 0,
            "selected_edges": json.dumps(occ_selected.tolist()),
            "prob_after_deletion": p_del_occ,
            "signed_deletion_response": orig_prob - p_del_occ,
            "prob_after_retention": p_ret_occ,
            "pred_class_after_deletion": pred_del_occ_class,
            "pred_class_after_retention": pred_ret_occ_class,
            "is_corrected": bool((cat in ["FP", "FN"]) and (pred_del_occ_class == y_true)),
            "is_induced": bool((cat in ["TP", "TN"]) and (pred_del_occ_class != y_true)),
            "generation_time_sec": occ_gen_time,
            "evaluation_time_sec": occ_eval_time,
            "failure": 0,
        })

        # 30 Uniform-Random Subsets
        for r_idx in range(30):
            rng_seed = 20261019 + r_idx * 1000 + (global_id % 1000)
            rng = np.random.default_rng(rng_seed)
            t0_rnd_gen = time.perf_counter()
            rand_selected = np.sort(rng.choice(num_cand_edges, size=budget_k, replace=False))
            t1_rnd_gen = time.perf_counter()
            rand_gen_time = t1_rnd_gen - t0_rnd_gen

            t0_rnd_eval = time.perf_counter()
            keep_del_rnd = torch.ones(num_cand_edges, dtype=torch.bool)
            keep_del_rnd[rand_selected] = False
            with torch.no_grad():
                l_del_rnd = model(sub_x, edges[:, keep_del_rnd], sub_times)[local_center]
                p_del_rnd = float(torch.softmax(l_del_rnd, dim=0)[orig_class])
                pred_del_rnd_class = int(l_del_rnd.argmax().item())

            keep_ret_rnd = torch.zeros(num_cand_edges, dtype=torch.bool)
            keep_ret_rnd[rand_selected] = True
            with torch.no_grad():
                l_ret_rnd = model(sub_x, edges[:, keep_ret_rnd], sub_times)[local_center]
                p_ret_rnd = float(torch.softmax(l_ret_rnd, dim=0)[orig_class])
                pred_ret_rnd_class = int(l_ret_rnd.argmax().item())
            t1_rnd_eval = time.perf_counter()
            rand_eval_time = t1_rnd_eval - t0_rnd_eval

            raw_interventions.append({
                "node_index": global_id,
                "case": cat,
                "true_label": y_true,
                "orig_class": orig_class,
                "orig_prob_orig_class": orig_prob,
                "prediction_margin": node_row["prediction_margin"],
                "candidate_edges_count": num_cand_edges,
                "budget_k": budget_k,
                "is_full_set": is_full_set,
                "method": "uniform_random",
                "seed": rng_seed,
                "repetition_id": r_idx,
                "selected_edges": json.dumps(rand_selected.tolist()),
                "prob_after_deletion": p_del_rnd,
                "signed_deletion_response": orig_prob - p_del_rnd,
                "prob_after_retention": p_ret_rnd,
                "pred_class_after_deletion": pred_del_rnd_class,
                "pred_class_after_retention": pred_ret_rnd_class,
                "is_corrected": bool((cat in ["FP", "FN"]) and (pred_del_rnd_class == y_true)),
                "is_induced": bool((cat in ["TP", "TN"]) and (pred_del_rnd_class != y_true)),
                "generation_time_sec": rand_gen_time,
                "evaluation_time_sec": rand_eval_time,
                "failure": 0,
            })

    # Save Candidate Edges & Raw Interventions
    pd.DataFrame(candidate_edges_records).to_csv(exp_dir / "candidate_edges.csv", index=False)
    raw_df = pd.DataFrame(raw_interventions)
    raw_df.to_csv(exp_dir / "raw_interventions.csv", index=False)

    with open(exp_dir_explanations / "raw_edge_masks.json", "w") as f:
        json.dump(explanations_dict, f, indent=2)

    # 6. Summarise Repetitions per Node Before Aggregate Comparisons
    node_summary_rows = []
    for global_id, group in raw_df.groupby("node_index"):
        node_meta = group.iloc[0]
        cat = node_meta["case"]
        orig_class = node_meta["orig_class"]
        orig_prob = node_meta["orig_prob_orig_class"]
        margin = node_meta["prediction_margin"]
        cand_count = node_meta["candidate_edges_count"]
        budget_k = node_meta["budget_k"]
        is_full = node_meta["is_full_set"]

        # GNNExplainer across 3 seeds
        ge_sub = group[group["method"] == "gnnexplainer"]
        ge_drop_mean = ge_sub["signed_deletion_response"].mean()
        ge_drop_std = ge_sub["signed_deletion_response"].std(ddof=1) if len(ge_sub) > 1 else 0.0
        ge_ret_mean = ge_sub["prob_after_retention"].mean()
        ge_ret_std = ge_sub["prob_after_retention"].std(ddof=1) if len(ge_sub) > 1 else 0.0
        ge_corr_rate = ge_sub["is_corrected"].mean()
        ge_ind_rate = ge_sub["is_induced"].mean()

        # Compute Jaccard similarity across GNNExplainer seeds
        ge_sets = [set(json.loads(s)) for s in ge_sub["selected_edges"]]
        jaccards = []
        for i in range(len(ge_sets)):
            for j in range(i + 1, len(ge_sets)):
                intersection = len(ge_sets[i] & ge_sets[j])
                union = len(ge_sets[i] | ge_sets[j])
                jaccards.append(intersection / union if union > 0 else 1.0)
        mean_jaccard = float(np.mean(jaccards)) if jaccards else 1.0

        # Random across 30 draws
        rnd_sub = group[group["method"] == "uniform_random"]
        rnd_drop_mean = rnd_sub["signed_deletion_response"].mean()
        rnd_drop_std = rnd_sub["signed_deletion_response"].std(ddof=1)
        rnd_ret_mean = rnd_sub["prob_after_retention"].mean()
        rnd_ret_std = rnd_sub["prob_after_retention"].std(ddof=1)
        rnd_corr_rate = rnd_sub["is_corrected"].mean()
        rnd_ind_rate = rnd_sub["is_induced"].mean()

        # Degree
        deg_sub = group[group["method"] == "source_out_degree"].iloc[0]
        deg_drop = deg_sub["signed_deletion_response"]
        deg_ret = deg_sub["prob_after_retention"]
        deg_corr = deg_sub["is_corrected"]
        deg_ind = deg_sub["is_induced"]

        # Occlusion
        occ_sub = group[group["method"] == "single_edge_occlusion"].iloc[0]
        occ_drop = occ_sub["signed_deletion_response"]
        occ_ret = occ_sub["prob_after_retention"]
        occ_corr = occ_sub["is_corrected"]
        occ_ind = occ_sub["is_induced"]

        # Paired difference: GNNExplainer mean drop minus Random mean drop
        paired_drop_diff_ge_rnd = ge_drop_mean - rnd_drop_mean
        paired_drop_diff_ge_deg = ge_drop_mean - deg_drop
        paired_drop_diff_ge_occ = ge_drop_mean - occ_drop

        node_summary_rows.append({
            "node_index": global_id,
            "case": cat,
            "orig_class": orig_class,
            "orig_prob_orig_class": orig_prob,
            "prediction_margin": margin,
            "candidate_edges_count": cand_count,
            "budget_k": budget_k,
            "is_full_set": is_full,
            "ge_jaccard_stability": mean_jaccard,
            "ge_deletion_drop_mean": ge_drop_mean,
            "ge_deletion_drop_std": ge_drop_std,
            "ge_retention_prob_mean": ge_ret_mean,
            "ge_retention_prob_std": ge_ret_std,
            "ge_corrected_rate": ge_corr_rate,
            "ge_induced_rate": ge_ind_rate,
            "rnd_deletion_drop_mean": rnd_drop_mean,
            "rnd_deletion_drop_std": rnd_drop_std,
            "rnd_retention_prob_mean": rnd_ret_mean,
            "rnd_retention_prob_std": rnd_ret_std,
            "rnd_corrected_rate": rnd_corr_rate,
            "rnd_induced_rate": rnd_ind_rate,
            "deg_deletion_drop": deg_drop,
            "deg_retention_prob": deg_ret,
            "deg_corrected": deg_corr,
            "deg_induced": deg_ind,
            "occ_deletion_drop": occ_drop,
            "occ_retention_prob": occ_ret,
            "occ_corrected": occ_corr,
            "occ_induced": occ_ind,
            "paired_diff_ge_minus_rnd": paired_drop_diff_ge_rnd,
            "paired_diff_ge_minus_deg": paired_drop_diff_ge_deg,
            "paired_diff_ge_minus_occ": paired_drop_diff_ge_occ,
        })

    node_summary_df = pd.DataFrame(node_summary_rows)
    node_summary_df.to_csv(exp_dir / "node_summary.csv", index=False)

    # 7. Aggregate Analysis Across Methods and Quadrants
    agg_rows = []
    for method_name, method_df in raw_df.groupby("method"):
        agg_rows.append({
            "method": method_name,
            "mean_signed_deletion_response": method_df["signed_deletion_response"].mean(),
            "std_signed_deletion_response": method_df["signed_deletion_response"].std(ddof=1),
            "mean_retention_prob": method_df["prob_after_retention"].mean(),
            "std_retention_prob": method_df["prob_after_retention"].std(ddof=1),
            "fp_correction_rate": method_df[method_df["case"] == "FP"]["is_corrected"].mean(),
            "fn_correction_rate": method_df[method_df["case"] == "FN"]["is_corrected"].mean(),
            "tp_error_induction_rate": method_df[method_df["case"] == "TP"]["is_induced"].mean(),
            "tn_error_induction_rate": method_df[method_df["case"] == "TN"]["is_induced"].mean(),
            "mean_generation_time_sec": method_df["generation_time_sec"].mean(),
            "mean_evaluation_time_sec": method_df["evaluation_time_sec"].mean(),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(exp_dir / "aggregate_summary.csv", index=False)

    # Quadrant breakdown
    quad_agg = node_summary_df.groupby("case")[[
        "ge_deletion_drop_mean", "rnd_deletion_drop_mean",
        "deg_deletion_drop", "occ_deletion_drop",
        "paired_diff_ge_minus_rnd", "ge_jaccard_stability"
    ]].mean().reset_index()
    quad_agg.to_csv(exp_dir / "quadrant_summary.csv", index=False)

    # 8. Per-Node Paired Plots
    fig, axes = plt.subplots(4, 5, figsize=(22, 16), sharey=True)
    axes = axes.flatten()

    for idx, row in node_summary_df.iterrows():
        ax = axes[idx]
        node_id = row["node_index"]
        cat = row["case"]
        margin = row["prediction_margin"]
        k = row["budget_k"]
        e_cand = row["candidate_edges_count"]

        # Raw random points for this node
        node_raw_rnd = raw_df[(raw_df["node_index"] == node_id) & (raw_df["method"] == "uniform_random")]
        rnd_drops = node_raw_rnd["signed_deletion_response"].to_numpy()

        # Raw GNNExplainer points for this node
        node_raw_ge = raw_df[(raw_df["node_index"] == node_id) & (raw_df["method"] == "gnnexplainer")]
        ge_drops = node_raw_ge["signed_deletion_response"].to_numpy()

        deg_drop = row["deg_deletion_drop"]
        occ_drop = row["occ_deletion_drop"]

        # Plot Random distribution boxplot
        bp = ax.boxplot(rnd_drops, positions=[1], widths=0.4, patch_artist=True,
                        boxprops=dict(facecolor="#d9d9d9", color="gray"),
                        medianprops=dict(color="black", linewidth=1.5))
        
        # Overlay jittered random points
        x_jitter = np.random.default_rng(42).normal(1, 0.04, size=len(rnd_drops))
        ax.scatter(x_jitter, rnd_drops, color="gray", alpha=0.5, s=15, zorder=3)

        # Plot GNNExplainer 3 seed points
        ax.scatter([2, 2, 2], ge_drops, color="#1f77b4", s=45, zorder=4, label="GNNExplainer (3 seeds)")
        ax.plot([1.8, 2.2], [row["ge_deletion_drop_mean"], row["ge_deletion_drop_mean"]], color="#1f77b4", linewidth=2)

        # Plot Degree and Occlusion points
        ax.scatter([3], [deg_drop], color="#ff7f0e", marker="s", s=50, zorder=4, label="Out-Degree")
        ax.scatter([4], [occ_drop], color="#2ca02c", marker="^", s=55, zorder=4, label="Occlusion")

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"Node {node_id} ({cat})\nMargin={margin:.3f} | k={k}/{e_cand}", fontsize=9, fontweight="bold")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Rnd(30)", "GE(3)", "Deg", "Occ"], fontsize=8)
        if idx % 5 == 0:
            ax.set_ylabel("Signed Deletion Drop\n($P_0 - P_{del}$)", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99), ncol=4, fontsize=10)
    fig.suptitle("Validation Feasibility Pilot: Per-Node Paired Deletion Response Comparison (25% Budget)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(plots_dir / "pilot_per_node_paired_deletion_responses.png", dpi=200, bbox_inches="tight")
    fig.savefig(plots_dir / "pilot_per_node_paired_deletion_responses.pdf", bbox_inches="tight")
    plt.close()

    # Paired Difference Bar Plot
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(node_summary_df))
    bars = ax2.bar(x_pos, node_summary_df["paired_diff_ge_minus_rnd"], color=[
        "#2ca02c" if v > 0 else "#d62728" for v in node_summary_df["paired_diff_ge_minus_rnd"]
    ], alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{row['case']}\n{row['node_index']}" for _, row in node_summary_df.iterrows()], fontsize=8)
    ax2.set_ylabel("Paired Drop Advantage: $\Delta_{\mathrm{GE}} - \Delta_{\mathrm{Rnd}}$", fontsize=11, fontweight="bold")
    ax2.set_title("GNNExplainer vs. Uniform-Random 30-Draw Paired Advantage per Node (Positive = GE Drops Confidence More)",
                  fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig2.savefig(plots_dir / "pilot_paired_advantage_bars.png", dpi=200, bbox_inches="tight")
    fig2.savefig(plots_dir / "pilot_paired_advantage_bars.pdf", bbox_inches="tight")
    plt.close()

    total_time_elapsed = time.perf_counter() - start_total_time
    env_metadata["total_runtime_seconds"] = total_time_elapsed

    with open(exp_dir / "config.json", "w") as f:
        json.dump(env_metadata, f, indent=2)

    print("Pilot completed successfully in", round(total_time_elapsed, 2), "seconds.")
    print("Results written to:", exp_dir)


if __name__ == "__main__":
    main()
