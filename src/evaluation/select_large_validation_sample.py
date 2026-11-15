"""
Pre-registration and Sampling Script for Larger Validation Evaluation (N=80 Nodes)
==================================================================================
Selects a balanced, reproducible stratified random sample of 80 validation nodes
(20 TP, 20 FP, 20 FN, 20 TN) from validation time steps 33-40 with E_cand >= 2.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import degree, k_hop_subgraph

import sys
sys.path.insert(0, str(Path(".").resolve()))

from src.models.etgnn import ETGNNClassifier


def select_sample(output_dir: str = "experiments/large_validation_audit", n_per_quadrant: int = 20, seed: int = 20261020):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    val_graph_path = "data/processed/validation_graph.pt"
    ckpt_path = "results/checkpoints/etgnn_local.pt"
    
    val_graph = torch.load(val_graph_path, map_location="cpu", weights_only=True)
    edge_index = val_graph["edge_index"]
    y = val_graph["y"]
    val_mask = val_graph["val_mask"]
    time_step = val_graph["time_step"]
    
    model = ETGNNClassifier(input_dim=93)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        logits = model(val_graph["x"][:, :93], edge_index, time_step)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= 0.5).long()
        
    val_indices = val_mask.nonzero().flatten().tolist()
    records = []
    for i in val_indices:
        label = int(y[i])
        if label == -1:
            continue
        pred = int(preds[i])
        cat = ("T" if label == pred else "F") + ("P" if pred == 1 else "N")
        p = float(probs[i])
        t = int(time_step[i])
        global_id = int(val_graph["original_node_index"][i])
        records.append({
            "val_pos": i,
            "node_index": global_id,
            "time_step": t,
            "true_label": label,
            "orig_class": pred,
            "case": cat,
            "orig_prob_illicit": p,
            "orig_prob_orig_class": p if pred == 1 else (1.0 - p),
            "prediction_margin": abs(p - 0.5),
        })
        
    df = pd.DataFrame(records)
    
    # Filter for E_cand >= 2 (2-hop subgraph edges >= 2)
    eligible = []
    candidate_edge_records = []
    
    num_val_nodes = val_graph["x"].shape[0]
    val_source = val_graph["edge_index"][0]
    val_out_degree = degree(val_source, num_nodes=num_val_nodes).long().numpy()
    
    for _, r in df.iterrows():
        pos = r["val_pos"]
        sub, edges, mapping, _ = k_hop_subgraph(pos, 2, edge_index, relabel_nodes=True, flow="source_to_target")
        cand_count = edges.shape[1]
        if cand_count >= 2:
            r_dict = dict(r)
            r_dict["incoming_degree"] = int(degree(edge_index[1], num_nodes=num_val_nodes)[pos])
            r_dict["candidate_edges_count"] = cand_count
            r_dict["subgraph_nodes"] = len(sub)
            r_dict["budget_k"] = max(1, int(np.ceil(0.25 * cand_count)))
            r_dict["is_full_set"] = bool(r_dict["budget_k"] == cand_count)
            eligible.append(r_dict)
            
    el_df = pd.DataFrame(eligible)
    print(f"Eligible validation nodes (E_cand >= 2): {len(el_df)}")
    print(el_df["case"].value_counts())
    
    # Reproducible stratified sampling
    rng = np.random.default_rng(seed)
    sampled = []
    for cat in ["TP", "FP", "FN", "TN"]:
        cat_df = el_df[el_df["case"] == cat]
        chosen = rng.choice(len(cat_df), size=n_per_quadrant, replace=False)
        sampled.append(cat_df.iloc[chosen])
        
    sample_df = pd.concat(sampled).reset_index(drop=True)
    
    # Sort deterministically by case then node_index
    sample_df = sample_df.sort_values(by=["case", "node_index"]).reset_index(drop=True)
    sample_csv = out_path / f"sampled_nodes_{len(sample_df)}.csv"
    sample_df.to_csv(sample_csv, index=False)
    
    # Extract candidate edges for the sampled nodes
    all_sample_edges = []
    for idx, row in sample_df.iterrows():
        pos = int(row["val_pos"])
        gid = int(row["node_index"])
        sub, edges, mapping, _ = k_hop_subgraph(pos, 2, edge_index, relabel_nodes=True, flow="source_to_target")
        source_global_nodes = val_graph["original_node_index"][sub[edges[0]]].numpy()
        target_global_nodes = val_graph["original_node_index"][sub[edges[1]]].numpy()
        edge_source_subset_indices = sub[edges[0]].numpy()
        edge_source_out_degrees = val_out_degree[edge_source_subset_indices]
        
        for e_idx in range(edges.shape[1]):
            all_sample_edges.append({
                "node_index": gid,
                "canonical_edge_idx": e_idx,
                "source_local": int(edges[0, e_idx]),
                "target_local": int(edges[1, e_idx]),
                "source_global": int(source_global_nodes[e_idx]),
                "target_global": int(target_global_nodes[e_idx]),
                "source_out_degree": int(edge_source_out_degrees[e_idx]),
            })
            
    sample_edges_df = pd.DataFrame(all_sample_edges)
    sample_edges_csv = out_path / f"candidate_edges_{len(sample_df)}.csv"
    sample_edges_df.to_csv(sample_edges_csv, index=False)
    
    print(f"\n[OK] Pre-registered sample saved to: {sample_csv}")
    print(f"[OK] Total sampled nodes: {len(sample_df)} ({n_per_quadrant} per quadrant)")
    print(f"[OK] Total candidate edges: {len(sample_edges_df)}")
    print("\nSummary by Quadrant:")
    print(sample_df.groupby("case")[["candidate_edges_count", "incoming_degree", "prediction_margin"]].agg(["mean", "min", "max"]))
    
    return sample_df, sample_edges_df

if __name__ == "__main__":
    select_sample()
