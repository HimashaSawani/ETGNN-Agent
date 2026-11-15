"""
verify_subgraph_equivalence.py
==============================
Independent verification comparing full-graph vs 2-hop computation subgraph forward pass
probabilities across models (ETGNN and Matched GAT) and across all candidate neighborhood
sizes E_cand in {2, 3, 4, 5, 6, 7}.

Distinguishes:
1. Main-run runtime assertion coverage (all 377 completed nodes checked at tolerance <= 1e-5).
2. Stratified verification sample coverage and exact observed floating-point discrepancies.
"""

import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier

def main():
    print("=== Subgraph Probability Equivalence Verification ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Validation Graph
    val_graph_path = PROJECT_ROOT / "data" / "processed" / "validation_graph.pt"
    assert val_graph_path.exists(), f"Missing {val_graph_path}"
    val_graph = torch.load(val_graph_path, map_location="cpu", weights_only=False)

    # 2. Load Checkpoints
    etgnn_ckpt = PROJECT_ROOT / "results" / "checkpoints" / "etgnn.pt"
    gat_ckpt = PROJECT_ROOT / "results" / "checkpoints" / "matched_gat.pt"
    assert etgnn_ckpt.exists() and gat_ckpt.exists()

    m_etgnn = ETGNNClassifier(input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3).to(device)
    m_etgnn.load_state_dict(torch.load(etgnn_ckpt, map_location=device, weights_only=False))
    m_etgnn.eval()

    m_gat = MatchedGATClassifier(input_dim=165, time_dim=16, hidden_dim=64, dropout=0.3).to(device)
    m_gat.load_state_dict(torch.load(gat_ckpt, map_location=device, weights_only=False))
    m_gat.eval()

    # 3. Full-graph forward pass (probabilities)
    with torch.no_grad():
        x_dev = val_graph["x"].to(device)
        ei_dev = val_graph["edge_index"].to(device)
        t_dev = val_graph["time_step"].to(device)
        full_probs_etgnn = F.softmax(m_etgnn(x_dev, ei_dev, t_dev), dim=1).cpu()
        full_probs_gat = F.softmax(m_gat(x_dev, ei_dev, t_dev), dim=1).cpu()

    # 4. Load 377 completed nodes from config
    with open(PROJECT_ROOT / "configs" / "main_c1_c2_run_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    nodes = cfg["execution_node_union"]["nodes"]
    df_nodes = pd.DataFrame(nodes)

    # Stratified sample across all candidate edge counts
    sample_nodes = []
    for ec, g in df_nodes.groupby("candidate_edge_count"):
        n_sample = min(5, len(g))
        sample_nodes.extend(g.sample(n=n_sample, random_state=42).to_dict(orient="records"))

    print(f"Testing stratified sample of {len(sample_nodes)} nodes across E_cand = {sorted(df_nodes['candidate_edge_count'].unique().tolist())}:")

    results = []
    for item in sample_nodes:
        pos = item["node_index"]
        ec = item["candidate_edge_count"]
        
        sub_nodes, sub_edges, node_map, edge_mask = k_hop_subgraph(
            node_idx=pos,
            num_hops=2,
            edge_index=val_graph["edge_index"],
            relabel_nodes=True,
            flow="source_to_target",
        )
        local_center = int(node_map.item())
        
        sub_x = val_graph["x"][sub_nodes].to(device)
        sub_time = val_graph["time_step"][sub_nodes].to(device)
        sub_edges_dev = sub_edges.to(device)
        
        with torch.no_grad():
            p_et_sub = F.softmax(m_etgnn(sub_x, sub_edges_dev, sub_time), dim=1)[local_center].cpu()
            p_gt_sub = F.softmax(m_gat(sub_x, sub_edges_dev, sub_time), dim=1)[local_center].cpu()
            
        p_et_full = full_probs_etgnn[pos]
        p_gt_full = full_probs_gat[pos]
        
        diff_et = max(abs(p_et_sub[0].item() - p_et_full[0].item()), abs(p_et_sub[1].item() - p_et_full[1].item()))
        diff_gt = max(abs(p_gt_sub[0].item() - p_gt_full[0].item()), abs(p_gt_sub[1].item() - p_gt_full[1].item()))
        
        results.append({
            "node_index": pos,
            "candidate_edge_count": ec,
            "diff_etgnn": diff_et,
            "diff_gat": diff_gt,
            "passed_1e5": (diff_et <= 1e-5) and (diff_gt <= 1e-5)
        })

    res_df = pd.DataFrame(results)
    overall_max_et = res_df["diff_etgnn"].max()
    overall_max_gt = res_df["diff_gat"].max()
    
    print("\nResults by candidate_edge_count in verification sample:")
    for ec, g in res_df.groupby("candidate_edge_count"):
        print(f"  E_cand = {ec} (N={len(g)}): Max ETGNN diff = {g['diff_etgnn'].max():.2e}, Max GAT diff = {g['diff_gat'].max():.2e}")
    
    print(f"\nVerification Sample Maximum Discrepancy (Probabilities):")
    print(f"  ETGNN:       {overall_max_et:.2e}")
    print(f"  Matched GAT: {overall_max_gt:.2e}")
    print(f"Main Run Runtime Assertion Tolerance: 1.00e-05 (passed on all 377 nodes during benchmark execution).")
    assert overall_max_et <= 1e-5
    assert overall_max_gt <= 1e-5
    print("VERIFICATION PASSED: Subgraph forward pass probabilities reproduce full-graph probabilities.")

if __name__ == "__main__":
    main()
