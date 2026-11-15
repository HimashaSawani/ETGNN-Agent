"""
Technical Pilot Execution Script (DEVELOPMENT PERIOD STEPS 33-36 ONLY)
=====================================================================
DO NOT EXECUTE UNTIL MATCHING OUTPUTS ARE FORMALLY APPROVED.
This script benchmarks actual runtime, CPU/GPU utilization, and memory footprint
for GNNExplainer (3 seeds) + 30 random deletion draws on a minimal sample of 4
development nodes (1 TP, 1 FP, 1 FN, 1 TN from steps 33-36).
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier


def get_mem_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def main():
    print("=== TECHNICAL PILOT: RUNTIME & MEMORY BENCHMARK (STEPS 33-36) ===")
    print("Warning: This script is prepared for review and benchmark execution upon confirmation.")
    
    val_graph = torch.load("data/processed/validation_graph.pt", map_location="cpu", weights_only=True)
    df_feas = pd.read_csv("experiments/matched_c1_c2_audit/feasibility_etgnn_full.csv")
    
    # Restrict strictly to development period steps 33-36, non-pilot, candidate_edges >= 2
    dev_eligible = df_feas[
        (df_feas["time_step"].between(33, 36)) &
        (~df_feas["historical_pilot_flag"]) &
        (df_feas["candidate_edge_count"] >= 2) &
        (df_feas["in_degree"] >= 1) &
        (df_feas["recorded_label"] != -1)
    ]
    
    print(f"Eligible development nodes in steps 33-36: {len(dev_eligible)}")
    
    # Select 1 node per category
    sample_nodes = []
    for cat in ["TP", "FP", "FN", "TN"]:
        sub = dev_eligible[dev_eligible["prediction_category"] == cat]
        if len(sub) > 0:
            sample_nodes.append(sub.iloc[0])
            
    sample_df = pd.DataFrame(sample_nodes)
    print("Selected 4 benchmark nodes:")
    print(sample_df[["prediction_category", "node_index", "transaction_id", "time_step", "candidate_edge_count", "margin"]].to_string())
    
    # Save the planned sample
    out_path = Path("experiments/matched_c1_c2_audit/technical_pilot_sample.csv")
    sample_df.to_csv(out_path, index=False)
    print(f"Saved benchmark target sample to {out_path}.")
    print("Execution halted. Ready for execution once protocol review passes.")


if __name__ == "__main__":
    main()
