import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathlib import Path

ROOT = Path(".")
val_path = ROOT / "data" / "processed" / "validation_graph.pt"
c2_pairs_path = ROOT / "results" / "main_c1_c2_results" / "c2_individual_pair_differences.csv"

if not val_path.exists() or not c2_pairs_path.exists():
    print(f"Missing: val={val_path.exists()}, c2={c2_pairs_path.exists()}")
    exit()

val = torch.load(val_path, map_location="cpu", weights_only=True)
edge_index = val["edge_index"].numpy()
num_nodes = val["x"].shape[0]

# Compute weakly connected components of the validation graph
adj = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
n_components, labels = connected_components(adj, directed=True, connection="weak")

print(f"Validation Graph: {num_nodes:,} nodes, {edge_index.shape[1]:,} edges")
print(f"Number of weakly connected components: {n_components:,}")

df_pairs = pd.read_csv(c2_pairs_path)
print(f"Total pairs in c2_individual_pair_differences: {len(df_pairs)}")

# Map treat and ctrl nodes to components
# Note: check if treat_node_index corresponds to global index or validation local index
print("Treat index min/max:", df_pairs["treat_node_index"].min(), df_pairs["treat_node_index"].max())
print("Ctrl index min/max:", df_pairs["ctrl_node_index"].min(), df_pairs["ctrl_node_index"].max())
print("Val nodes total:", num_nodes)

