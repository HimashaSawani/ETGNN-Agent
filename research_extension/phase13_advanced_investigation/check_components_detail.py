import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathlib import Path

ROOT = Path(".")
val_path = ROOT / "data" / "processed" / "validation_graph.pt"
c2_pairs_path = ROOT / "results" / "main_c1_c2_results" / "c2_individual_pair_differences.csv"

val = torch.load(val_path, map_location="cpu", weights_only=True)
edge_index = val["edge_index"].numpy()
num_nodes = val["x"].shape[0]

# Compute components on undirected version (weakly connected components)
row = np.concatenate([edge_index[0], edge_index[1]])
col = np.concatenate([edge_index[1], edge_index[0]])
data = np.ones(len(row))
adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
n_components, comp_labels = connected_components(adj, directed=False)

print(f"Number of weakly connected components (undirected): {n_components:,}")

df_pairs = pd.read_csv(c2_pairs_path)
t_idx = df_pairs["treat_node_index"].values
c_idx = df_pairs["ctrl_node_index"].values

df_pairs["treat_comp"] = comp_labels[t_idx]
df_pairs["ctrl_comp"] = comp_labels[c_idx]
df_pairs["same_comp"] = df_pairs["treat_comp"] == df_pairs["ctrl_comp"]

print("Same component fraction:", df_pairs["same_comp"].mean())
print("Unique treat components:", df_pairs["treat_comp"].nunique())
print("Unique ctrl components:", df_pairs["ctrl_comp"].nunique())

# Group by register
for reg, group in df_pairs.groupby("register_name"):
    unique_clusters = len(set(group["treat_comp"]).union(set(group["ctrl_comp"])))
    print(f"  Register: {reg:30s} | Pairs: {len(group):3d} | Unique Graph Clusters: {unique_clusters:3d} | Shared Comp: {group['same_comp'].sum():2d}")

