from pathlib import Path
import numpy as np
import pandas as pd
import torch

out = Path("data/processed")
nodes = pd.read_csv(out / "nodes_with_masks.csv")
edges = pd.read_csv(out / "mapped_edges.csv")
features = np.load(out / "scaled_features.npy", mmap_mode="r")
total = len(nodes)

source = edges["source"].to_numpy(dtype=np.int64)
target = edges["target"].to_numpy(dtype=np.int64)
times = nodes["time_step"].to_numpy()

for name, cutoff in [("train", 32), ("validation", 40), ("test", 49)]:
    global_ids = np.flatnonzero(times <= cutoff)
    local_index = np.full(total, -1, dtype=np.int64)
    local_index[global_ids] = np.arange(len(global_ids))

    edge_visible = (times[source] <= cutoff) & (times[target] <= cutoff)
    local_edges = np.stack([
        local_index[source[edge_visible]],
        local_index[target[edge_visible]],
    ])

    selected = nodes.iloc[global_ids]
    graph = {
        "x": torch.from_numpy(np.array(features[global_ids], copy=True)),
        "edge_index": torch.from_numpy(local_edges),
        "y": torch.tensor(selected["label"].to_numpy(), dtype=torch.long),
        "time_step": torch.tensor(selected["time_step"].to_numpy(), dtype=torch.long),
        "train_mask": torch.tensor(selected["train_mask"].to_numpy(), dtype=torch.bool),
        "val_mask": torch.tensor(selected["val_mask"].to_numpy(), dtype=torch.bool),
        "test_mask": torch.tensor(selected["test_mask"].to_numpy(), dtype=torch.bool),
        "original_node_index": torch.from_numpy(global_ids),
    }
    torch.save(graph, out / f"{name}_graph.pt")
    print(name, "nodes:", len(global_ids), "edges:", local_edges.shape[1])
