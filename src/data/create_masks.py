from pathlib import Path
import pandas as pd

path = Path("data/processed")
nodes = pd.read_csv(path / "node_labels.csv")

labelled = nodes["label"] != -1
nodes["train_mask"] = labelled & nodes["time_step"].between(1, 32)
nodes["val_mask"] = labelled & nodes["time_step"].between(33, 40)
nodes["test_mask"] = labelled & nodes["time_step"].between(41, 49)

masks = ["train_mask", "val_mask", "test_mask"]
assert (nodes[masks].sum(axis=1) <= 1).all(), "Masks overlap"
assert (nodes[masks].any(axis=1) == labelled).all(), "Labelled nodes missing"

nodes.to_csv(path / "nodes_with_masks.csv", index=False)

for mask in masks:
    subset = nodes.loc[nodes[mask]]
    print(mask, "total:", len(subset),
          "licit:", (subset["label"] == 0).sum(),
          "illicit:", (subset["label"] == 1).sum())
