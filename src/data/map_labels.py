from pathlib import Path
import pandas as pd

raw = Path("data/raw/elliptic")
processed = Path("data/processed")

nodes = pd.read_csv(processed / "node_mapping.csv")
classes = pd.read_csv(raw / "elliptic_txs_classes.csv", dtype={"class": str})

aligned = nodes.merge(classes, on="txId", how="left", validate="one_to_one")
if aligned["class"].isna().any():
    raise ValueError("Some transactions have no class record")

aligned["label"] = aligned["class"].map({"1": 1, "2": 0, "unknown": -1})
if aligned["label"].isna().any():
    raise ValueError("Unexpected class value found")

aligned[["txId", "node_index", "time_step", "label"]].to_csv(
    processed / "node_labels.csv", index=False
)
print(aligned["label"].value_counts().sort_index().to_string())
