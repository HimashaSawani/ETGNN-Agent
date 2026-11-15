import pandas as pd
from pathlib import Path

raw = Path("data/raw/elliptic")
saved = pd.read_csv("data/processed/node_mapping.csv")

features = pd.read_csv(
    raw / "elliptic_txs_features.csv",
    header=None,
    usecols=[0, 1],
    names=["txId", "time_step"],
)

assert features["txId"].equals(saved["txId"])
assert features["time_step"].equals(saved["time_step"])
assert saved["node_index"].equals(pd.Series(range(len(saved)), name="node_index"))
print("Feature rows and saved mapping align:", len(saved), "nodes")
