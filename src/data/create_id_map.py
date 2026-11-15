from pathlib import Path
import pandas as pd

raw = Path("data/raw/elliptic")
processed = Path("data/processed")
processed.mkdir(parents=True, exist_ok=True)

nodes = pd.read_csv(
    raw / "elliptic_txs_features.csv",
    header=None,
    usecols=[0, 1],
    names=["txId", "time_step"],
)
nodes.insert(1, "node_index", range(len(nodes)))

if nodes["txId"].duplicated().any():
    raise ValueError("Duplicate transaction IDs found")

nodes.to_csv(processed / "node_mapping.csv", index=False)

print("Mapped nodes:", len(nodes))
print(nodes.head().to_string(index=False))
print("Saved:", processed / "node_mapping.csv")
