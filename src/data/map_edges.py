from pathlib import Path
import pandas as pd

raw = Path("data/raw/elliptic")
processed = Path("data/processed")

mapping = pd.read_csv(processed / "node_mapping.csv")
edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
id_to_index = mapping.set_index("txId")["node_index"]

mapped = pd.DataFrame({
    "source": edges["txId1"].map(id_to_index),
    "target": edges["txId2"].map(id_to_index),
})

if mapped.isna().any().any():
    raise ValueError("An edge refers to a missing transaction ID")

mapped = mapped.astype("int64")
mapped.to_csv(processed / "mapped_edges.csv", index=False)

print("Mapped edges:", len(mapped))
print(mapped.head().to_string(index=False))
print("Saved:", processed / "mapped_edges.csv")
