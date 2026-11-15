import pandas as pd
from pathlib import Path

path = Path("data/processed")
nodes = pd.read_csv(path / "node_mapping.csv")
edges = pd.read_csv(path / "mapped_edges.csv")

time_by_index = nodes.set_index("node_index")["time_step"]
source_time = edges["source"].map(time_by_index)
target_time = edges["target"].map(time_by_index)

assert source_time.notna().all() and target_time.notna().all()
assert (source_time == target_time).all()

for name, lo, hi in [
    ("train", 1, 32),
    ("validation", 33, 40),
    ("test", 41, 49),
]:
    print(name, "edges:", source_time.between(lo, hi).sum())
