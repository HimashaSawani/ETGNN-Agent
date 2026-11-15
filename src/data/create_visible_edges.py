import pandas as pd
from pathlib import Path

path = Path("data/processed")
nodes = pd.read_csv(path / "node_mapping.csv")
edges = pd.read_csv(path / "mapped_edges.csv")
time_by_index = nodes.set_index("node_index")["time_step"]

source_time = edges["source"].map(time_by_index)
target_time = edges["target"].map(time_by_index)

for name, cutoff in [("train", 32), ("validation", 40), ("test", 49)]:
    visible = (source_time <= cutoff) & (target_time <= cutoff)
    selected = edges.loc[visible]
    selected.to_csv(path / f"{name}_visible_edges.csv", index=False)
    print(name, "visible nodes:", (nodes["time_step"] <= cutoff).sum(),
          "visible edges:", len(selected))
