from pathlib import Path

import pandas as pd
import torch

from src.models.etgnn import ETGNNClassifier

torch.set_num_threads(2)
graph = torch.load("data/processed/validation_graph.pt",
                   map_location="cpu", weights_only=True)
model = ETGNNClassifier()
model.load_state_dict(torch.load("results/checkpoints/etgnn.pt",
                                 map_location="cpu", weights_only=True))
model.eval()

with torch.no_grad():
    probability = torch.softmax(
        model(graph["x"], graph["edge_index"], graph["time_step"]),
        dim=1,
    )[:, 1]

mask = graph["val_mask"]
selected = pd.DataFrame({
    "node_index": graph["original_node_index"][mask].numpy(),
    "time_step": graph["time_step"][mask].numpy(),
    "actual": graph["y"][mask].numpy(),
    "probability": probability[mask].numpy(),
})
selected["predicted"] = (selected["probability"] >= 0.5).astype(int)
selected["case"] = selected.apply(
    lambda row: ("T" if row.actual == row.predicted else "F")
    + ("P" if row.predicted == 1 else "N"),
    axis=1,
)
selected["confidence"] = abs(selected["probability"] - 0.5)

cases = (
    selected.sort_values("confidence", ascending=False)
    .groupby("case", group_keys=False)
    .head(3)
    .drop(columns="confidence")
)
out = Path("results/explanations")
out.mkdir(parents=True, exist_ok=True)
cases.to_csv(out / "selected_validation_cases.csv", index=False)

print("Selected cases:")
print(cases["case"].value_counts().to_string())
print(cases.sort_values("case").to_string(index=False))
