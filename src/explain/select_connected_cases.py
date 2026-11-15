import pandas as pd
import torch
from torch_geometric.utils import degree, k_hop_subgraph

from src.models.etgnn import ETGNNClassifier

torch.set_num_threads(2)
graph = torch.load("data/processed/validation_graph.pt",
                   map_location="cpu", weights_only=True)

model = ETGNNClassifier()
model.load_state_dict(torch.load("results/checkpoints/etgnn.pt",
                                 map_location="cpu", weights_only=True))
model.eval()
with torch.no_grad():
    prob = torch.softmax(
        model(graph["x"], graph["edge_index"], graph["time_step"]),
        dim=1,
    )[:, 1]

incoming = degree(
    graph["edge_index"][1], num_nodes=graph["x"].shape[0]
).long()
positions = graph["val_mask"].nonzero().flatten()

rows = []
for case in ("TP", "FP", "FN", "TN"):
    candidates = []
    for pos in positions.tolist():
        actual = int(graph["y"][pos])
        predicted = int(prob[pos] >= 0.5)
        label = ("T" if actual == predicted else "F") + (
            "P" if predicted == 1 else "N"
        )
        if label == case and incoming[pos] >= 2:
            candidates.append(pos)

    candidates.sort(key=lambda pos: int(incoming[pos]), reverse=True)
    for pos in candidates[:3]:
        subset, edges, _, _ = k_hop_subgraph(
            pos, 2, graph["edge_index"],
            relabel_nodes=True, flow="source_to_target",
        )
        rows.append({
            "case": case,
            "node_index": int(graph["original_node_index"][pos]),
            "time_step": int(graph["time_step"][pos]),
            "probability": float(prob[pos]),
            "incoming_edges": int(incoming[pos]),
            "two_hop_nodes": len(subset),
            "two_hop_edges": edges.shape[1],
        })

table = pd.DataFrame(rows)
print(table.to_string(index=False) if len(table) else
      "No validation candidates with at least two incoming edges")
table.to_csv(
    "results/explanations/connected_validation_cases.csv", index=False
)
