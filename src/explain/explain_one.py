import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from src.models.etgnn import ETGNNClassifier

torch.manual_seed(42)
torch.set_num_threads(2)

graph = torch.load("data/processed/validation_graph.pt",
                   map_location="cpu", weights_only=True)
cases = pd.read_csv("results/explanations/selected_validation_cases.csv")
global_id = int(cases.loc[cases["case"] == "TP", "node_index"].iloc[0])
center = int((graph["original_node_index"] == global_id).nonzero()[0])

subset, edge_index, mapping, _ = k_hop_subgraph(
    center, 2, graph["edge_index"],
    relabel_nodes=True, flow="source_to_target",
)

model = ETGNNClassifier()
model.load_state_dict(torch.load("results/checkpoints/etgnn.pt",
                                 map_location="cpu", weights_only=True))
model.eval()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=50),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config={
        "mode": "multiclass_classification",
        "task_level": "node",
        "return_type": "raw",
    },
)
explanation = explainer(
    graph["x"][subset],
    edge_index,
    index=int(mapping.item()),
    time_step=graph["time_step"][subset],
)

feature_scores = explanation.node_mask[int(mapping.item())]
top_features = torch.topk(feature_scores, k=10)
top_edges = torch.topk(
    explanation.edge_mask,
    k=min(10, explanation.edge_mask.numel()),
)

result = {
    "case": "TP",
    "original_node_index": global_id,
    "subgraph_nodes": int(len(subset)),
    "subgraph_edges": int(edge_index.shape[1]),
    "top_feature_columns_zero_based": top_features.indices.tolist(),
    "top_feature_mask_values": top_features.values.tolist(),
    "top_edge_source_node_indices": subset[
        edge_index[0, top_edges.indices]
    ].tolist(),
    "top_edge_target_node_indices": subset[
        edge_index[1, top_edges.indices]
    ].tolist(),
    "top_edge_mask_values": top_edges.values.tolist(),
}
out = Path("results/explanations/tp_example.json")
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
