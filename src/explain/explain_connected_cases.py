import json
from pathlib import Path

import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from src.models.etgnn import ETGNNClassifier

torch.set_num_threads(2)
graph = torch.load("data/processed/validation_graph.pt",
                   map_location="cpu", weights_only=True)
model = ETGNNClassifier()
model.load_state_dict(torch.load("results/checkpoints/etgnn.pt",
                                 map_location="cpu", weights_only=True))
model.eval()

chosen = {"TP": 133786, "FP": 151484, "FN": 148265, "TN": 154373}
out = Path("results/explanations")
out.mkdir(parents=True, exist_ok=True)

for number, (case, global_id) in enumerate(chosen.items()):
    torch.manual_seed(42 + number)
    center = int(
        (graph["original_node_index"] == global_id).nonzero()[0]
    )
    subset, edges, mapping, _ = k_hop_subgraph(
        center, 2, graph["edge_index"],
        relabel_nodes=True, flow="source_to_target",
    )
    x = graph["x"][subset]
    times = graph["time_step"][subset]
    local_center = int(mapping.item())

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
        x, edges, index=local_center, time_step=times
    )
    feature_scores = explanation.node_mask[local_center]
    feature_top = torch.topk(feature_scores, k=10)
    edge_top = torch.topk(explanation.edge_mask,
                          k=min(5, edges.shape[1]))

    with torch.no_grad():
        original = torch.softmax(
            model(x, edges, times)[local_center], dim=0
        )
        predicted_class = int(original.argmax())
        keep = torch.ones(edges.shape[1], dtype=torch.bool)
        keep[edge_top.indices] = False
        after_removal = torch.softmax(
            model(x, edges[:, keep], times)[local_center], dim=0
        )

    top_edges = []
    for pos, mask_value in zip(edge_top.indices, edge_top.values):
        source_local = int(edges[0, pos])
        target_local = int(edges[1, pos])
        top_edges.append({
            "source_node_index": int(
                graph["original_node_index"][subset[source_local]]
            ),
            "target_node_index": int(
                graph["original_node_index"][subset[target_local]]
            ),
            "mask_value": float(mask_value),
        })

    result = {
        "case": case,
        "node_index": global_id,
        "subgraph_nodes": len(subset),
        "subgraph_edges": edges.shape[1],
        "predicted_class": predicted_class,
        "original_predicted_class_probability": float(
            original[predicted_class]
        ),
        "probability_after_removing_top_edges": float(
            after_removal[predicted_class]
        ),
        "top_feature_columns_zero_based": feature_top.indices.tolist(),
        "top_feature_mask_values": feature_top.values.tolist(),
        "top_edges": top_edges,
        "full_center_feature_mask": feature_scores.detach().cpu().tolist(),
        "full_edge_mask": explanation.edge_mask.detach().cpu().tolist(),
    }
    (out / f"connected_{case.lower()}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        case, "edges:", edges.shape[1],
        "class:", predicted_class,
        "probability:", round(result[
            "original_predicted_class_probability"], 4),
        "after removing top edges:", round(result[
            "probability_after_removing_top_edges"], 4),
        "top features:", result["top_feature_columns_zero_based"][:5],
        flush=True,
    )
