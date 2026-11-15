import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph

from src.models.etgnn import ETGNNClassifier


EXPLANATION_DIR = Path("results/explanations")
OUTPUT = Path("results/tables/explanation_edge_deletion_controls.csv")
CASES = ("tp", "fp", "fn", "tn")
RANDOM_SEED = 20260926

torch.set_num_threads(2)
graph = torch.load(
    "data/processed/validation_graph.pt", map_location="cpu", weights_only=True
)
model = ETGNNClassifier()
model.load_state_dict(
    torch.load(
        "results/checkpoints/etgnn.pt", map_location="cpu", weights_only=True
    )
)
model.eval()

rows = []
for case_number, case in enumerate(CASES):
    data = json.loads(
        (EXPLANATION_DIR / f"connected_{case}.json").read_text(
            encoding="utf-8"
        )
    )
    global_index = int(data["node_index"])
    positions = (graph["original_node_index"] == global_index).nonzero()
    assert positions.shape == (1, 1), f"Unexpected node match for {case}"
    center = int(positions[0, 0])
    subset, edges, mapping, _ = k_hop_subgraph(
        center,
        2,
        graph["edge_index"],
        relabel_nodes=True,
        flow="source_to_target",
    )
    x = graph["x"][subset]
    times = graph["time_step"][subset]
    local_center = int(mapping.item())
    edge_mask = np.asarray(data["full_edge_mask"], dtype=float)
    feature_mask = np.asarray(data["full_center_feature_mask"], dtype=float)

    assert len(subset) == int(data["subgraph_nodes"])
    assert edges.shape[1] == int(data["subgraph_edges"])
    assert edge_mask.shape == (edges.shape[1],)
    assert feature_mask.shape == (x.shape[1],) == (165,)
    assert np.isfinite(edge_mask).all() and np.isfinite(feature_mask).all()

    with torch.no_grad():
        original_probabilities = torch.softmax(
            model(x, edges, times)[local_center], dim=0
        )
    predicted_class = int(data["predicted_class"])
    original_probability = float(original_probabilities[predicted_class])
    assert np.isclose(
        original_probability,
        float(data["original_predicted_class_probability"]),
        atol=1e-7,
    )

    delete_count = min(5, edges.shape[1])
    order = np.argsort(edge_mask, kind="stable")
    rng = np.random.default_rng(RANDOM_SEED + case_number)
    selections = {
        "top": order[-delete_count:],
        "bottom": order[:delete_count],
        "random": np.sort(
            rng.choice(edges.shape[1], size=delete_count, replace=False)
        ),
    }

    for strategy, deleted in selections.items():
        keep = torch.ones(edges.shape[1], dtype=torch.bool)
        keep[torch.as_tensor(deleted, dtype=torch.long)] = False
        with torch.no_grad():
            probabilities = torch.softmax(
                model(x, edges[:, keep], times)[local_center], dim=0
            )
        after = float(probabilities[predicted_class])
        after_class = int(probabilities.argmax())
        rows.append(
            {
                "case": case.upper(),
                "strategy": strategy,
                "predicted_class": predicted_class,
                "total_edges": edges.shape[1],
                "deleted_edges": delete_count,
                "deleted_edge_fraction": delete_count / edges.shape[1],
                "probability_before": original_probability,
                "probability_after_deletion": after,
                "signed_probability_drop": original_probability - after,
                "predicted_class_after_deletion": after_class,
                "class_flipped": after_class != predicted_class,
                "random_seed": (
                    RANDOM_SEED + case_number
                    if strategy == "random"
                    else pd.NA
                ),
            }
        )

table = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUTPUT, index=False)
print(table.to_string(index=False))
print("Saved:", OUTPUT)
