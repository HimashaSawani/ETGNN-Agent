import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_binary
from src.models.time_encoding import encode_time
from src.models.gcn import GCNClassifier
from src.models.graphsage import GraphSAGEClassifier
from src.models.gat import GATClassifier
from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier

torch.set_num_threads(2)
output = Path("results/tables/final_test_results.csv")
if output.exists():
    raise FileExistsError(f"Final test results already exist: {output}")

nodes = pd.read_csv("data/processed/nodes_with_masks.csv")
features = np.load("data/processed/scaled_features.npy", mmap_mode="r")
test_mask = nodes["test_mask"].to_numpy(dtype=bool)
test_ids = np.flatnonzero(test_mask)
y_test = nodes.loc[test_mask, "label"].to_numpy()

graph = torch.load("data/processed/test_graph.pt",
                   map_location="cpu", weights_only=True)
graph_ids = graph["original_node_index"][graph["test_mask"]].numpy()
np.testing.assert_array_equal(test_ids, graph_ids)
np.testing.assert_array_equal(
    y_test, graph["y"][graph["test_mask"]].numpy()
)
assert len(y_test) == 9973
assert set(np.unique(y_test)) == {0, 1}

results = {}
probabilities = {}

def record(name, probability):
    probability = np.asarray(probability)
    assert probability.shape == (len(y_test),)
    assert np.isfinite(probability).all()
    probabilities[name] = probability
    results[name] = evaluate_binary(y_test, probability, threshold=0.5)
    print(name, "AUPRC:", round(results[name]["auprc"], 4), flush=True)

# Tabular models: only test-period labelled rows are scored.
rf = joblib.load("results/checkpoints/random_forest.joblib")
record("Random Forest", rf.predict_proba(features[test_mask])[:, 1])
del rf

rf_time = joblib.load("results/checkpoints/rf_with_time.joblib")
time_test = encode_time(
    torch.tensor(nodes.loc[test_mask, "time_step"].to_numpy())
).numpy()
x_test_time = np.concatenate((features[test_mask], time_test), axis=1)
record("RF + time", rf_time.predict_proba(x_test_time)[:, 1])
del rf_time, x_test_time

rf_local = joblib.load("results/checkpoints/rf_local.joblib") if Path(
    "results/checkpoints/rf_local.joblib"
).exists() else None
if rf_local is not None:
    record("RF local only",
           rf_local.predict_proba(features[test_mask, :93])[:, 1])
    del rf_local

xgb = XGBClassifier(n_jobs=2)
xgb.load_model("results/checkpoints/xgboost.json")
record("XGBoost", xgb.predict_proba(features[test_mask])[:, 1])
del xgb

def score_graph(name, model, checkpoint, local=False, no_edges=False):
    model.load_state_dict(torch.load(
        f"results/checkpoints/{checkpoint}",
        map_location="cpu", weights_only=True
    ))
    model.eval()
    x = graph["x"][:, :93] if local else graph["x"]
    edges = (
        torch.empty((2, 0), dtype=torch.long)
        if no_edges else graph["edge_index"]
    )
    with torch.no_grad():
        if isinstance(model, (ETGNNClassifier, MatchedGATClassifier)):
            logits = model(x, edges, graph["time_step"])
        else:
            logits = model(x, edges)
        probability = torch.softmax(
            logits[graph["test_mask"]], dim=1
        )[:, 1].numpy()
    record(name, probability)

score_graph("GCN", GCNClassifier(), "gcn.pt")
score_graph("GraphSAGE", GraphSAGEClassifier(), "graphsage.pt")
score_graph("GAT", GATClassifier(), "gat.pt")
score_graph("Matched GAT", MatchedGATClassifier(), "matched_gat.pt")
score_graph("ETGNN", ETGNNClassifier(), "etgnn.pt")
score_graph("ETGNN local only", ETGNNClassifier(input_dim=93),
            "etgnn_local.pt", local=True)
score_graph("ETGNN local no edges", ETGNNClassifier(input_dim=93),
            "etgnn_local_no_edges.pt", local=True, no_edges=True)

table = pd.DataFrame.from_dict(results, orient="index")
table.index.name = "model"
table.to_csv(output)

np.savez_compressed(
    "data/processed/final_test_probabilities.npz",
    y_test=y_test,
    **{f"model_{i}": p for i, p in enumerate(probabilities.values())},
)
print("\nTest illicit prevalence:", round(float(y_test.mean()), 4))
print(table.round(4).to_string())
print("\nSaved:", output)

