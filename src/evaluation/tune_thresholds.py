import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_binary
from src.models.gcn import GCNClassifier

torch.set_num_threads(2)
out = Path("results/tables")
nodes = pd.read_csv("data/processed/nodes_with_masks.csv")
features = np.load("data/processed/scaled_features.npy", mmap_mode="r")
mask = nodes["val_mask"].to_numpy(dtype=bool)
y = nodes.loc[mask, "label"].to_numpy()

rf = joblib.load("results/checkpoints/random_forest.joblib")
rf_prob = rf.predict_proba(features[mask])[:, 1]

xgb = XGBClassifier()
xgb.load_model("results/checkpoints/xgboost.json")
xgb_prob = xgb.predict_proba(features[mask])[:, 1]

val_graph = torch.load("data/processed/validation_graph.pt",
                       map_location="cpu", weights_only=True)
gcn = GCNClassifier()
gcn.load_state_dict(torch.load("results/checkpoints/gcn.pt",
                               map_location="cpu", weights_only=True))
gcn.eval()
with torch.no_grad():
    gcn_prob = torch.softmax(
        gcn(val_graph["x"], val_graph["edge_index"])[val_graph["val_mask"]],
        dim=1,
    )[:, 1].numpy()

results = {}
for name, prob in [
    ("Random Forest", rf_prob),
    ("XGBoost", xgb_prob),
    ("GCN", gcn_prob),
]:
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best = int(np.argmax(f1_values))
    threshold = float(thresholds[best])
    scores = evaluate_binary(y, prob, threshold)
    results[name] = {"threshold": threshold, **scores}

out.joinpath("validation_tuned_thresholds.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
print(pd.DataFrame.from_dict(results, orient="index").round(4).to_string())
