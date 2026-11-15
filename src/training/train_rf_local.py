import joblib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import evaluate_binary

nodes = pd.read_csv("data/processed/nodes_with_masks.csv")
features = np.load("data/processed/scaled_features.npy", mmap_mode="r")

train = nodes["train_mask"].to_numpy(dtype=bool)
val = nodes["val_mask"].to_numpy(dtype=bool)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced_subsample",
    n_jobs=2,
    random_state=42,
)
model.fit(features[train, :93], nodes.loc[train, "label"].to_numpy())
joblib.dump(model, "results/checkpoints/rf_local.joblib")
probability = model.predict_proba(features[val, :93])[:, 1]
scores = evaluate_binary(nodes.loc[val, "label"].to_numpy(), probability)

Path("results/tables/rf_local_validation.json").write_text(
    json.dumps(scores, indent=2), encoding="utf-8"
)
print("RF using 93 local numeric features:")
print(json.dumps(scores, indent=2))

