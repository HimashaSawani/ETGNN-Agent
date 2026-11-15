import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import evaluate_binary

processed = Path("data/processed")
nodes = pd.read_csv(processed / "nodes_with_masks.csv")
features = np.load(processed / "scaled_features.npy", mmap_mode="r")

train = nodes["train_mask"].to_numpy(dtype=bool)
val = nodes["val_mask"].to_numpy(dtype=bool)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced_subsample",
    n_jobs=2,
    random_state=42,
)
model.fit(features[train], nodes.loc[train, "label"].to_numpy())

probability = model.predict_proba(features[val])[:, 1]
scores = evaluate_binary(
    nodes.loc[val, "label"].to_numpy(), probability
)

Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)
joblib.dump(model, "results/checkpoints/random_forest.joblib")
Path("results/tables/rf_validation.json").write_text(
    json.dumps(scores, indent=2), encoding="utf-8"
)

print("Random Forest validation results (threshold=0.5):")
print(json.dumps(scores, indent=2))
