import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import evaluate_binary
from src.models.time_encoding import encode_time

nodes = pd.read_csv("data/processed/nodes_with_masks.csv")
features = np.load("data/processed/scaled_features.npy", mmap_mode="r")
time = encode_time(torch.tensor(nodes["time_step"].to_numpy())).numpy()

train = nodes["train_mask"].to_numpy(dtype=bool)
val = nodes["val_mask"].to_numpy(dtype=bool)
x_train = np.concatenate((features[train], time[train]), axis=1)
x_val = np.concatenate((features[val], time[val]), axis=1)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced_subsample",
    n_jobs=2,
    random_state=42,
)
model.fit(x_train, nodes.loc[train, "label"].to_numpy())
probability = model.predict_proba(x_val)[:, 1]
scores = evaluate_binary(nodes.loc[val, "label"].to_numpy(), probability)

joblib.dump(model, "results/checkpoints/rf_with_time.joblib")
Path("results/tables/rf_with_time_validation.json").write_text(
    json.dumps(scores, indent=2), encoding="utf-8"
)
print(json.dumps(scores, indent=2))
