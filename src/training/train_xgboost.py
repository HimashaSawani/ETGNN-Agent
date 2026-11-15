import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_binary

processed = Path("data/processed")
nodes = pd.read_csv(processed / "nodes_with_masks.csv")
features = np.load(processed / "scaled_features.npy", mmap_mode="r")

train = nodes["train_mask"].to_numpy(dtype=bool)
val = nodes["val_mask"].to_numpy(dtype=bool)
y_train = nodes.loc[train, "label"].to_numpy()
y_val = nodes.loc[val, "label"].to_numpy()

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=negative / positive,
    tree_method="hist",
    eval_metric="aucpr",
    early_stopping_rounds=20,
    n_jobs=2,
    random_state=42,
)
model.fit(
    features[train], y_train,
    eval_set=[(features[val], y_val)],
    verbose=False,
)

probability = model.predict_proba(features[val])[:, 1]
scores = evaluate_binary(y_val, probability)

Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)
model.save_model("results/checkpoints/xgboost.json")
Path("results/tables/xgb_validation.json").write_text(
    json.dumps(scores, indent=2), encoding="utf-8"
)

print("Best iteration:", model.best_iteration)
print("XGBoost validation results (threshold=0.5):")
print(json.dumps(scores, indent=2))
