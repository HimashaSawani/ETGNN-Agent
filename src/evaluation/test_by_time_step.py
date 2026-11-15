from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

nodes = pd.read_csv("data/processed/nodes_with_masks.csv")
test = nodes.loc[nodes["test_mask"]].reset_index(drop=True)
saved = np.load("data/processed/final_test_probabilities.npz")

y = saved["y_test"]
assert len(test) == len(y) == 9973
np.testing.assert_array_equal(test["label"].to_numpy(), y)

models = {
    "rf_auprc": saved["model_0"],
    "etgnn_auprc": saved["model_8"],
    "no_edge_auprc": saved["model_10"],
}
rows = []
for step in range(41, 50):
    mask = test["time_step"].to_numpy() == step
    actual = y[mask]
    row = {
        "time_step": step,
        "labelled_count": int(mask.sum()),
        "illicit_count": int(actual.sum()),
        "illicit_prevalence": float(actual.mean()),
    }
    for name, probability in models.items():
        row[name] = float(average_precision_score(
            actual, probability[mask]
        ))
    rows.append(row)

table = pd.DataFrame(rows)
out = Path("results/tables/test_by_time_step.csv")
table.to_csv(out, index=False)
print(table.round(4).to_string(index=False))
