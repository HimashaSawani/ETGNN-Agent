import json
from pathlib import Path

import numpy as np
import pandas as pd


folder = Path("results/explanations")
rows = []

for case in ("tp", "fp", "fn", "tn"):
    data = json.loads((folder / f"connected_{case}.json").read_text())

    for mask_type, values in (
        ("center_features", data["full_center_feature_mask"]),
        ("edges", data["full_edge_mask"]),
    ):
        mask = np.asarray(values, dtype=float)
        assert mask.size > 0 and np.isfinite(mask).all()
        assert ((mask >= 0) & (mask <= 1)).all()

        for threshold in (0.3, 0.5, 0.7):
            selected = int((mask >= threshold).sum())
            rows.append(
                {
                    "case": case.upper(),
                    "mask_type": mask_type,
                    "threshold": threshold,
                    "total_entries": int(mask.size),
                    "selected_entries": selected,
                    "sparsity": float(1 - selected / mask.size),
                }
            )

table = pd.DataFrame(rows)
output = Path("results/tables/explanation_mask_sparsity.csv")
table.to_csv(output, index=False)
print(table.to_string(index=False))
print("Saved:", output)
