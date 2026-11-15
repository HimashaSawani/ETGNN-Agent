import json
from pathlib import Path

import pandas as pd


EXPLANATION_DIR = Path("results/explanations")
OUTPUT = Path("results/tables/explanation_deletion_summary.csv")


rows = []
for case in ("tp", "fp", "fn", "tn"):
    data = json.loads(
        (EXPLANATION_DIR / f"connected_{case}.json").read_text(
            encoding="utf-8"
        )
    )
    total_edges = int(data["subgraph_edges"])
    selected_edges = min(5, total_edges)
    before = float(data["original_predicted_class_probability"])
    after = float(data["probability_after_removing_top_edges"])
    rows.append(
        {
            "case": case.upper(),
            "predicted_class": int(data["predicted_class"]),
            "total_edges": total_edges,
            "selected_edges": selected_edges,
            "selected_edge_fraction": selected_edges / total_edges,
            "probability_before": before,
            "probability_after_deletion": after,
            "signed_deletion_drop": before - after,
            "absolute_probability_change": abs(before - after),
        }
    )

table = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUTPUT, index=False)
print(table.to_string(index=False))
print("Saved:", OUTPUT)
