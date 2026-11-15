"""
create_derived_metadata.py
==========================
Joins node_index from frozen execution configuration (configs/main_c1_c2_run_config.json)
to the raw benchmark results and creates metadata-corrected derived files with separate
etgnn_category and matched_gat_category columns.

Validates:
- Complete 1-to-1 mapping (377/377 unique execution nodes, 0 nulls).
- Many-to-one validation across repeated intervention draws.
- Strict numeric invariance across probabilities, deletion drops, edge sets, and results.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "main_c1_c2_run_config.json"
BASE_DIR = PROJECT_ROOT / "results" / "main_c1_c2_results"
DERIVED_DIR = BASE_DIR / "derived"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== Generating Corrected Derived Metadata Files ===")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    nodes = cfg["execution_node_union"]["nodes"]
    assert len(nodes) == 377, f"Expected 377 nodes, got {len(nodes)}"

    map_rows = []
    for n in nodes:
        map_rows.append({
            "node_index": int(n["node_index"]),
            "etgnn_category": str(n["etgnn_category"]),
            "matched_gat_category": str(n["matched_gat_category"]),
        })
    map_df = pd.DataFrame(map_rows)
    assert map_df["node_index"].nunique() == 377

    # 1. Main benchmark summary
    summary_orig = pd.read_csv(BASE_DIR / "main_benchmark_summary.csv")
    assert len(summary_orig) == 377
    summary_derived = summary_orig.merge(map_df, on="node_index", how="left")
    assert summary_derived["etgnn_category"].isnull().sum() == 0
    assert summary_derived["matched_gat_category"].isnull().sum() == 0
    cols_sum = list(summary_orig.columns)
    cat_idx = cols_sum.index("category")
    new_cols_sum = cols_sum[:cat_idx+1] + ["etgnn_category", "matched_gat_category"] + cols_sum[cat_idx+1:]
    summary_derived = summary_derived[new_cols_sum]

    for col in cols_sum:
        if col == "category":
            continue
        assert (summary_derived[col] == summary_orig[col]).all() or np.allclose(summary_derived[col], summary_orig[col], equal_nan=True)
    summary_derived.to_csv(DERIVED_DIR / "main_benchmark_summary_derived.csv", index=False)
    print(f"  Wrote {DERIVED_DIR / 'main_benchmark_summary_derived.csv'} ({len(summary_derived)} rows)")

    # 2. Raw benchmark records
    raw_orig = pd.read_csv(BASE_DIR / "raw_benchmark_records.csv")
    assert len(raw_orig) == 13195
    raw_derived = raw_orig.merge(map_df, on="node_index", how="left")
    assert raw_derived["etgnn_category"].isnull().sum() == 0
    assert raw_derived["matched_gat_category"].isnull().sum() == 0
    cols_raw = list(raw_orig.columns)
    cat_idx_raw = cols_raw.index("category")
    new_cols_raw = cols_raw[:cat_idx_raw+1] + ["etgnn_category", "matched_gat_category"] + cols_raw[cat_idx_raw+1:]
    raw_derived = raw_derived[new_cols_raw]

    for col in cols_raw:
        if col == "category":
            continue
        assert (raw_derived[col] == raw_orig[col]).all() or np.allclose(raw_derived[col], raw_orig[col], equal_nan=True)
    raw_derived.to_csv(DERIVED_DIR / "raw_benchmark_records_derived.csv", index=False)
    print(f"  Wrote {DERIVED_DIR / 'raw_benchmark_records_derived.csv'} ({len(raw_derived)} rows)")

    # 3. Detailed intervention records
    det_orig = pd.read_csv(BASE_DIR / "detailed_intervention_records.csv")
    assert len(det_orig) == 39585
    det_derived = det_orig.merge(map_df, on="node_index", how="left")
    assert det_derived["etgnn_category"].isnull().sum() == 0
    assert det_derived["matched_gat_category"].isnull().sum() == 0
    cols_det = list(det_orig.columns)
    cat_idx_det = cols_det.index("category")
    new_cols_det = cols_det[:cat_idx_det+1] + ["etgnn_category", "matched_gat_category"] + cols_det[cat_idx_det+1:]
    det_derived = det_derived[new_cols_det]

    for col in cols_det:
        if col == "category":
            continue
        assert (det_derived[col] == det_orig[col]).all() or np.allclose(det_derived[col], det_orig[col], equal_nan=True)
    det_derived.to_csv(DERIVED_DIR / "detailed_intervention_records_derived.csv", index=False)
    print(f"  Wrote {DERIVED_DIR / 'detailed_intervention_records_derived.csv'} ({len(det_derived)} rows)")

    # 4. C1 statistical results
    c1_orig = pd.read_csv(BASE_DIR / "c1_statistical_results.csv")
    assert len(c1_orig) == 240
    c1_derived = c1_orig.merge(map_df, on="node_index", how="left")
    assert c1_derived["etgnn_category"].isnull().sum() == 0
    assert c1_derived["matched_gat_category"].isnull().sum() == 0
    c1_cols = ["node_index", "etgnn_category", "matched_gat_category", "adv_etgnn", "adv_gat_common", "delta_adj", "delta_raw"]
    c1_derived = c1_derived[c1_cols]
    for col in ["adv_etgnn", "adv_gat_common", "delta_adj", "delta_raw"]:
        assert np.allclose(c1_derived[col], c1_orig[col], equal_nan=True)
    c1_derived.to_csv(DERIVED_DIR / "c1_statistical_results_derived.csv", index=False)
    print(f"  Wrote {DERIVED_DIR / 'c1_statistical_results_derived.csv'} ({len(c1_derived)} rows)")

    print("ALL DERIVED FILES CREATED SUCCESSFULLY WITH ZERO NUMERICAL MODIFICATIONS.")

if __name__ == "__main__":
    main()
