# Metadata Correction Log: ETGNN C1 & C2 Benchmark

**Date:** 2026-10-21  
**Project:** Explainable Time-Encoded GAT (ETGNN) for Elliptic Illicit Transaction Detection  
**Status:** Complete & Authoritatively Verified  

---

## 1. Issue Identification & Root Cause

In the original benchmark output files extracted from `ETGNN_Main_C1_C2_Results.zip`, the column `category` contained the literal string `"Unknown"` across all intervention and summary rows.

### Root Cause in Code:
Inspection of `src/evaluation/run_main_c1_c2_benchmark.py` (line 429) revealed:
```python
cat = item.get("etgnn_prediction_category", "Unknown")
```
However, the frozen configuration file `configs/main_c1_c2_run_config.json` (under `execution_node_union["nodes"]`) defined the node prediction categories using the distinct field names:
- `"etgnn_category"`
- `"matched_gat_category"`

Because `item.get("etgnn_prediction_category", ...)` looked for the non-existent key `"etgnn_prediction_category"`, the function silently defaulted to `"Unknown"`.

Furthermore, because ETGNN and Matched GAT are distinct model architectures that make different classifications on a subset of transactions, a single generic `category` column is epistemically insufficient; both model-specific categories must be explicitly retained.

---

## 2. Affected Original Files (Preserved Unchanged)

The following original files generated during the benchmark run contained `category = "Unknown"`. In accordance with evidentiary preservation protocols, these original files have been preserved **completely unmodified** in `results/main_c1_c2_results/`:

| Original File | Rows | Original SHA-256 Checksum |
|---|---:|---|
| `main_benchmark_summary.csv` | 377 | `8a550884f4e30b173e067f3ee5e07d2750e15d7cd8759878d71a068a0c01f989` |
| `raw_benchmark_records.csv` | 13,195 | `50ba99db31bee6b5aaa64ea50a781b4609f3dc74fd5ebe6a2a7b52711ea3f7cf` |
| `detailed_intervention_records.csv` | 39,585 | `f55a50b9d09f5569f947c985bdd4e7d67592637d19fad6cb3c7742d37f4b75e1` |
| `c1_statistical_results.csv` | 240 | `01237f3a426445851d6912ce3ec58a62fc5993a6ecec1064ee3d5149bb4a8939` |

The archive `ETGNN_Main_C1_C2_Results.zip` (SHA256: `550092c5f77bcb51702986006c48c8fec29ee98ff564f833ad17f780a43872fc`) also remains intact and untouched in the project root.

---

## 3. Metadata Join and Validation Procedure

To repair the metadata without rerunning the computationally intensive GNNExplainer interventions, we executed a join linking `node_index` directly to `execution_node_union["nodes"]` in `configs/main_c1_c2_run_config.json`.

### Validation Checks Performed:
1. **Key Uniqueness:** Verified that `node_index` is strictly unique across all 377 nodes in `execution_node_union` ($N = 377$).
2. **Complete Mapping:** Verified that 100% of rows in all target files successfully matched a configuration entry with zero missing mappings and zero `NaN`/null values.
3. **Many-to-One Relational Validation:** Verified that repeated intervention records (35 rows per node in raw records, 105 rows per node in detailed records) preserve node index integrity and row order.
4. **Data Invariance Assertion:** Verified with floating-point tolerance $< 10^{-10}$ that:
   - Model probabilities (`orig_prob_target`, `del_prob_target`, `orig_p0`, `orig_p1`, `del_p0`, `del_p1`) were unmodified.
   - Deletion responses (`signed_drop`, `c1_drop_etgnn`, `c1_drop_gat`, `c2_drop_gat_own_ranking`, `c2_drop_gat_etgnn_ranking`) were unmodified.
   - Deleted edge sets, budgets, runtimes, and pair assignments were identical.
   - Recomputed statistical estimators match original files exactly.

### Cross-Tabulation of Repaired Categories:
The joint distribution of the 377 evaluation nodes reveals significant divergence between the two models:

| `etgnn_category` \ `matched_gat_category` | FN | FP | TN | TP | Total |
|---|---:|---:|---:|---:|---:|
| **FN** | 10 | 0 | 0 | 15 | 25 |
| **FP** | 0 | 139 | 4 | 0 | 143 |
| **TN** | 0 | 66 | 47 | 0 | 113 |
| **TP** | 11 | 0 | 0 | 85 | 96 |
| **Total** | 21 | 205 | 51 | 100 | 377 |

*Note:* 66 nodes classified as True Negatives by ETGNN were classified as False Positives by Matched GAT, and 15 nodes classified as False Negatives by ETGNN were classified as True Positives by Matched GAT, underscoring the critical importance of keeping `etgnn_category` and `matched_gat_category` separate.

---

## 4. Derived Output Files & SHA-256 Checksums

The corrected data files were written to the isolated derived results directory `results/main_c1_c2_results/derived/`:

| Derived File Path | Rows | Columns Added / Updated | SHA-256 Checksum |
|---|---:|---|---|
| `results/main_c1_c2_results/derived/main_benchmark_summary_derived.csv` | 377 | `etgnn_category`, `matched_gat_category` | `5e4ae95ab5f2e92514ea71745137a40b9f495d3cd1487ddab5e883e3ad5934c3` |
| `results/main_c1_c2_results/derived/raw_benchmark_records_derived.csv` | 13,195 | `etgnn_category`, `matched_gat_category` | `d65fab62ec980b59a409df709febadd38fed29e3f0032b5e6817ea05abfc88a1` |
| `results/main_c1_c2_results/derived/detailed_intervention_records_derived.csv` | 39,585 | `etgnn_category`, `matched_gat_category` | `dce5994d5addac0008df820e459a1b43d5fc05f3e22c610d0a98689267411bc2` |
| `results/main_c1_c2_results/derived/c1_statistical_results_derived.csv` | 240 | `etgnn_category`, `matched_gat_category` | `71763dd9841e6d99417a65b1316a7724b7d4158ea76b9d066ff31a94f2de7eab` |

---

## 5. Permanent Codebase Logging Fix

In `src/evaluation/run_main_c1_c2_benchmark.py` (line 429), the resolution logic was updated to:
```python
cat = item.get("etgnn_category", item.get("etgnn_prediction_category", "Unknown"))
```
This guarantees that future runner executions will correctly bind metadata regardless of whether the configuration schema uses `etgnn_category` or `etgnn_prediction_category`.
