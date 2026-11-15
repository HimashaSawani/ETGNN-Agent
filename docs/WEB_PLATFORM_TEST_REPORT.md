# ETGNN Web Platform Automated Test & Quality Assurance Report

## 1. Executive Summary

| Category | Status | Details |
|---|---|---|
| **Backend Test Suite** | **13 / 13 PASSED** | Executed via `pytest` and `fastapi.testclient.TestClient` |
| **Frontend Production Build** | **PASSED (0 errors)** | Compiled via `tsc -b && vite build` (1,943 modules) |
| **Browser Subagent E2E Session** | **VERIFIED** | Navigated all 9 tabs on live server at `http://127.0.0.1:8000/` |
| **Artifact Integrity** | **UNMODIFIED** | Zero checkpoints, datasets, or thesis documents altered |
| **Execution Environment** | Windows 11 / Python 3.13 / Node v22.15.1 | CPU-friendly, no GPU required |

---

## 2. Automated Backend Test Matrix (`tests/test_web_api.py`)

| Test Function | Target Endpoint | Verification Criteria | Status |
|---|---|---|---|
| `test_health_endpoint` | `GET /api/health` | Healthy status, dataset scale (203k), loaded models | **PASSED** |
| `test_models_metadata_endpoint` | `GET /api/models` | Specifications for ETGNN, Matched GAT, RF, XGB | **PASSED** |
| `test_benchmark_metrics_parity` | `GET /api/metrics` | Exact numerical parity with `final_test_results.csv` (RF: 0.6323, ETGNN: 0.2250) | **PASSED** |
| `test_transactions_pagination_and_filtering` | `GET /api/transactions` | Pagination (limit/offset), split filtering, label filtering | **PASSED** |
| `test_get_single_transaction_and_404` | `GET /api/transactions/{id}` | Correct node features, degree, predictions; 404 on invalid ID | **PASSED** |
| `test_get_subgraph` | `GET /api/transactions/{id}/subgraph` | 2-hop neighborhood extraction, node and edge counts, center node flag | **PASSED** |
| `test_predict_endpoint` | `POST /api/predict` | Multi-model live forward pass, probabilities in [0, 1] | **PASSED** |
| `test_predict_with_feature_override` | `POST /api/predict` | Dynamic feature modification with exact 2-hop GAT forward pass | **PASSED** |
| `test_get_explanation_frozen_exemplar` | `GET /api/explanations/{id}` | Frozen TP exemplar (Tx 288717023), top edges, safety notice | **PASSED** |
| `test_research_c1_endpoint` | `GET /api/research/c1` | Inconclusive hypothesis disclosure, cohort records from CSV | **PASSED** |
| `test_research_c2_endpoint` | `GET /api/research/c2` | All 8 registers, paired differences, 95% bootstrap CIs | **PASSED** |
| `test_research_ablations_endpoint` | `GET /api/research/ablations` | Multi-seed repeat ablation data, +0.2468 edge anomaly | **PASSED** |
| `test_spa_index_served` | `GET /` | Root serves compiled React SPA index.html with 200 OK | **PASSED** |

---

## 3. Numerical & Scientific Parity Validation

### 3.1. Held-Out Test Evaluation Parity
The platform was cross-referenced against `results/tables/final_test_results.csv` on chronological test steps 41–49 (N=9,973 nodes, illicit prevalence=10.47%):
- **Random Forest**: Precision = 0.6966, Recall = 0.5782, F1 = 0.6319, ROC-AUC = 0.8735, AUPRC = **0.6323**
- **XGBoost**: Precision = 0.2823, Recall = 0.6050, F1 = 0.3849, ROC-AUC = 0.8074, AUPRC = **0.5472**
- **ETGNN**: Precision = 0.2614, Recall = 0.4504, F1 = 0.3308, ROC-AUC = 0.8062, AUPRC = **0.2250**
- **Matched GAT**: Precision = 0.1205, Recall = 0.7424, F1 = 0.2074, ROC-AUC = 0.8175, AUPRC = **0.2099**
- **ETGNN Local No Edges**: Precision = 0.2445, Recall = 0.6164, F1 = 0.3501, ROC-AUC = 0.8107, AUPRC = **0.4812**

### 3.2. Subgraph Forward Pass Equivalence
A unit test proved that extracting a 2-hop local subgraph using `torch_geometric.utils.k_hop_subgraph` and passing it through the 2-layer ETGNN architecture produces the identical prediction probability as whole-graph inference down to floating-point precision:
$$\left| P_{\text{subgraph}} - P_{\text{full\_graph}} \right| = 0.000000 \times 10^0$$

---

## 4. End-to-End Browser Subagent Verification

An automated browser subagent executed a full interactive walkthrough on the live platform (`http://127.0.0.1:8000/`):
1. **Header & Navigation**: Verified system title, prototype status badge, and all 9 navigation tabs.
2. **Overview Tab**: Confirmed display of key metrics (`203,769` transactions, `0.6323` top baseline AUPRC, `0.2250` ETGNN AUPRC, `+0.2468` edge removal jump) and the 4 exemplar cards (TP, FP, TN, FN).
3. **Transaction Explorer Tab**: Verified paginated transaction table loading, split filters (train, validation, test), and action buttons.
4. **Prediction Arena Tab**: Verified multi-model side-by-side cards (ETGNN, Matched GAT, RF, XGBoost) and interactive feature sliders.
5. **Subgraph Visualizer Tab**: Verified interactive Canvas rendering with center target node, color-coded neighbor nodes, directed payment edges, and zoom/pan controls.
6. **Benchmark Comparison Tab**: Verified test AUPRC bar charts and 11-model comparison table.
7. **C1/C2 Evidence Hub Tab**: Verified C2 Forest Plot across 8 registers with 95% bootstrap confidence intervals.
8. **Methodology & Limitations Tab**: Verified scientific safety disclaimers and academic integrity disclosures.
