# ETGNN-Agent — Full Regression Verification Report

**Repository:** `D:\ETGNN-Fraud-Detection`  
**Execution Timestamp:** 2026-11-11  
**Auditor:** QA Automation Specialist & Verification Engineer  

---

### 1. Test Execution Summary

The entire repository test suite was executed across unit, integration, security, and benchmark suites:

| Suite Name | Test File | Items | Passed | Failed | Execution Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 5 Evaluation & Drift** | `tests/test_phase5_evaluation_and_drift.py` | 20 | 20 | 0 | 14.2s | **PASSED** |
| **Phase 6 Application Security** | `tests/test_phase6_security.py` | 7 | 7 | 0 | 3.3s | **PASSED** |
| **Phase 6 Adversarial Stress Suite** | `src/agent/evaluation/adversarial_evaluation_suite.py` | 16 | 16 | 0 | 12.1s | **PASSED** |
| **Frontend TypeScript Build** | `web/frontend/` (`tsc -b && vite build`) | 1,957 modules | 1,957 | 0 | 6.5s | **PASSED** |
| **Baseline Checkpoint Manifest** | `artifacts/baseline_manifest.json` | 11 checkpoints | 11 | 0 | 0.8s | **VERIFIED** |
| **Raw Dataset Manifest** | `artifacts/baseline_manifest.json` | 4 raw datasets | 4 | 0 | 1.1s | **VERIFIED** |

**Total Verification Items:** 58 tests / checks  
**Overall Success Rate:** **100.00%** (58 / 58)  
**Regressions Detected:** **0**  

---

### 2. Verified Test Case Inventory

#### `tests/test_phase5_evaluation_and_drift.py` (20 Tests)
1. `test_01_evaluation_dataset_structure`: Validates 6 core task schemas, difficulty tiers, and query constraints.
2. `test_02_citation_metric_precision`: Verifies citation validator detects format errors and unprovenanced IDs.
3. `test_03_unsupported_claim_detection`: Verifies claim validator detects ungrounded numerical claims.
4. `test_04_retrieval_metrics_mrr_and_recall`: Tests hybrid retrieval rank reciprocal and recall calculations.
5. `test_05_temporal_leakage_rejection`: Verifies point-in-time snapshot filters block future transaction leakage.
6. `test_06_unknown_labels_preserved`: Validates that unlabelled transactions (`label = -1`) are never imputed.
7. `test_07_snapshot_43_caveat`: Verifies required scientific caveat on Snapshot 43 structural shifts.
8. `test_08_feature_drift_metrics`: Validates Kolmogorov-Smirnov 2-sample feature drift computation.
9. `test_09_prediction_drift`: Validates prediction probability distribution shift calculations.
10. `test_10_graph_drift_calculation`: Verifies mean in-degree and zero-in-degree census algorithms.
11. `test_11_label_drift_prevalence`: Verifies illicit transaction class prevalence drift tracking.
12. `test_12_insufficient_sample_auprc`: Verifies small sample warning on Snapshot 43 illicit class ($N=24$).
13. `test_13_candidate_isolation_and_safety`: Verifies candidate checkpoints remain isolated in candidate dir.
14. `test_14_model_promotion_rules`: Validates strict $+0.015$ AUPRC improvement promotion gate threshold.
15. `test_15_feedback_evaluator_governance`: Tests analyst feedback separation from ground truth training labels.
16. `test_16_provider_failure_handling_metric`: Validates graceful fallback metric under simulated provider fault.
17. `test_17_semantic_citation_benchmark_reproducibility`: Verifies benchmark JSON structure reproducibility.
18. `test_18_report_serialization`: Tests Pydantic JSON serialization across all report dataclasses.
19. `test_19_dashboard_api_contracts`: Validates responses for all 4 `/api/monitoring/*` endpoints.
20. `test_20_baseline_manifest_integrity`: Validates exact SHA-256 baseline checkpoint hashes against manifest.

#### `tests/test_phase6_security.py` (7 Tests)
1. `test_sql_injection_defense_in_feedback`: Confirms SQLite parameterized binding neutralizes DDL injection.
2. `test_predict_request_transaction_id_bounds`: Confirms rejection of negative IDs and 64-bit overflows.
3. `test_predict_request_feature_override_dimension_and_finite_checks`: Confirms rejection of $>165$ dims and `NaN`/`Inf`.
4. `test_path_traversal_rejection_in_pdf_export`: Confirms rejection of `../`, `%2e%2e`, and slash characters.
5. `test_feedback_reviewer_id_sanitization`: Confirms rejection of shell injection strings in `reviewer_id`.
6. `test_feedback_category_whitelist`: Confirms rejection of non-whitelisted feedback categories.
7. `test_spa_path_traversal_protection`: Confirms directory escape prevention on static SPA asset handlers.

