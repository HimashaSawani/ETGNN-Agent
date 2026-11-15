# Web Platform Implementation Plan: ETGNN Research Demonstration Platform

**Repository:** `ETGNN-Fraud-Detection`  
**System Name:** ETGNN Interactive Research Demonstration Platform  
**Target:** Professional Full-Stack Web Platform (FastAPI + React/TypeScript/Vite)  
**Date:** October 2026  
**Auditor / Lead Engineer:** Senior Full-Stack AI Engineer & PyG Specialist

---

## 1. Executive Architecture Overview

This document specifies the technical architecture, data flow, API design, frontend components, and quality assurance framework for the **ETGNN Web-Based Research Demonstration Platform**.

The platform is designed to make the completed empirical research accessible, interactive, and transparent for academic examiners, supervisors, peer reviewers, and financial compliance researchers. It directly connects to real trained checkpoints (`results/checkpoints/`), precomputed features (`data/processed/scaled_features.npy`), validation and test graph snapshots (`data/processed/`), and verified C1/C2 statistical tables (`results/tables/` and `results/main_c1_c2_results/`).

```
System Architecture Diagram:
+-----------------------------------------------------------------------------------------+
|                                    REACT + VITE FRONTEND                                |
|  - Tab 1: Research Overview & Executive Findings (RF Dominance, Topological Noise)       |
|  - Tab 2: Transaction Explorer (Paginated, Searchable Table of Bitcoin Transactions)   |
|  - Tab 3: Fraud Prediction Arena (Side-by-Side Model Comparison: RF, ETGNN, GAT, XGB)    |
|  - Tab 4: Interactive Graph Neighborhood Visualizer (Canvas/SVG Topology & Flow)       |
|  - Tab 5: GNNExplainer Attribution Inspector (Selected Edges, Random Baseline, Drop)   |
|  - Tab 6: Model Performance Benchmark (Official Held-Out Test Scores across 11 Models)  |
|  - Tab 7: C1 / C2 Research Results (Forest Plot, Bootstrap CIs, Register Disclosures)   |
|  - Tab 8: Topological Ablation Analysis (+0.2468 Jump without Edges, Prevalence Shift)  |
|  - Tab 9: Scientific Methodology, Safety & Epistemic Boundaries                        |
+-----------------------------------------------------------------------------------------+
                                           |  HTTP / REST JSON
                                           v
+-----------------------------------------------------------------------------------------+
|                                      FASTAPI BACKEND                                    |
|  - GET  /api/health                     - System health, artifact status, hardware      |
|  - GET  /api/models                     - Model registry & architecture metadata        |
|  - GET  /api/metrics                    - Official frozen held-out test split metrics   |
|  - GET  /api/transactions               - Paginated transactions with filter/search     |
|  - GET  /api/transactions/{id}          - Comprehensive transaction profile & scores    |
|  - GET  /api/transactions/{id}/subgraph - Extracted 1-hop / 2-hop computation subgraph   |
|  - POST /api/predict                    - Live model scoring across frozen checkpoints  |
|  - GET  /api/explanations/{id}          - Saved / computed GNNExplainer attribution     |
|  - GET  /api/research/c1                - C1 shared-intervention response statistics    |
|  - GET  /api/research/c2                - C2 8-register bootstrap paired differences    |
|  - GET  /api/research/ablations         - Local edge ablation & multi-seed results      |
+-----------------------------------------------------------------------------------------+
                                           |  In-Memory / Memory-Mapped I/O
                                           v
+-----------------------------------------------------------------------------------------+
|                                REPOSITORY ARTIFACT LAYER                                |
|  - Checkpoints: etgnn.pt, matched_gat.pt, random_forest.joblib, xgboost.json            |
|  - Graph Data: validation_graph.pt, test_graph.pt (234,355 directed edges)             |
|  - Feature Store: scaled_features.npy (203,769 x 165 memmap), node_mapping.csv           |
|  - Research Outputs: c1_response_summary.csv, c2_registers_bootstrap_summary.csv        |
|  - Explanations: detailed_intervention_records.csv (3,393 GNNExplainer records)         |
+-----------------------------------------------------------------------------------------+
```

---

## 2. API Endpoint Specification (Phase 2)

All API endpoints will be hosted at `/api/` using FastAPI with strict Pydantic v2 data models:

1. **`GET /api/health`**  
   Returns API status, version, device (CPU/CUDA), memory usage, and artifact verification status.
2. **`GET /api/models`**  
   Lists the 7 supported model architectures (Random Forest, XGBoost, GCN, GraphSAGE, Standard GAT, Matched GAT, ETGNN), their parameter shapes, input dimensions, and checkpoint hashes.
3. **`GET /api/metrics`**  
   Returns the official test set performance metrics from `results/tables/final_test_results.csv` (AP, Precision, Recall, F1, ROC-AUC) across all 11 evaluation conditions.
4. **`GET /api/transactions`**  
   Query parameters: `page`, `limit`, `time_step`, `split`, `label`, `search_id`. Returns paginated transaction summaries.
5. **`GET /api/transactions/{id}`**  
   Returns deep transaction details: transaction ID, time step, ground-truth label, chronological partition (train/val/test), local features summary, and precomputed prediction scores.
6. **`GET /api/transactions/{id}/subgraph`**  
   Extracts the local 1-hop or 2-hop computation subgraph around the target transaction, returning nodes, directed edges, candidate edges ($E_{cand}$), and neighbor labels for graph visualization.
7. **`POST /api/predict`**  
   Body: `{"transaction_id": int}`. Executes real PyTorch Geometric and scikit-learn forward passes across ETGNN, Matched GAT, Random Forest, and XGBoost, returning predicted illicit probabilities and classification labels at threshold $\tau = 0.50$.
8. **`GET /api/explanations/{id}`**  
   Returns GNNExplainer attribution:
   - Checks `detailed_intervention_records.csv` and `results/explanations/connected_{case}.json`.
   - If present in the frozen C1/C2 benchmark, returns exact deleted edge IDs, budget $k$, candidate edges $E_{cand}$, probability drop, and random comparator baseline across seeds 42, 43, 44.
   - If not in frozen records, generates on-the-fly demonstration explanation, clearly tagged as *"Demonstration Output (Non-Frozen)"*.
9. **`GET /api/research/c1`**  
   Returns C1 summary data from `results/tables/c1_response_summary.csv`: mean adjusted difference (+0.002903), sample SD, raw difference, and cohort breakdown.
10. **`GET /api/research/c2`**  
    Returns C2 bootstrap summary from `results/tables/c2_registers_bootstrap_summary.csv`: all 8 registers, matched pair counts, mean differences, 95% bootstrap CIs, and zero-crossing indicators.
11. **`GET /api/research/ablations`**  
    Returns local edge ablation metrics (+0.2468 jump without edges), multi-seed validation scores (seeds 7, 42, 123), and temporal prevalence volatility across test steps 41–49.

---

## 3. Frontend Component & Page Hierarchy (Phase 3)

The frontend will be built in `web/frontend/` using **React 18 + TypeScript + Vite + TailwindCSS / Lucide Icons**:

```
Frontend Navigation & Tab Views:
1. [Overview]
   - Executive Research Summary, Problem Statement, Key Findings Banner
   - Highlights: Random Forest Supremacy (0.6323 vs 0.2250), Topological Noise Jump (+0.2468)
2. [Transactions Explorer]
   - Searchable, filterable table of Elliptic transactions with time step, split, and label badges
   - Quick "Inspect" action routing to Prediction, Graph, and Explanation views
3. [Prediction Arena]
   - Live multi-model prediction dashboard for any transaction
   - Side-by-side gauge cards: ETGNN, Matched GAT, Random Forest, XGBoost
   - Confidence margin visualization (|p - 0.50|)
4. [Graph Visualizer]
   - Interactive node-link graph visualization of the transaction's neighborhood
   - Distinguishes target node, licit neighbors, illicit neighbors, and unknown entities
   - Highlights candidate edges ($E_{cand}$) and direction of payment flow
5. [Explanation Inspector]
   - Visualizes GNNExplainer edge selection on the computation subgraph
   - Bar chart contrasting explanation deletion drop vs 30-draw random deletion baseline
   - Displays exact deleted edge IDs, budget $k$, and seed provenance
6. [Benchmark Comparison]
   - Official held-out test results table with interactive metric sort and bar charts
   - Highlights the rejection of Hypothesis H1 (GNN superiority)
7. [C1 & C2 Research Hub]
   - Interactive Forest Plot of Contrast 2 showing all 8 registers with 95% bootstrap CIs
   - Highlights ETGNN False Positive fragility (+0.0393 drop difference)
   - Contrast 1 distribution histogram and sample size disclosures
8. [Ablation Analysis]
   - Interactive cards illustrating the local topological ablation (+0.2468 increase without edges)
   - Explains the GAT MLP-collapse and heterophily/noise mitigation mechanism
   - Prevalence drift chart across steps 41–49
9. [Methodology & Boundaries]
   - Explicit scientific disclosures: static time-encoded GAT, research prototype status
   - Epistemic limitations: edge deletion as sensitivity vs causal crime attribution
```

---

## 4. Performance & Scalability Strategy (Phase 8)

1. **Memory-Mapped Feature Access:** Features (`scaled_features.npy`, $203,769 \times 165$) are loaded via `numpy.memmap` in read-only mode (`mmap_mode="r"`). RAM overhead is $< 50\text{ MB}$.
2. **Pre-Loaded Model Registry:** Checkpoints are loaded into memory once on FastAPI startup and cached in a singleton state.
3. **Local Subgraph Extraction:** Graph visualization transfers only the 1-hop or 2-hop neighborhood ($< 20$ nodes on average), preventing multi-megabyte graph transfer to the client browser.
4. **Pure CPU Inference:** Inference executes in $< 15\text{ ms}$ on standard multi-core CPUs; GPU acceleration is supported if available, but not required.

---

## 5. Testing and Validation Plan (Phase 7)

1. **Backend Tests (`tests/test_web_api.py`):**
   - Startup and health check assert 200 OK.
   - Exact numerical parity: verify `/api/metrics` matches `final_test_results.csv`.
   - Exact C1/C2 parity: verify `/api/research/c2` matches `c2_registers_bootstrap_summary.csv`.
   - Prediction flow: assert real forward pass on test node produces finite probabilities in $[0, 1]$.
   - 404 error handling for non-existent transaction IDs.
2. **Frontend Validation:**
   - Production Vite build compilation (`npm run build`).
   - Zero TypeScript compilation errors.
   - Headless HTTP API integration validation.

---

## 6. Implementation Milestones

- **Milestone 1:** Create implementation plan and platform documentation structure (`docs/`).
- **Milestone 2:** Implement FastAPI backend in `src/web/` with all 11 endpoints, data caching, and inference logic.
- **Milestone 3:** Create automated pytest test suite in `tests/test_web_api.py` and verify all tests pass.
- **Milestone 4:** Scaffold and implement modern React/TypeScript/Vite frontend in `web/frontend/`.
- **Milestone 5:** Compile frontend, verify end-to-end integration, and generate comprehensive architectural documentation.
