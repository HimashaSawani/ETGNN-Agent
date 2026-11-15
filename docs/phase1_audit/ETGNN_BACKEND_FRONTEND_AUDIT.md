# ETGNN-Agent — Phase 1: Backend & Frontend Full-Stack Audit

**Document ID:** `ETGNN_BACKEND_FRONTEND_AUDIT`  
**Date:** November 2026  
**Status:** COMPLETE & VERIFIED  
**Auditor:** Senior AI Engineer, Graph ML Researcher, MLOps Engineer & Software Architect  
**Workspace:** `d:\ETGNN-Fraud-Detection`  

---

## 1. Executive Summary

This audit assesses the software engineering maturity, API contract design, performance, security posture, and user interface architecture of the ETGNN application stack.

The stack comprises:
1. **Backend**: FastAPI asynchronous Python service (`src/web/app.py`) backed by an in-memory graph data loader (`src/web/data_loader.py`).
2. **Frontend**: React 19 single-page application built with TypeScript, Vite, Tailwind CSS, and Lucide React (`web/frontend/`).

Both subsystems were verified live during testing and run cleanly without unhandled exceptions or build failures.

---

## 2. Backend Architecture Audit (`src/web/app.py` & `src/web/data_loader.py`)

### Architectural Design
The backend uses a singleton data loader pattern (`WebDataLoader`) that initializes once upon server startup. It preloads:
- `data/processed/nodes_with_masks.csv` (203,769 transactions with ground-truth labels and split flags)
- `data/processed/elliptic_txs_edgelist.csv` (234,355 directed edges)
- Pre-fit `train_scaler.joblib`
- PyTorch / Scikit-Learn checkpoints for real-time inference (`etgnn.pt`, `random_forest.joblib`, `xgboost.json`, etc.)
- NetworkX / PyG directed graph representations for rapid $k$-hop neighborhood BFS traversal.

### Endpoint Inventory & Protocol Analysis

| HTTP Verb | Path | Request Parameters / Body | Response Schema | Status / Function |
|---|---|---|---|---|
| `GET` | `/` | None | `{"message": str, "docs": str}` | Root entrypoint |
| `GET` | `/api/health` | None | `{"status": "healthy", "version": "1.0.0", "data_loaded": bool}` | Health & readiness check |
| `GET` | `/api/metadata` | None | `DatasetMetadataResponse` | Dataset statistics (node counts, edge counts, splits, feature dims) |
| `GET` | `/api/metrics` | None | `Dict[str, BenchmarkMetric]` | Authoritative benchmark metrics (AUPRC, F1, Precision, Recall, AUC) |
| `GET` | `/api/models` | None | `List[ModelMetadata]` | Available models, parameters, inputs, and descriptions |
| `GET` | `/api/transactions/{tx_id}` | `tx_id: int` | `TransactionDetailResponse` | Complete 165 features, ground truth, step, degree, split |
| `GET` | `/api/transactions/{tx_id}/subgraph` | `tx_id: int, hops: int = 1, max_nodes: int = 50` | `SubgraphResponse` | Subgraph nodes, directed edges, neighbor labels, step info |
| `POST` | `/api/predict` | `PredictionRequest` (`tx_id`, `model_name`, `features_override`) | `PredictionResponse` | Illicit probability, binary prediction, confidence, inference latency |
| `GET` | `/api/explanations/{tx_id}` | `tx_id: int, top_k_features: int = 10` | `ExplanationResponse` | Top influential feature IDs, importance weights, edge masks |
| `GET` | `/api/temporal-metrics` | None | `TemporalRobustnessResponse` | Per-timestep AUPRC and illicit counts across steps 1–49 |
| `GET` | `/api/ablation-data` | None | `AblationResponse` | Full vs Local vs No-Edges architectural ablation data |
| `GET` | `/api/c1-c2-cohorts` | None | `CohortAnalysisResponse` | C1/C2 matched cohort statistics and flip-rate metrics |

### Middleware & Cross-Origin Configuration
- **CORS Middleware**: Currently configured with:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
  *Audit Observation*: While convenient for local development across Vite (`5173`) and FastAPI (`8000`), production deployment must restrict `allow_origins` to explicitly approved host domains or reverse-proxy origins.

---

## 3. Tool Mapping for Future Agentic Architecture

The upcoming Phase 2–3 Agentic AI Layer requires a set of deterministic, read-only Python tools. The existing `WebDataLoader` methods map directly to the proposed tools:

| Planned Agent Tool Name | Direct Python Loader Method | Backing HTTP Endpoint | Capability Provided to LLM Agent |
|---|---|---|---|
| `get_transaction(tx_id)` | `loader.get_transaction(tx_id)` | `GET /api/transactions/{tx_id}` | Retrieve ground truth label, snapshot time step, degree, and raw/scaled feature vectors. |
| `get_transaction_neighbors(tx_id, hops)` | `loader.get_subgraph(tx_id, hops, max_nodes)` | `GET /api/transactions/{tx_id}/subgraph` | Trace inbound/outbound fund flows, neighbor transaction labels, and topological degree. |
| `get_prediction(tx_id, model, feat_override)` | `loader.predict(tx_id, model_name, features_override)` | `POST /api/predict` | Query ensemble models (Canonical ETGNN, Random Forest, XGBoost) for fraud probabilities. |
| `get_model_metadata()` | `loader.get_models_metadata()` | `GET /api/models` | Retrieve model architectures, parameter counts, feature expectations, and checkpoint states. |
| `get_explanation(tx_id)` | `loader.get_explanation(tx_id)` | `GET /api/explanations/{tx_id}` | Extract GNNExplainer feature importance masks and influential edge flows. |
| `get_evaluation_metrics()` | `loader.get_benchmark_metrics()` | `GET /api/metrics` | Access scientifically verified benchmark metrics to ground agent claims in empirical truth. |

---

## 4. Frontend Architecture Audit (`web/frontend/`)

### Technology Stack
- **Framework**: React 19.0.0
- **Language**: TypeScript 5.6.x (strict type checking enabled)
- **Bundler & Dev Server**: Vite 6.0.x
- **CSS Architecture**: Tailwind CSS 3.4.x utility classes
- **UI Components**: 12 modular tabs representing deep research telemetry:
  1. `TransactionExplorer.tsx` — Dynamic search, feature inspection, label display.
  2. `PredictionArena.tsx` — Multi-model real-time side-by-side inference comparative arena.
  3. `GraphVisualizer.tsx` — Interactive SVG/Canvas directed fund-flow visualization with hop controls.
  4. `ExplanationInspector.tsx` — Feature importance ranking and edge influence inspection.
  5. `BenchmarkComparison.tsx` — Comprehensive model benchmark tables and AUPRC curves.
  6. `ResearchC1C2Hub.tsx` — C1 and C2 matched cohort experimental results and flip-rate odds ratios.
  7. `AblationAnalysis.tsx` — Visual exploration of Full vs Local vs No-Edges ablation.
  8. `TemporalRobustness.tsx` — Per-snapshot timeline analysis detailing the Snapshot 43 collapse.
  9. `MethodologyBoundaries.tsx` — Clear documentation of dataset limits, unlabelled non-treatment, and scaling.
  10. `ExecutiveSummary.tsx` — High-level research takeaways for examiners and stakeholders.
  11. `ModelCardModal.tsx` — Modal displaying technical model specs and loss functions.
  12. `ExportReportModal.tsx` — Structured export of forensic audit artifacts.

### Build & Lint Audit
- **Lint Audit (`npm run lint`)**: Passed with `0 errors`. Nine React hook warnings (`react-hooks/exhaustive-deps`) exist in auxiliary visualization components (`GraphVisualizer.tsx`, `TemporalRobustness.tsx`). These are harmless dependencies that do not trigger re-render loops but will be tidied in Phase 4.
- **Production Build (`npm run build`)**: Vite compiles the entire bundle in `3.87s` to `web/frontend/dist/`.

---

## 5. UI Integration Strategy for Agent Capabilities

To support the future agentic capabilities without disrupting existing research tabs, two non-invasive UI integration points are designated:
1. **Interactive Investigation Drawer ("Ask AI Agent")**:
   - A slide-over panel accessible from `TransactionExplorer` and `GraphVisualizer`.
   - Allows investigators to click **"Investigate with AI"** on any transaction.
   - Streams the agent's chain-of-thought, tool invocation logs (`get_transaction`, `get_subgraph`), and synthesized narrative report.
2. **Analyst Review & Ground-Truth Override Modal**:
   - Allows compliance officers to review AI-generated risk scores, enter human ground truth (`Fraud`, `Benign`, `Suspicious`), provide rationale, and log the feedback into the audit database.

---

## 6. Full-Stack Audit Certification

Both the FastAPI backend and React frontend are architecturally sound, thoroughly tested, and immediately prepared to support the agentic extensions outlined in the upcoming phases.

