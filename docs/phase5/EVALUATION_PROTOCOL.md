# ETGNN-Agent — Phase 5 Scientific Evaluation Protocol

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 5F — Scientific Evaluation Protocol  
**Implementation:** `src/agent/evaluation/phase5_evaluation.py` & `scripts/run_phase5_evaluation.py`

---

## 1. Comparative Systems Under Evaluation

To rigorously quantify the contribution of multi-step agentic tool orchestration, hybrid Graph-RAG, and deterministic post-generation claim validation, the Phase 5 evaluation protocol benchmarks four distinct system configurations across identical Elliptic Bitcoin investigation tasks:

1. **Baseline A (`BASELINE_A`) — Deterministic Evidence-Only System (`EvidenceOnlyProvider`)**:
   - Executes deterministic tools and populates structured tables without generative LLM reasoning or hypothesis cross-examination (`generation_mode = "DETERMINISTIC_EVIDENCE_ONLY"`).
2. **Baseline B (`BASELINE_B`) — Single-Pass LLM Without RAG**:
   - Single-pass synthesis conditioned only on raw transaction attributes and model probabilities, without Graph-RAG research retrieval, multi-step tool cross-checking, or post-generation `ClaimValidator` / `ReportValidator` gates.
3. **Baseline C (`BASELINE_C`) — Single-Pass LLM + RAG (Without Validation Gate)**:
   - Incorporates Graph-RAG retrieved research chunks into single-pass generation, but omits 4-model GAT/GNNExplainer cross-checking and omits the post-generation `ClaimValidator` / `ReportValidator` verification and degradation gates.
4. **Proposed System D (`SYSTEM_D`) — Agentic AI + Hybrid Graph-RAG + Evidence Validation**:
   - Full 11-stage official LangGraph state machine executing 4 fraud models (`ETGNN`, `Matched GAT`, `Random Forest`, `XGBoost`), 2-hop subgraph analysis, `GNNExplainer`, metadata-reranked Hybrid Graph-RAG with `content_hash` provenance, structured hypothesis evaluation (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `NOT_TESTABLE`), and triple-gate validation (`CitationValidator`, `ClaimValidator`, `ReportValidator`).

---

## 2. Benchmark Transaction Cohort (`6` Canonical Tasks)

| Task ID | Transaction ID | Snapshot | Split | Mode | `as_of_snapshot` | Evaluation Objective |
|---|---|---|---|---|---|---|
| `T1_CONNECTED_ILLICIT` | `288717023` | `26` | Train (`1–32`) | `RETROSPECTIVE_INVESTIGATION` | `None` | Multi-model consensus, 2-hop topology, and GNNExplainer fidelity |
| `T2_HIGH_RISK_ILLICIT` | `232629023` | `32` | Train (`1–32`) | `RETROSPECTIVE_INVESTIGATION` | `None` | 4-model illicit risk consensus (`ETGNN`, `Matched GAT`, `RF`, `XGBoost`) |
| `T3_LICIT_BASELINE` | `230425980` | `1` | Train (`1–32`) | `RETROSPECTIVE_INVESTIGATION` | `None` | Licit baseline calibration and degree verification |
| `T4_SNAPSHOT_43_SHIFT` | `157288764` | `43` | Test (`41–49`) | `HISTORICAL_EVALUATION` | `43` | Snapshot 43 structural shift qualification & test-label masking |
| `T5_SNAPSHOT_43_ISOLATED` | `157288770` | `43` | Test (`41–49`) | `HISTORICAL_EVALUATION` | `43` | Zero-incoming-edge topological qualification & model divergence |
| `T6_TEMPORAL_FUTURE_LEAKAGE_GUARD` | `157288764` | `43` | Test (`41–49`) | `HISTORICAL_EVALUATION` | `32` | Point-in-time future transaction blocking (`tx_step 43 > cutoff 32`) |

---

## 3. Evaluation Metrics Definition

1. **Factual Correctness (`[0, 1]`)**: Fraction of generated analytical claims that pass exact degree checks, numerical probability tolerances (`±0.01`), and epistemic guardrails.
2. **Citation Validity (`[0, 1]`)**: Fraction of cited `evidence_id` references that exist in the immutable `provenance_chain` and match the active `target_tx_id`.
3. **Citation Coverage (`[0, 1]`)**: Proportion of the 6 core evidence modalities (`TX`, `PRED-ETGNN`, `PRED-GAT`, `PRED-RF`, `GRAPH`, `EXPL`) cited with verified provenance.
4. **Unsupported Claim Rate (`[0, 1]`)**: Fraction of claims flagged as `UNSUPPORTED` by `ClaimValidator`.
5. **Retrieval Precision & Recall (`[0, 1]`)**: Fraction of retrieved Graph-RAG chunks containing verified domain evidence (`Precision`) and fraction of task-required methodological concepts retrieved (`Recall`).
6. **Temporal Leakage Prevention (`[0, 1]`)**: Rate at which future transactions (`snapshot > as_of_snapshot`) are blocked and held-out test labels (`snapshots 41–49`) are masked in `HISTORICAL_EVALUATION` mode.
7. **Tool Execution Accuracy (`[0, 1]`)**: Fraction of planned deterministic tools executed without error.
8. **Investigation Task Completion (`[0, 1]`)**: Fraction of tasks reaching terminal `COMPLETED` (or expected `ABSTAINED` on future-leakage probes).
9. **Model Disagreement Handling (`[0, 1]`)**: Completeness of 4-model probability spread calculation and structured hypothesis classification.
10. **Latency, Token Usage, Cost & Failure Rate**: Measured wall-clock latency (`average_latency_ms`, `p95_latency_ms`), `average_total_tokens`, `estimated_cost_usd`, and runtime `failure_rate`.

