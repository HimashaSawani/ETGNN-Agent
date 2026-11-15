# ETGNN-Agent — Phase 5 Scientific Evaluation Results

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 5F — Scientific Evaluation Results  
**Artifact Source:** `results/benchmarks/phase5_evaluation_results.json`  
**Runner Script:** `scripts/run_phase5_evaluation.py`

---

## 1. Executive Summary of Comparative Benchmark Results

All four investigation architectures were evaluated across the 6 canonical Elliptic Bitcoin investigation tasks (`T1_CONNECTED_ILLICIT` through `T6_TEMPORAL_FUTURE_LEAKAGE_GUARD`) under the authoritative chronological split (**Train: Snapshots `1–32`, Validation: Snapshots `33–40`, Test: Snapshots `41–49`**).

### 1.1 Aggregate System Comparison Table (`6` Tasks per System, `24` Total Runs)

| Metric | Baseline A (Evidence-Only) | Baseline B (Single-Pass LLM No RAG) | Baseline C (LLM + RAG No Validation Gate) | Proposed System D (11-Stage Agentic AI + RAG + Validation) |
|---|---|---|---|---|
| **Factual Correctness** | `1.0000` | `0.5833` | `0.9583` | **`1.0000`** |
| **Citation Validity** | `1.0000` | `0.5833` | `1.0000` | **`1.0000`** |
| **Citation Coverage** | `0.8083` | `0.3333` | `0.7917` | **`1.0000`** |
| **Unsupported Claim Rate** | `0.0000` | `0.2500` | `0.0417` | **`0.0000`** |
| **Retrieval Precision** | `0.9250` | `0.0000` | `0.9583` | **`0.9583`** |
| **Retrieval Recall** | `0.6889` | `0.0000` | `0.7222` | **`0.7222`** |
| **Temporal Leakage Prevention** | `1.0000` | `1.0000` | `1.0000` | **`1.0000`** |
| **Tool Execution Accuracy** | `1.0000` | `0.5000` | `0.7500` | **`1.0000`** |
| **Task Completion Rate** | `1.0000` | `1.0000` | `1.0000` | **`1.0000`** |
| **Model Disagreement Handling** | `0.7917` | `0.4167` | `0.7917` | **`1.0000`** |
| **Average Latency (`ms`)** | `1296.63 ms` | `164.56 ms` | `147.62 ms` | **`1083.99 ms`** |
| **P95 Latency (`ms`)** | `2012.72 ms` | `228.79 ms` | `199.14 ms` | **`1983.57 ms`** |
| **Average Total Tokens** | `0.0` | `205.8` | `335.0` | **`813.2`** |
| **Estimated Cost (`USD`)** | `$0.000000` | `$0.000000` | `$0.000000` | **`$0.000000` (Offline)** |
| **Failure Rate** | `0.0000` | `0.0000` | `0.0000` | **`0.0000`** |

---

## 2. Key Scientific Findings

1. **Elimination of Hallucinations & Unsupported Claims (`System D: 0.0000` vs `Baseline B: 0.2500` vs `Baseline C: 0.0417`)**:
   - Without Graph-RAG or `ClaimValidator` (`Baseline B`), single-pass generation hallucinates semantic feature names (`wallet_balance`) and attributes Snapshot 43 to unverified darknet takedowns (`unsupported_claim_rate = 0.2500`).
   - Adding Graph-RAG (`Baseline C`) raises factual correctness from `0.5833` to `0.9583`, but without the `ClaimValidator` + `ReportValidator` gate, uncalibrated confidence claims and numerical rounding drift (`0.0417` unsupported claim rate) still occur.
   - **Proposed System D** achieves `1.0000` factual correctness, `1.0000` citation validity, `1.0000` citation coverage across all 6 evidence modalities (`TX`, `PRED-ETGNN`, `PRED-GAT`, `PRED-RF`, `GRAPH`, `EXPL`), and `0.0000` unsupported claim rate.
2. **4-Model Consensus & Hypothesis Reasoning (`System D: 1.0000` vs `Baseline A: 0.7917`)**:
   - While deterministic evidence-only extraction (`Baseline A`) is factually accurate (`1.0000`), it does not synthesize cross-model hypothesis evaluations (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `NOT_TESTABLE`). **System D** achieves `1.0000` on model disagreement handling by combining `ETGNN`, `Matched GAT`, `Random Forest`, and `XGBoost` with structured hypothesis classification.
3. **Strict Point-in-Time Leakage Prevention (`1.0000`)**:
   - On `T4_SNAPSHOT_43_SHIFT` and `T5_SNAPSHOT_43_ISOLATED` (`HISTORICAL_EVALUATION`, `as_of_snapshot = 43`), held-out test labels are strictly masked (`LabelStatus.RESTRICTED`, `raw_label_code = None`).
   - On `T6_TEMPORAL_FUTURE_LEAKAGE_GUARD` (`tx_id` at Snapshot 43 queried at `as_of_snapshot = 32`), the system immediately abstains (`status = "ABSTAINED"`, `TEMPORAL_LEAKAGE_BLOCKED`) in `35.43 ms`.

---

## 3. Live External LLM API Disclosure

- **Offline / Deterministic Benchmark (`MockLLMProvider` & `EvidenceOnlyProvider`)**: `EXECUTED_AND_VERIFIED` across all 24 benchmark runs.
- **External Cloud / Local Daemon Providers (`GeminiProvider`, `AnthropicProvider`, `OllamaProvider`)**: Reported honestly as `NOT_RUN` in the sandboxed benchmark environment because external API credentials (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) and a local Ollama daemon are not configured in the test environment. Provider error-handling tests (`tests/test_phase4_5_reliability_and_integrity.py`) independently verify that missing credentials raise `LLMAuthenticationError` and never silently fall back to simulated output.

