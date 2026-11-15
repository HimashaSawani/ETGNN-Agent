# Agent Workflow Specification

## 1. Investigation Lifecycle

Every transaction analysis passes through a strictly sequenced lifecycle:

1. **Request Ingestion**: The client supplies `transaction_id`, optional user `question`, `evidence_mode` (`RETROSPECTIVE_INVESTIGATION` or `HISTORICAL_EVALUATION`), and optional `as_of_snapshot`.
2. **Deterministic Preflight Verification**:
   - Transaction ID existence is confirmed.
   - Snapshot cutoff compliance is validated.
   - Nonexistent transactions or future-dated records cause graceful abstention with typed diagnostic codes (`TRANSACTION_NOT_FOUND`, `TEMPORAL_LEAKAGE_BLOCKED`).
3. **Adaptive Tool Planning**:
   - The planning engine determines whether structural explanation (GNNExplainer) is required.
   - For isolated nodes (zero incoming edges), structural explanation is bypassed or accompanied by an explicit degree disclaimer.
   - Predictions are planned across three model paradigms: Temporal GNN (ETGNN), structural GNN (Matched GAT), and tabular baselines (Random Forest, XGBoost).
4. **Deterministic Tool Execution**:
   - Phase 2 tools execute in isolation.
   - Every tool result produces an immutable `EvidenceRecord` with SHA-256 integrity checksums.
   - The unique `evidence_id` is appended to the state's `provenance_chain`.
5. **Cross-Model Evidence Synthesis**:
   - Compares GNN probability against Random Forest and XGBoost probabilities.
   - Divergence ($\ge 0.30$ spread) triggers specialized comparative analysis highlighting topological vs. tabular discrepancies.
6. **Structured Explanation Drafting**:
   - The LLM creates an analytical draft adhering to the `ForensicInvestigationReport` schema.
   - All factual assertions must reference a valid `EV-...` evidence identifier.
7. **Two-Tier Grounding Audit**:
   - **Tier 1 (Factual / Numerical Audit)**: Tests numerical tolerances, forbids hallucinated feature names, and bans definitive guilt assertions.
   - **Tier 2 (Provenance & Domain Audit)**: Validates citation existence against the provenance chain, enforces Snapshot 43 context, and degrades unverified assertions.
8. **Final Report Formulation & Telemetry Archival**:
   - The validated report is cached for instant retrieval.
   - The complete event trace is recorded in the observable circular event buffer.

---

## 2. Planning Heuristics

The investigation planner generates an execution plan with the following deterministic priority order:
1. `get_prediction`: Requests model inference for `etgnn`, `random_forest`, and `xgboost`.
2. `get_transaction_neighbors`: Requests 2-hop ego-net expansion (capped at 100 nodes).
3. `get_explanation`: If the target node has incoming edges and risk is non-trivial, requests GNNExplainer attribution.
4. `get_evaluation_metrics`: Retrieves held-out test benchmark metrics for model baseline context.
5. `search_research`: Retrieves academic literature context on Elliptic darknet shifts and GNN graph structure.

---

## 3. Error Handling and Degradation Protocols

| Failure Mode | Agent Response | State Action |
| :--- | :--- | :--- |
| **Transaction not in dataset** | Immediate abstention | Set `status=ABSTAINED`, record `abstention_reason`, skip tools. |
| **Point-in-time boundary breach** | Immediate abstention | Block temporal leakage, record audit warning, transition to `ABSTAINED`. |
| **Prompt injection detected** | Sanitize & warn | Strip malicious syntax, log warning, proceed with sanitized prompt. |
| **Tool execution timeout** | Graceful fallback | Mark tool as failed in `tool_history`, continue with available evidence. |
| **Uncited factual claim** | Claim degradation | Move finding to `uncertain_findings`, remove from `supported_findings`. |
| **Numerical drift > 5%** | Re-evaluation / drop | Flag claim as `UNSUPPORTED`, drop or regenerate. |
| **LLM Provider API down** | Provider fallback | Automatically switch to offline `MockLLMProvider`, log warning. |

