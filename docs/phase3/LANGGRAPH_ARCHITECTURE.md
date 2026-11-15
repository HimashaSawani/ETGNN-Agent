# LangGraph Orchestration Architecture

## 1. Overview & Motivation

Autonomous fraud investigation in decentralized transaction graphs requires structured, auditable reasoning rather than unconstrained LLM loops. The `ETGNN-Agent` state graph enforces a deterministic execution DAG with conditional edge transitions, bounded regeneration loops, and hard resource limits.

The architecture implements standard LangGraph semantics (`add_node`, `add_edge`, `add_conditional_edges`, `set_entry_point`, `compile`, `invoke`, `ainvoke`), allowing transparent execution both with lightweight in-memory compilation and full enterprise LangGraph orchestrators.

---

## 2. Central State Schema (`InvestigationState`)

The workflow communicates via a single typed state dictionary defined in `src/agent/orchestration/state.py`:

```python
class InvestigationState(TypedDict, total=False):
    investigation_id: str
    tx_id: int
    question: str
    evidence_mode: str              # RETROSPECTIVE_INVESTIGATION or HISTORICAL_EVALUATION
    as_of_snapshot: Optional[int]   # Point-in-time snapshot cutoff (1..49)
    status: str                     # InvestigationStatus enum value
    plan: Optional[Dict[str, Any]]
    tool_history: List[Dict[str, Any]]
    evidence_bundle: Dict[str, Any]
    provenance_chain: List[str]     # Ordered list of immutable Evidence IDs
    warnings: List[str]
    errors: List[str]
    raw_explanation: Optional[Dict[str, Any]]
    claim_validation_result: Optional[Dict[str, Any]]
    report_validation_result: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    retries_count: int
    execution_limits_exceeded: bool
    abstention_reason: Optional[str]
    start_time: float
    duration_ms: int
```

---

## 3. Node Specifications & State Transitions

### Node 1: `validate_request`
- **Inputs**: `tx_id`, `question`, `evidence_mode`, `as_of_snapshot`.
- **Logic**:
  - Validates `tx_id` against the Elliptic transaction index (`loader.tx_to_index`).
  - Evaluates point-in-time constraints: in `HISTORICAL_EVALUATION` mode, if `tx_snapshot > as_of_snapshot`, the investigation immediately transitions to `ABSTAINED`.
  - Sanitizes `question` against prompt injection attacks (regex detection of `ignore previous`, `system prompt`, `drop table`, `exfiltrate`).
- **Target Edge**: Conditional routing via `route_after_request_validation`:
  - `status == ABSTAINED` $\to$ `finalize_investigation`.
  - `status == VALIDATING_REQUEST` $\to$ `plan_investigation`.

### Node 2: `plan_investigation`
- **Inputs**: Verified transaction metadata, degree characteristics.
- **Logic**:
  - Determines required investigative steps: model predictions, subgraph expansion, GNNExplainer attribution, research literature retrieval, evaluation metrics.
  - Generates typed `ToolCallRequest` items with explicit rationales.
  - Caps planned tools to `MAX_TOOL_CALLS = 8`.
- **Target Edge**: Fixed edge $\to$ `execute_tools`.

### Node 3: `execute_tools`
- **Inputs**: Planned tool list.
- **Logic**:
  - Executes deterministic tools sequentially through `InvestigationToolRegistry` and `HybridRetriever`.
  - Enforces `ExecutionBudgetTracker` (max 8 tools, max 2-hop radius, max 100 subgraph nodes, max 4 RAG documents).
  - Populates `evidence_bundle` and appends verified IDs to `provenance_chain`.
- **Target Edge**: Fixed edge $\to$ `consolidate_evidence`.

### Node 4: `consolidate_evidence`
- **Inputs**: `evidence_bundle["predictions"]`.
- **Logic**:
  - Computes probability spread across models: $\Delta p = \max(p) - \min(p)$.
  - Flags model divergence if $\Delta p \ge 0.30$ (e.g. GNN predicts illicit while tabular models predict licit).
  - Emits `EVIDENCE_CONSOLIDATED` telemetry event.
- **Target Edge**: Fixed edge $\to$ `generate_explanation`.

### Node 5: `generate_explanation`
- **Inputs**: Consolidated evidence bundle, question, provenance chain.
- **Logic**:
  - Constructs system and user prompts injecting full point-in-time evidence and valid provenance IDs.
  - Queries active LLM provider (`MockLLMProvider`, `GeminiProvider`, etc.).
  - Extracts and parses JSON structure into initial report dictionary.
  - Tracks regeneration cycles via `retries_count`.
- **Target Edge**: Fixed edge $\to$ `validate_claims`.

### Node 6: `validate_claims`
- **Inputs**: Raw explanation findings, evidence bundle.
- **Logic**:
  - Executes `ClaimValidator`: tests for prohibited certainty phrases, hallucinated feature names, and numerical value consistency against source records.
- **Target Edge**: Conditional routing via `route_after_claim_validation`:
  - `not is_valid and retries_count < MAX_REGENERATION_ATTEMPTS` $\to$ `generate_explanation`.
  - `is_valid or retries_count >= MAX_REGENERATION_ATTEMPTS` $\to$ `validate_final_report`.

### Node 7: `validate_final_report`
- **Inputs**: Raw explanation, claim summary, snapshot metadata.
- **Logic**:
  - Executes `ReportValidator`: checks citation validity against provenance chain, verifies Snapshot 43 historical qualifications, inserts GNNExplainer interpretation disclaimers, and degrades unverified claims to `uncertain_findings`.
- **Target Edge**: Conditional routing via `route_after_report_validation`:
  - `should_regenerate and retries_count < MAX_REGENERATION_ATTEMPTS` $\to$ `generate_explanation`.
  - `otherwise` $\to$ `finalize_investigation`.

### Node 8: `finalize_investigation`
- **Inputs**: Final state dictionary.
- **Logic**:
  - Instantiates immutable Pydantic `ForensicInvestigationReport`.
  - Emits `INVESTIGATION_COMPLETED` or `INVESTIGATION_ABSTAINED` event.
  - Caches completed investigation in `InvestigationRunner`.
- **Target Edge**: Fixed edge $\to$ `END`.

---

## 4. Execution Limits and Loop Safety

To prevent infinite loops, catastrophic token expenditure, or runaway recursive tool calls, the engine enforces strict compile-time and runtime budgets:
- **Maximum Execution Steps**: 25 graph iterations maximum before hard termination.
- **Maximum Tool Invocations**: 8 deterministic tool calls per investigation.
- **Maximum Graph Hops**: 2 hops radius (`num_hops <= 2`).
- **Maximum Subgraph Nodes**: 100 nodes per extraction.
- **Maximum RAG Chunks**: 4 academic evidence passages.
- **Maximum Regeneration Attempts**: Exactly 1 retry attempt (`MAX_REGENERATION_ATTEMPTS = 1`).
- **Execution Timeout**: 60.0 seconds wall-clock limit.

