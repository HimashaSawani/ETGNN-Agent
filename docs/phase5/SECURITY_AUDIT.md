# ETGNN-Agent — Phase 5 Security, Reliability & Governance Audit

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 5G — Security & Reliability Verification  
**Status:** AUDITED & VERIFIED

---

## 1. Security & Governance Controls Implemented

### 1.1 Strict Request Validation & Tool Allowlisting
- All FastAPI investigation endpoints (`/agent/investigate`, `/agent/chat`, `/agent/feedback`) enforce Pydantic v2 schema validation with bounded numeric and string constraints.
- Tool execution in `execute_tools_node` (`src/agent/orchestration/nodes.py`) is strictly allowlisted to the 6 deterministic methods on `InvestigationToolRegistry` (`get_transaction`, `get_transaction_neighbors`, `get_prediction`, `get_explanation`, `get_model_metadata`, `get_evaluation_metrics`) plus `HybridRetriever.retrieve`. Arbitrary shell execution, dynamic `eval`, or filesystem write tools do not exist.

### 1.2 Prompt Injection Defenses (Input & Follow-Up Chat)
- User questions in `/agent/investigate` and `/agent/chat` are scanned against adversarial override patterns (`ignore previous instructions`, `override system prompt`, `drop table`, `override guardrails`, `<script>`).
- Matches are neutralized and logged as explicit security warnings in the investigation state and telemetry stream without altering deterministic tool arguments or bypassing `ClaimValidator` / `ReportValidator`.

### 1.3 Retrieval Evidence Isolation (Untrusted Document Sanitization)
- **Phase 5G Control (`src/agent/rag/retriever.py`)**: Retrieved documents are treated strictly as untrusted data, never executable instructions.
- `sanitize_retrieved_text()` scans every retrieved chunk for instruction-injection directives (`ignore all previous instructions`, `override system prompt`, `execute tool:`, `<script>`) and replaces them with `[REDACTED_UNTRUSTED_DIRECTIVE]` while appending a provenance warning.

### 1.4 Resource Budgets & Bounded Orchestration
- `ExecutionBudgetTracker` (`src/agent/orchestration/execution_limits.py`) enforces hard caps:
  - `MAX_TOOL_CALLS = 10`
  - `MAX_HOPS = 2` (`ResourceLimitExceededError` raised if `hops not in (1, 2)`)
  - `MAX_SUBGRAPH_NODES = 100` (`ResourceLimitExceededError` raised if `max_nodes > 100`)
  - `MAX_RAG_CHUNKS = 5`
  - `MAX_REGENERATION_ATTEMPTS = 1` (prevents infinite LangGraph loops)

### 1.5 Secret Redaction & Provider Transparency
- `AgentInvestigationEvent` (`src/agent/telemetry/events.py`) strips raw prompts, API keys (`api_key`, `token`), and internal stack traces from `safe_input_summary` and `safe_output_summary`.
- Live LLM failures (`GeminiProvider`, `AnthropicProvider`, `OllamaProvider`) raise typed exceptions (`LLMAuthenticationError`, `LLMRateLimitError`, `LLMTimeoutError`) and **never** silently fall back to `MockLLMProvider` (`generation_mode = "SIMULATED"`). When `allow_evidence_only_fallback=True`, failures degrade transparently to `EvidenceOnlyProvider` (`generation_mode = "DETERMINISTIC_EVIDENCE_ONLY"`).

### 1.6 Human-in-the-Loop Feedback Isolation
- Analyst feedback submitted via `/agent/feedback` is stored exclusively in an isolated SQLite table (`data/feedback/analyst_feedback.db`, `src/agent/feedback/store.py`).
- Automated SHA-256 verification confirms that submitting feedback never modifies `results/checkpoints/etgnn.pt`, `results/checkpoints/random_forest.joblib`, `results/checkpoints/xgboost.json`, or `results/checkpoints/matched_gat.pt`.

### 1.7 Deterministic SHA-256 Export Verification
- JSON and ReportLab PDF exports (`/agent/investigations/{id}/pdf`, `src/agent/reports/pdf_exporter.py`) compute and embed a deterministic SHA-256 audit hash (`provenance_hash`) over `investigation_id`, `target_transaction_id`, `consensus_verdict`, and `provenance_chain`.

