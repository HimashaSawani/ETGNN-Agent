# ETGNN-Agent — Phase 4: Application Security & Data Integrity Review
## Threat Modeling, Input Validation, Epistemic Safety & Sanitization

**Auditor:** Application Security Engineer / Research Integrity Auditor  
**Scope:** Phase 4 React Frontend, FastAPI Endpoints, SQLite Feedback Storage, PDF Exporter  
**Verdict:** APPROVED (Zero Critical Vulnerabilities, Checkpoints Fully Isolated)  

---

## 1. Security Threat Analysis

| Threat Vector | Mitigation Strategy | Verification Result |
| :--- | :--- | :--- |
| **SQL Injection** | Parameterized queries (`cursor.execute(sql, (params,))`) across all feedback store methods in `src/agent/feedback/store.py`. | **VERIFIED**: Zero raw string interpolation into SQL queries. |
| **Prompt Injection & Jailbreaking** | User input in follow-up chat is strictly wrapped into structured schemas. Responses are validated against factual evidence bundles; hallucinated facts without evidence citations are dropped. | **VERIFIED**: System prompts enforce evidence grounding and reject out-of-domain instruction overrides. |
| **Cross-Site Scripting (XSS)** | React JSX automatically escapes dynamic string values. Markdown/text fields are rendered without `dangerouslySetInnerHTML`. | **VERIFIED**: Zero usage of unescaped HTML injection. |
| **Checkpoint Mutation Vulnerability** | Feedback storage resides in `data/analyst_feedback.db`. Write permissions to `checkpoints/` and `data/raw/` are non-existent in the feedback store. | **VERIFIED**: Model weights and raw CSV files remain byte-identical. |
| **DoS via PDF Generator** | Bounded tables and paragraphs in ReportLab; max finding counts capped; canvas build executes synchronously with timeouts. | **VERIFIED**: Generates sub-second forensic PDFs without memory exhaustion. |
| **Information Disclosure** | Live API keys (e.g., `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) are accessed strictly via environment variables; never logged or serialized to frontend JSON payloads. | **VERIFIED**: UI displays provider type without revealing credentials. |

---

## 2. Invariant Verification

1. `data/raw/elliptic_txs_classes.csv`: Unmodified.
2. `data/raw/elliptic_txs_features.csv`: Unmodified.
3. `data/raw/elliptic_txs_edgelist.csv`: Unmodified.
4. `experiments/`: Experiment logs and baseline evaluation metrics preserved.
5. `data/analyst_feedback.db`: Thread-safe SQLite store with explicit foreign key indexing and WAL mode.

