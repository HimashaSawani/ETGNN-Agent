# Agent Security & Confidentiality Review

## 1. Threat Model & Security Boundaries

The `ETGNN-Agent` operates as an analytical co-pilot processing untrusted user queries while interacting with sensitive financial models and graph topologies.

```
+-------------------------------------------------------+
|                   Client Perimeter                    |
|   Untrusted User Query / Prompt Injection Payloads   |
+--------------------------+----------------------------+
                           |
                           v
+-------------------------------------------------------+
|                   Security Perimeter                  |
|  - Sanitization & Adversarial Intent Scanning        |
|  - Zero Chain-of-Thought / Secret Leakage             |
|  - Point-in-Time Temporal Sandbox                     |
+--------------------------+----------------------------+
                           |
                           v
+-------------------------------------------------------+
|                   Execution Perimeter                 |
|  - Read-Only Checkpoints (SHA-256 Verified)           |
|  - Bounded Memory Circular Event Buffer               |
+-------------------------------------------------------+
```

---

## 2. Prompt Injection Defenses

User questions submitted to `/api/agent/investigate` are sanitized in `validate_request_node`:
- Regular expression patterns detect prompt injection idioms:
  - `(?i)(ignore\s+(all\s+)?previous\s+instructions)`
  - `(?i)(system\s+prompt|drop\s+table|delete\s+from|base64)`
  - `(?i)(disregard\s+guidelines|you\s+are\s+now|override\s+policy)`
- When detected:
  - The malicious prompt injection is neutralised.
  - A permanent audit warning is recorded in the report: `"Adversarial prompt injection pattern detected in query and neutralized."`
  - The model prompt uses pre-sanitized template values with strict system-instruction delimiters.

---

## 3. Confidentiality & Zero-Leakage Policy

Client-facing telemetry (`ObservableEventSink`) and API responses enforce strict redaction:
- **Zero Chain-of-Thought Leakage**: Internal scratchpad reasoning, hidden thinking traces, and raw prompt chains are never exposed via HTTP or WebSocket events. Only structured, typed summaries (`safe_input_summary`, `safe_output_summary`) are emitted.
- **Zero Secret Exposure**: Provider API keys, local filesystem paths, and internal server credentials are excluded from state serialization and error payloads.
- **Path Traversal Protection**: The FastAPI static server strictly rejects directory traversal tokens (`..`, `%2e%2e`).

---

## 4. Point-in-Time Temporal Boundary Enforcement

Temporal data leakage represents a critical integrity vulnerability in financial graph machine learning:
- When running in `HISTORICAL_EVALUATION` mode with `as_of_snapshot = T`, the agent enforces that no tool may retrieve transaction labels, features, or edges dated $> T$.
- `validate_request_node` rejects forward-looking transactions upfront.
- `InvestigationToolRegistry` raises `TemporalBoundaryViolationError` if any neighbor expansion accesses future time-slices.
- All baseline models remain pinned to their historical training cutoff (Snapshots 1–32).

