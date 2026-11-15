# Phase 1.5: Agent Transparency & Evidence-Trace Specification

**Document ID:** `AGENT_TRANSPARENCY_SPECIFICATION`  
**Date:** November 2026  
**Auditor / Architect:** Senior AI Engineer, Software Architect, Security Engineer  
**Workspace:** `d:\ETGNN-Fraud-Detection`  
**Status:** ARCHITECTURAL & API CONTRACT SPECIFICATION  

---

## 1. Executive Summary & Privacy Principles

Financial crime compliance systems demand deep auditability, but enterprise security mandates that private internal model prompts, raw LLM chain-of-thought traces, system instructions, and credential keys remain strictly shielded from user-facing clients.

This specification designs the **Observable Event Stream** and **Structured Investigation Report Contract** for the upcoming ETGNN-Agent (Phases 2–4).

### Core Architectural Principles:
1. **Zero Chain-of-Thought Leakage**: Internal scratchpad reasoning is quarantined within the backend orchestrator and NEVER streamed to the browser.
2. **Deterministic Tool Traces**: The UI displays concrete, typed tool invocations (e.g. `get_transaction(23042587) -> Status: OK, 2 Hops, In-Degree: 3`).
3. **Evidence Citations**: Every analytical assertion must carry a traceable `evidence_id`.
4. **Transparent Limitation Disclosure**: The agent must explicitly output an `uncertain_findings` and `limitations` block in every forensic report.

---

## 2. Observable Event Stream Schema (Server-Sent Events / WebSockets)

During an autonomous transaction investigation, the backend streams typed progress events conforming to the following JSON schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AgentInvestigationEvent",
  "type": "object",
  "required": [
    "event_id",
    "investigation_id",
    "event_type",
    "timestamp",
    "execution_status",
    "duration_ms"
  ],
  "properties": {
    "event_id": { "type": "string", "example": "evt_01HXYZ789" },
    "investigation_id": { "type": "string", "example": "inv_987654" },
    "event_type": {
      "type": "string",
      "enum": [
        "INVESTIGATION_STARTED",
        "TRANSACTION_RETRIEVED",
        "PREDICTION_MODEL_INVOKED",
        "MODEL_RESULT_RECEIVED",
        "GRAPH_NEIGHBORS_RETRIEVED",
        "EXPLANATION_RETRIEVED",
        "RESEARCH_EVIDENCE_RETRIEVED",
        "EVIDENCE_VALIDATION_COMPLETED",
        "REPORT_GENERATION_COMPLETED",
        "INVESTIGATION_FAILED",
        "INVESTIGATION_ABSTAINED"
      ]
    },
    "timestamp": { "type": "string", "format": "date-time" },
    "tool_name": { "type": "string", "nullable": true, "example": "get_transaction_neighbors" },
    "execution_status": { "type": "string", "enum": ["RUNNING", "SUCCESS", "FAILED", "SKIPPED"] },
    "duration_ms": { "type": "integer", "example": 142 },
    "safe_input_summary": {
      "type": "object",
      "description": "Sanitized parameters only. Zero prompts or secrets.",
      "example": { "tx_id": 23042587, "hops": 2 }
    },
    "safe_output_summary": {
      "type": "object",
      "description": "High-level summary metrics.",
      "example": { "nodes_found": 14, "edges_found": 18, "illicit_ancestors": 1 }
    },
    "evidence_ref_ids": {
      "type": "array",
      "items": { "type": "string" },
      "example": ["EV-TX-23042587", "EV-SUB-23042587-2HOP"]
    },
    "error_code": { "type": "string", "nullable": true }
  }
}
```

---

## 3. Final Investigation Report Schema

Once tool execution and Layer E guardrail verification complete, the agent delivers a structured, verifiable forensic report:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ForensicInvestigationReport",
  "type": "object",
  "required": [
    "investigation_id",
    "target_transaction_id",
    "snapshot_id",
    "model_predictions",
    "consensus_verdict",
    "graph_evidence",
    "supported_findings",
    "uncertain_findings",
    "limitations",
    "model_versions"
  ],
  "properties": {
    "investigation_id": { "type": "string" },
    "target_transaction_id": { "type": "integer" },
    "snapshot_id": { "type": "integer" },
    "model_predictions": {
      "type": "object",
      "properties": {
        "random_forest_prob": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "xgboost_prob": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "canonical_etgnn_prob": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "model_divergence_detected": { "type": "boolean" },
        "divergence_explanation": { "type": "string" }
      }
    },
    "consensus_verdict": {
      "type": "string",
      "enum": ["HIGH_RISK_ILLICIT", "MODERATE_SUSPICION", "LOW_RISK_LICIT", "INCONCLUSIVE_DATA_GAP"]
    },
    "graph_evidence": {
      "type": "object",
      "properties": {
        "incoming_edges_count": { "type": "integer" },
        "outgoing_edges_count": { "type": "integer" },
        "subgraph_nodes_2hop": { "type": "integer" },
        "illicit_neighbors_count": { "type": "integer" },
        "is_isolated_incoming": { "type": "boolean" }
      }
    },
    "gnnexplainer_evidence": {
      "type": "object",
      "properties": {
        "top_feature_indices": { "type": "array", "items": { "type": "integer" } },
        "fidelity_drop": { "type": "number" },
        "interpretation_disclaimer": { "type": "string" }
      }
    },
    "supported_findings": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "finding": { "type": "string" },
          "evidence_id": { "type": "string" },
          "confidence": { "type": "string", "enum": ["HIGH", "MEDIUM", "LOW"] }
        }
      }
    },
    "uncertain_findings": {
      "type": "array",
      "items": { "type": "string" }
    },
    "limitations": {
      "type": "array",
      "items": { "type": "string" }
    },
    "model_versions": {
      "type": "object",
      "properties": {
        "etgnn_hash": { "type": "string" },
        "rf_hash": { "type": "string" },
        "agent_version": { "type": "string" }
      }
    }
  }
}
```

---

## 4. Security & Privacy Boundary Enforcement

| Layer / Interface | Permitted Data | Strictly Prohibited Data |
|---|---|---|
| **Client / UI** | Event type, tool progress pills, elapsed time, sanitized feature IDs, citations, structured conclusions, risk scores. | Raw LLM chain-of-thought, system prompts, API keys, private system paths, database credentials, unescaped prompt templates. |
| **Backend Logger** | Full structured events, tool latency telemetry, anonymized query traces. | Plaintext user secrets, external LLM provider tokens. |
| **Agent Prompt Context** | Numeric feature arrays, snapshot integer, subgraph connectivity matrices, research text chunks. | Hallucinated feature names, fabricated real-world names, non-point-in-time test labels. |

---

## 5. Specification Certification

This specification establishes an unbreachable privacy boundary for future agent development while guaranteeing full forensic auditability for human compliance investigators.

