# Phase 2 to Phase 3: Integration Contract Specification

**Document ID:** `PHASE3_INTEGRATION_CONTRACT`  
**Date:** November 2026  
**Author:** Principal AI Engineer, Software Architect  
**Workspace:** `d:\ETGNN-Fraud-Detection`  

---

## 1. Contract Overview

This document specifies the exact programmatic interface, data models, error codes, and architectural boundaries governing the transition from **Phase 2 (Investigation Tools & Graph-RAG)** to **Phase 3 (LangGraph Autonomous Investigation Agent)**.

In Phase 3, the LangGraph agent orchestrates multi-step forensic reasoning by invoking the Phase 2 tools. To prevent security vulnerabilities and epistemic drift, the agent interacts **exclusively through typed Python methods** exposed by `InvestigationToolRegistry` and `InvestigationContextService`.

---

## 2. Python Tool Call Interface for LangGraph

```python
from src.agent.tools.investigation_tools import (
    get_transaction,
    get_transaction_neighbors,
    get_prediction,
    get_explanation,
    get_model_metadata,
    get_evaluation_metrics,
)
from src.agent.services.investigation_context_service import InvestigationContextService
from src.agent.schemas import (
    InvestigationContext,
    TransactionEvidence,
    GraphEvidence,
    PredictionEvidence,
    ExplanationEvidence,
    ModelMetricEvidence,
    ModelMetadataEvidence,
    AgentToolError,
    TransactionNotFoundError,
    TemporalBoundaryViolationError,
    ModelNotSupportedError,
)
```

### LangGraph State Schema (Preview for Phase 3)
```python
class InvestigationState(TypedDict):
    tx_id: int
    as_of_snapshot: Optional[int]
    evidence_mode: str
    query_question: str
    
    # Accumulated Evidence
    transaction: Optional[TransactionEvidence]
    subgraph: Optional[GraphEvidence]
    predictions: Dict[str, PredictionEvidence]
    explanation: Optional[ExplanationEvidence]
    retrieved_research: List[ResearchDocumentEvidence]
    reproduced_metrics: List[ModelMetricEvidence]
    
    # Analytical Synthesis
    divergence_detected: bool
    risk_verdict: Optional[str]
    forensic_narrative: Optional[str]
    citations: List[str]
    warnings: List[str]
```

---

## 3. High-Level Service Call Contract

For one-shot or batch investigations, Phase 3 can call `InvestigationContextService`:

```python
service = InvestigationContextService()
context: InvestigationContext = service.build_investigation_context(
    tx_id=23042587,
    question="Investigate high transaction value and counterparty connections",
    evidence_mode="HISTORICAL_EVALUATION",
    as_of_snapshot=35,
    top_k_research=4,
)
```

### Guaranteed Context Invariants:
1. `context.tx_id == tx_id`
2. `context.transaction_evidence` is strictly non-null and point-in-time validated.
3. `context.predictions` contains deterministic results for `etgnn`, `random_forest`, and `xgboost`.
4. `context.disagreement` computes exact probability spread and flags divergence if $\ge 0.30$.
5. `context.provenance_chain` provides an ordered array of immutable evidence IDs.
6. `context.investigation_warnings` logs any sanitized prompt injection or temporal issues.

---

## 4. Error Handling Contract

Phase 3 orchestrators must catch typed `AgentToolError` subclasses:

| Exception Class | Trigger Condition | Recommended Agent Handling |
|---|---|---|
| `TransactionNotFoundError` | `tx_id` does not exist in dataset | Halt investigation; return `INVESTIGATION_ABSTAINED` with explanation. |
| `TemporalBoundaryViolationError` | `tx_id` belongs to snapshot beyond `as_of_snapshot` in historical mode | Reject query; inform user that transaction occurred after simulation cutoff. |
| `ResourceLimitExceededError` | Traversal exceeds 2 hops or 100 nodes | Fall back to 1-hop traversal with max 50 nodes. |
| `ModelNotSupportedError` | Model identifier not in verified registry | Fall back to canonical `etgnn` and `random_forest`. |

---

## 5. Security & Boundary Guardrails

1. **No Filesystem Access**: The LangGraph agent will NOT have tools to read arbitrary files, run shell commands, or write to disk. It interacts strictly with these typed Python functions.
2. **Feature Semantics Guardrail**: If an LLM prompt synthesizes text, Phase 3 Layer E guardrails will regex-match against unauthorized feature names (e.g. `feat_.* means .*`) and reject the response if synthetic meanings are manufactured.
3. **Point-in-Time Quarantine**: The agent cannot bypass historical evaluation mode to view test ground-truth labels.

