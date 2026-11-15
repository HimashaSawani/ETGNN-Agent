from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class InvestigationStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    VALIDATING_REQUEST = "VALIDATING_REQUEST"
    PLANNING = "PLANNING"
    EXECUTING_TOOLS = "EXECUTING_TOOLS"
    CONSOLIDATING_EVIDENCE = "CONSOLIDATING_EVIDENCE"
    GENERATING_EXPLANATION = "GENERATING_EXPLANATION"
    VALIDATING_CLAIMS = "VALIDATING_CLAIMS"
    VALIDATING_REPORT = "VALIDATING_REPORT"
    COMPLETED = "COMPLETED"
    PARTIAL_RESULT = "PARTIAL_RESULT"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    ABSTAINED = "ABSTAINED"
    FAILED = "FAILED"


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    rationale: str
    executed: bool = False
    result_evidence_id: Optional[str] = None
    error: Optional[str] = None


class InvestigationPlan(BaseModel):
    planned_tools: List[ToolCallRequest] = Field(default_factory=list)
    reasoning_summary: str = ""
    requires_graph: bool = True
    requires_explanation: bool = True
    requires_research: bool = True


class InvestigationState(TypedDict, total=False):
    """
    Central state dictionary passed across LangGraph nodes.
    Maintains complete provenance, accumulated evidence, and audit logs.
    """
    investigation_id: str
    tx_id: int
    question: str
    evidence_mode: str
    as_of_snapshot: Optional[int]
    status: str
    plan: Optional[Dict[str, Any]]
    tool_history: List[Dict[str, Any]]
    evidence_bundle: Dict[str, Any]
    provenance_chain: List[str]
    warnings: List[str]
    errors: List[str]
    raw_explanation: Optional[Dict[str, Any]]
    claim_validation_result: Optional[Dict[str, Any]]
    report_validation_result: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    retries_count: int
    execution_limits_exceeded: bool
    abstention_reason: Optional[str]
    llm_provider: Optional[str]
    allow_evidence_only_fallback: bool
    evidence_completeness: str
    workflow_stages_completed: List[str]
    start_time: float
    duration_ms: int

