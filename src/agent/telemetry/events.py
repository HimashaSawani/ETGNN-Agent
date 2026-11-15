from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, ConfigDict


class InvestigationEventType(str, Enum):
    INVESTIGATION_STARTED = "INVESTIGATION_STARTED"
    REQUEST_VALIDATED = "REQUEST_VALIDATED"
    PLAN_CREATED = "PLAN_CREATED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    EVIDENCE_CONSOLIDATED = "EVIDENCE_CONSOLIDATED"
    EXPLANATION_STARTED = "EXPLANATION_STARTED"
    EXPLANATION_GENERATED = "EXPLANATION_GENERATED"
    CLAIMS_VALIDATED = "CLAIMS_VALIDATED"
    REPORT_VALIDATED = "REPORT_VALIDATED"
    INVESTIGATION_COMPLETED = "INVESTIGATION_COMPLETED"
    INVESTIGATION_ABSTAINED = "INVESTIGATION_ABSTAINED"
    INVESTIGATION_FAILED = "INVESTIGATION_FAILED"


class ExecutionStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class AgentInvestigationEvent(BaseModel):
    """
    Observable audit event emitted during an investigation.
    Strictly sanitizes internal chain-of-thought and secrets.
    """
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    investigation_id: str = Field(..., description="Unique investigation run ID")
    event_type: InvestigationEventType
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    tool_name: Optional[str] = None
    execution_status: ExecutionStatus = Field(default=ExecutionStatus.SUCCESS)
    duration_ms: int = Field(default=0, ge=0)
    safe_input_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sanitized parameters only. Zero prompts, secrets, or internal paths."
    )
    safe_output_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="High-level summary metrics and counts."
    )
    evidence_ref_ids: List[str] = Field(default_factory=list, description="Associated evidence IDs")
    error_code: Optional[str] = None
    message: Optional[str] = None

