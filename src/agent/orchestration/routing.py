from __future__ import annotations

from typing import Dict, Any
from src.agent.orchestration.state import InvestigationState, InvestigationStatus
from src.agent.orchestration.execution_limits import MAX_REGENERATION_ATTEMPTS


def route_after_request_validation(state: InvestigationState) -> str:
    """Routes to planning if validation passed, otherwise directly to finalization/abstention."""
    status = state.get("status")
    if status in (InvestigationStatus.ABSTAINED.value, InvestigationStatus.FAILED.value):
        return "finalize_investigation"
    return "plan_investigation"


def route_after_claim_validation(state: InvestigationState) -> str:
    """
    Evaluates claim validation outcome.
    Allows 1 regeneration if claims are unsupported and retry budget remains.
    """
    claim_res = state.get("claim_validation_result") or {}
    is_valid = claim_res.get("is_valid", True)
    retries = state.get("retries_count", 0)

    if not is_valid and retries < MAX_REGENERATION_ATTEMPTS:
        return "generate_explanation"

    return "validate_final_report"


def route_after_report_validation(state: InvestigationState) -> str:
    """
    Evaluates final report validation.
    Routes to regeneration if critical errors exist and retries remain,
    otherwise proceeds to finalization.
    """
    report_res = state.get("report_validation_result") or {}
    should_regen = report_res.get("should_regenerate", False)
    retries = state.get("retries_count", 0)

    if should_regen and retries < MAX_REGENERATION_ATTEMPTS:
        return "generate_explanation"

    return "finalize_investigation"

