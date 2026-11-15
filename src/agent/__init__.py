"""
ETGNN-Agent: Evidence-Grounded Agentic AI System for Bitcoin Fraud Detection.
Phase 3: Advanced LangGraph Agentic Bitcoin Transaction Investigator.
"""

from src.agent.orchestration.state import InvestigationState, InvestigationStatus
from src.agent.reports.report_builder import ForensicInvestigationReport
from src.agent.telemetry.event_sink import get_event_sink, ObservableEventSink
from src.agent.telemetry.events import AgentInvestigationEvent, InvestigationEventType

__version__ = "3.0.0"

def build_investigation_graph(*args, **kwargs):
    from src.agent.orchestration.graph import build_investigation_graph as _build
    return _build(*args, **kwargs)

__all__ = [
    "build_investigation_graph",
    "InvestigationState",
    "InvestigationStatus",
    "ForensicInvestigationReport",
    "get_event_sink",
    "ObservableEventSink",
    "AgentInvestigationEvent",
    "InvestigationEventType",
]
