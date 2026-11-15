from __future__ import annotations

import threading
import uuid
import time
from typing import Any, Dict, List, Optional
from src.agent.orchestration.graph import build_investigation_graph
from src.agent.orchestration.state import InvestigationState, InvestigationStatus
from src.agent.telemetry.event_sink import get_event_sink
from src.agent.telemetry.events import AgentInvestigationEvent


class InvestigationRunner:
    """
    Thread-safe runner and cache for LangGraph investigations.
    Provides execution, caching, and retrieval of full forensic reports.
    """
    def __init__(self, max_cached: int = 100):
        self._max_cached = max_cached
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._graph = build_investigation_graph()

    def run_investigation(
        self,
        tx_id: int,
        question: Optional[str] = None,
        evidence_mode: str = "RETROSPECTIVE_INVESTIGATION",
        as_of_snapshot: Optional[int] = None,
        investigation_id: Optional[str] = None,
        llm_provider: Optional[str] = None,
        allow_evidence_only_fallback: bool = True,
    ) -> Dict[str, Any]:
        inv_id = investigation_id or f"inv_{uuid.uuid4().hex[:10]}"
        q = question or f"Investigate transaction {tx_id} for illicit patterns and counterparty risk."

        initial_state: InvestigationState = {
            "investigation_id": inv_id,
            "tx_id": tx_id,
            "question": q,
            "evidence_mode": evidence_mode,
            "as_of_snapshot": as_of_snapshot,
            "status": InvestigationStatus.NOT_STARTED.value,
            "llm_provider": llm_provider,
            "allow_evidence_only_fallback": allow_evidence_only_fallback,
            "plan": None,
            "tool_history": [],
            "evidence_bundle": {},
            "provenance_chain": [],
            "warnings": [],
            "errors": [],
            "raw_explanation": None,
            "claim_validation_result": None,
            "report_validation_result": None,
            "final_report": None,
            "retries_count": 0,
            "execution_limits_exceeded": False,
            "abstention_reason": None,
            "start_time": time.time(),
            "duration_ms": 0,
        }

        final_state = self._graph.invoke(initial_state)

        with self._lock:
            if len(self._cache) >= self._max_cached:
                # Evict oldest entry
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)
            self._cache[inv_id] = final_state

        return final_state

    def get_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(investigation_id)

    def list_investigations(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())


# Singleton instance
_runner_instance: Optional[InvestigationRunner] = None


def get_investigation_runner() -> InvestigationRunner:
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = InvestigationRunner()
    return _runner_instance

