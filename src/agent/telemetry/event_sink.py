from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, List, Optional
from src.agent.telemetry.events import AgentInvestigationEvent


class ObservableEventSink:
    """
    Thread-safe in-memory event bus and circular telemetry buffer.
    Maintains observable event history for investigation traces.
    """
    def __init__(self, max_events_per_investigation: int = 200):
        self._max_events = max_events_per_investigation
        self._events_by_inv: Dict[str, List[AgentInvestigationEvent]] = defaultdict(list)
        self._lock = threading.RLock()

    def emit(self, event: AgentInvestigationEvent) -> None:
        with self._lock:
            history = self._events_by_inv[event.investigation_id]
            if len(history) >= self._max_events:
                history.pop(0)
            history.append(event)

    def get_events(self, investigation_id: str) -> List[AgentInvestigationEvent]:
        with self._lock:
            return list(self._events_by_inv.get(investigation_id, []))

    def get_investigation_ids(self) -> List[str]:
        with self._lock:
            return list(self._events_by_inv.keys())

    def clear(self, investigation_id: Optional[str] = None) -> None:
        with self._lock:
            if investigation_id:
                self._events_by_inv.pop(investigation_id, None)
            else:
                self._events_by_inv.clear()


# Global default sink
_default_event_sink = ObservableEventSink()


def get_event_sink() -> ObservableEventSink:
    return _default_event_sink

