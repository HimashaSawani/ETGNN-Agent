from __future__ import annotations

import time
from typing import Optional
from pydantic import BaseModel, Field
from src.agent.schemas import ResourceLimitExceededError


MAX_TOOL_CALLS = 8
MAX_HOPS = 2
MAX_SUBGRAPH_NODES = 100
MAX_RAG_CHUNKS = 6
MAX_REGENERATION_ATTEMPTS = 1
MAX_EXECUTION_TIME_SECONDS = 60.0


class ExecutionBudgetTracker(BaseModel):
    """
    Monitors and enforces resource bounds during graph execution.
    Prevents infinite loops, resource exhaustion, and memory leaks.
    """
    start_time: float = Field(default_factory=time.time)
    tool_calls_count: int = 0
    regeneration_attempts: int = 0
    max_tool_calls: int = MAX_TOOL_CALLS
    max_regenerations: int = MAX_REGENERATION_ATTEMPTS
    timeout_seconds: float = MAX_EXECUTION_TIME_SECONDS

    def record_tool_call(self, tool_name: str) -> None:
        self.tool_calls_count += 1
        self.check_limits()

    def record_regeneration(self) -> None:
        self.regeneration_attempts += 1
        if self.regeneration_attempts > self.max_regenerations:
            raise ResourceLimitExceededError(
                f"Maximum regeneration attempts ({self.max_regenerations}) exceeded."
            )

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def check_limits(self) -> None:
        if self.tool_calls_count > self.max_tool_calls:
            raise ResourceLimitExceededError(
                f"Execution budget exceeded: Tool call limit of {self.max_tool_calls} reached."
            )
        if self.elapsed_time() > self.timeout_seconds:
            raise ResourceLimitExceededError(
                f"Execution timeout: Exceeded maximum time of {self.timeout_seconds} seconds."
            )

