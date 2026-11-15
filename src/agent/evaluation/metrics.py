"""
Comprehensive Evaluation Metrics Aggregator for ETGNN-Agent.
Combines Grounding, Tool, Retrieval, Latency, and Temporal metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from src.agent.evaluation.grounding_metrics import GroundingMetricResult, GroundingMetricsEvaluator
from src.agent.evaluation.latency_metrics import LatencyMetricResult, LatencyMetricsEvaluator
from src.agent.evaluation.retrieval_metrics import RetrievalMetricResult, RetrievalMetricsEvaluator
from src.agent.evaluation.temporal_metrics import TemporalMetricResult, TemporalMetricsEvaluator
from src.agent.evaluation.tool_metrics import ToolMetricResult, ToolMetricsEvaluator


@dataclass
class ConsolidatedAgentEvaluationReport:
    grounding: GroundingMetricResult
    tools: ToolMetricResult
    retrieval: RetrievalMetricResult
    temporal: TemporalMetricResult
    latency: LatencyMetricResult
    total_tasks_evaluated: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounding": asdict(self.grounding),
            "tools": asdict(self.tools),
            "retrieval": asdict(self.retrieval),
            "temporal": asdict(self.temporal),
            "latency": asdict(self.latency),
            "total_tasks_evaluated": self.total_tasks_evaluated,
        }


class AgentMetricsHub:
    """Central evaluator for all agent evaluation metric suites."""

    def __init__(self):
        self.grounding_evaluator = GroundingMetricsEvaluator()
        self.tool_evaluator = ToolMetricsEvaluator()
        self.retrieval_evaluator = RetrievalMetricsEvaluator()
        self.temporal_evaluator = TemporalMetricsEvaluator()
        self.latency_evaluator = LatencyMetricsEvaluator()

    def build_report(
        self,
        grounding_records: List[Dict[str, Any]],
        tool_records: List[Dict[str, Any]],
        retrieval_records: List[Dict[str, Any]],
        temporal_records: List[Dict[str, Any]],
        latencies_ms: List[float],
        failures: List[bool],
        provider_calls_total: int,
        provider_calls_failed: int,
        reports_generated: int,
        reports_validation_failed: int,
    ) -> ConsolidatedAgentEvaluationReport:
        """Assembles a consolidated, reproducible evaluation report."""
        gr = self.grounding_evaluator.aggregate_grounding(grounding_records)
        tr = self.tool_evaluator.aggregate_tool_metrics(tool_records)
        rr = self.retrieval_evaluator.aggregate_retrieval(retrieval_records)
        tm = self.temporal_evaluator.aggregate_temporal_metrics(temporal_records)
        lr = self.latency_evaluator.evaluate_reliability(
            latencies_ms=latencies_ms,
            failures=failures,
            provider_calls_total=provider_calls_total,
            provider_calls_failed=provider_calls_failed,
            reports_generated=reports_generated,
            reports_validation_failed=reports_validation_failed,
        )

        return ConsolidatedAgentEvaluationReport(
            grounding=gr,
            tools=tr,
            retrieval=rr,
            temporal=tm,
            latency=lr,
            total_tasks_evaluated=len(grounding_records),
        )
