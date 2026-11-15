"""
ETGNN-Agent Evaluation Framework Module.
Provides modular datasets, metrics, semantic citation benchmarks, and ablation runners.
"""

from __future__ import annotations

from src.agent.evaluation.datasets import (
    CohortType,
    EvaluationFixture,
    PHASE5_EVALUATION_DATASET,
    get_evaluation_dataset,
)
from src.agent.evaluation.grounding_metrics import (
    GroundingMetricResult,
    GroundingMetricsEvaluator,
)
from src.agent.evaluation.tool_metrics import (
    ToolMetricResult,
    ToolMetricsEvaluator,
)
from src.agent.evaluation.retrieval_metrics import (
    RetrievalMetricResult,
    RetrievalMetricsEvaluator,
)
from src.agent.evaluation.latency_metrics import (
    LatencyMetricResult,
    LatencyMetricsEvaluator,
)
from src.agent.evaluation.temporal_metrics import (
    TemporalMetricResult,
    TemporalMetricsEvaluator,
)
from src.agent.evaluation.adversarial_cases import (
    AdversarialCase,
    ADVERSARIAL_CASES,
    get_adversarial_suite,
)
from src.agent.evaluation.semantic_citation import (
    SemanticCitationBenchmarkItem,
    SemanticCitationEvaluator,
    SupportClassification,
    SEMANTIC_CITATION_BENCHMARK,
)
from src.agent.evaluation.metrics import (
    AgentMetricsHub,
    ConsolidatedAgentEvaluationReport,
)
from src.agent.evaluation.runner import (
    AblationSystemResult,
    Phase5AblationRunner,
    run_phase5_ablation_benchmark,
)

__all__ = [
    "CohortType",
    "EvaluationFixture",
    "PHASE5_EVALUATION_DATASET",
    "get_evaluation_dataset",
    "GroundingMetricResult",
    "GroundingMetricsEvaluator",
    "ToolMetricResult",
    "ToolMetricsEvaluator",
    "RetrievalMetricResult",
    "RetrievalMetricsEvaluator",
    "LatencyMetricResult",
    "LatencyMetricsEvaluator",
    "TemporalMetricResult",
    "TemporalMetricsEvaluator",
    "AdversarialCase",
    "ADVERSARIAL_CASES",
    "get_adversarial_suite",
    "SemanticCitationBenchmarkItem",
    "SemanticCitationEvaluator",
    "SupportClassification",
    "SEMANTIC_CITATION_BENCHMARK",
    "AgentMetricsHub",
    "ConsolidatedAgentEvaluationReport",
    "AblationSystemResult",
    "Phase5AblationRunner",
    "run_phase5_ablation_benchmark",
]
