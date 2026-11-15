"""
Retrieval Metrics for Hybrid Graph-RAG.
Evaluates Recall@K, Precision@K, Mean Reciprocal Rank (MRR),
evidence source coverage, and retrieval latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class RetrievalMetricResult:
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    source_coverage: float
    mean_retrieval_latency_ms: float
    k: int
    sample_sizes: Dict[str, int]
    denominators: Dict[str, str]


class RetrievalMetricsEvaluator:
    """Evaluates RAG and graph evidence retrieval performance."""

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate_retrieval_query(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        retrieved_sources: Set[str],
        expected_sources: Set[str],
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Evaluates a single retrieval step."""
        top_k = retrieved_ids[: self.k]
        num_relevant = len(relevant_ids)
        if num_relevant == 0:
            return {
                "recall_at_k": 1.0,
                "precision_at_k": 1.0,
                "mrr": 1.0,
                "source_coverage": 1.0,
                "latency_ms": latency_ms,
            }

        # Recall@K and Precision@K
        hits = sum(1 for item in top_k if item in relevant_ids)
        recall = hits / max(1, num_relevant)
        precision = hits / max(1, len(top_k))

        # MRR
        mrr = 0.0
        for rank, item in enumerate(top_k, start=1):
            if item in relevant_ids:
                mrr = 1.0 / rank
                break

        # Source coverage (e.g. graph, tabular, literature)
        if expected_sources:
            cov = len(retrieved_sources.intersection(expected_sources)) / len(expected_sources)
        else:
            cov = 1.0

        return {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "mrr": mrr,
            "source_coverage": cov,
            "latency_ms": latency_ms,
        }

    def aggregate_retrieval(self, query_evals: List[Dict[str, Any]]) -> RetrievalMetricResult:
        """Aggregates retrieval evaluations across all queries."""
        n = len(query_evals)
        if n == 0:
            return RetrievalMetricResult(0.0, 0.0, 0.0, 0.0, 0.0, self.k, {}, {})

        mean_rec = sum(q["recall_at_k"] for q in query_evals) / n
        mean_prec = sum(q["precision_at_k"] for q in query_evals) / n
        mean_mrr = sum(q["mrr"] for q in query_evals) / n
        mean_cov = sum(q["source_coverage"] for q in query_evals) / n
        mean_lat = sum(q["latency_ms"] for q in query_evals) / n

        sample_sizes = {
            "retrieval_queries_evaluated": n,
            "top_k_cutoff": self.k,
        }

        denominators = {
            "recall_at_k": f"Total relevant evidence items per query (cutoff K={self.k})",
            "precision_at_k": f"Top K retrieved items per query (K={self.k})",
            "mean_reciprocal_rank": "Total retrieval queries evaluated",
            "source_coverage": "Expected distinct evidence modality sources (graph, tabular, literature)",
            "mean_retrieval_latency_ms": "Total retrieval queries executed",
        }

        return RetrievalMetricResult(
            recall_at_k=round(mean_rec, 4),
            precision_at_k=round(mean_prec, 4),
            mean_reciprocal_rank=round(mean_mrr, 4),
            source_coverage=round(mean_cov, 4),
            mean_retrieval_latency_ms=round(mean_lat, 2),
            k=self.k,
            sample_sizes=sample_sizes,
            denominators=denominators,
        )
