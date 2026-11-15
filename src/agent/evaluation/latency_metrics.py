"""
Latency and System Reliability Metrics.
Computes Mean, Median, and P95 latency alongside failure, timeout,
provider error, and report validation failure rates with explicit sample sizes.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class LatencyMetricResult:
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    failure_rate: float
    timeout_rate: float
    provider_error_rate: float
    report_validation_failure_rate: float
    sample_sizes: Dict[str, int]
    denominators: Dict[str, str]


class LatencyMetricsEvaluator:
    """Evaluates latency percentiles and reliability error rates."""

    TIMEOUT_THRESHOLD_MS = 30000.0

    def evaluate_reliability(
        self,
        latencies_ms: List[float],
        failures: List[bool],
        provider_calls_total: int,
        provider_calls_failed: int,
        reports_generated: int,
        reports_validation_failed: int,
    ) -> LatencyMetricResult:
        """Computes comprehensive latency and reliability metrics."""
        n = len(latencies_ms)
        if n == 0:
            return LatencyMetricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})

        mean_lat = statistics.mean(latencies_ms)
        median_lat = statistics.median(latencies_ms)

        # Compute P95
        sorted_lat = sorted(latencies_ms)
        idx_p95 = int(0.95 * len(sorted_lat))
        p95_lat = sorted_lat[min(idx_p95, len(sorted_lat) - 1)]

        # Error rates
        fail_rate = sum(1 for f in failures if f) / max(1, n)
        timeouts = sum(1 for lat in latencies_ms if lat >= self.TIMEOUT_THRESHOLD_MS)
        timeout_rate = timeouts / max(1, n)

        prov_err_rate = provider_calls_failed / max(1, provider_calls_total) if provider_calls_total > 0 else 0.0
        val_fail_rate = reports_validation_failed / max(1, reports_generated) if reports_generated > 0 else 0.0

        sample_sizes = {
            "total_runs_measured": n,
            "total_provider_calls": provider_calls_total,
            "total_reports_generated": reports_generated,
            "timeout_events_count": timeouts,
        }

        denominators = {
            "mean_latency_ms": "Total executed tasks (n)",
            "median_latency_ms": "Total executed tasks (n)",
            "p95_latency_ms": "Total executed tasks (n)",
            "failure_rate": "Total executed tasks (n)",
            "timeout_rate": f"Total executed tasks (cutoff >= {self.TIMEOUT_THRESHOLD_MS} ms)",
            "provider_error_rate": "Total LLM / model provider inference calls",
            "report_validation_failure_rate": "Total generated investigation reports validated",
        }

        return LatencyMetricResult(
            mean_latency_ms=round(mean_lat, 2),
            median_latency_ms=round(median_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            failure_rate=round(fail_rate, 4),
            timeout_rate=round(timeout_rate, 4),
            provider_error_rate=round(prov_err_rate, 4),
            report_validation_failure_rate=round(val_fail_rate, 4),
            sample_sizes=sample_sizes,
            denominators=denominators,
        )
