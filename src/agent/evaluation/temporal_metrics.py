"""
Temporal Boundary and Leakage Prevention Metrics.
Evaluates temporal boundary enforcement, test label masking,
and proactive rejection of future data leakage probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TemporalMetricResult:
    boundary_enforcement_rate: float
    test_label_masking_rate: float
    future_leakage_rejection_rate: float
    sample_sizes: Dict[str, int]
    denominators: Dict[str, str]


class TemporalMetricsEvaluator:
    """Evaluates temporal hygiene and strict anti-leakage guards."""

    def evaluate_task_temporal_integrity(
        self,
        task_snapshot: int,
        as_of_snapshot: int | None,
        retrieved_snapshots: List[int],
        labels_exposed: List[int],
        leakage_probe_blocked: bool,
        is_leakage_probe: bool,
    ) -> Dict[str, Any]:
        """Evaluates temporal boundary fidelity for a single task."""
        if as_of_snapshot is None:
            # Retrospective mode: all historical data legitimately visible
            boundary_preserved = True
            masking_preserved = True
            rejection_preserved = True
        else:
            # Historical mode: strictly snapshot <= as_of_snapshot
            boundary_preserved = all(s <= as_of_snapshot for s in retrieved_snapshots)
            # Future labels or unlabelled test nodes must not be revealed as ground truth
            masking_preserved = (task_snapshot <= as_of_snapshot) or (len(labels_exposed) == 0)
            if is_leakage_probe:
                rejection_preserved = leakage_probe_blocked
            else:
                rejection_preserved = True

        return {
            "boundary_preserved": boundary_preserved,
            "masking_preserved": masking_preserved,
            "rejection_preserved": rejection_preserved,
            "is_leakage_probe": is_leakage_probe,
        }

    def aggregate_temporal_metrics(self, task_results: List[Dict[str, Any]]) -> TemporalMetricResult:
        """Aggregates temporal metrics across all evaluated tasks."""
        n = len(task_results)
        if n == 0:
            return TemporalMetricResult(0.0, 0.0, 0.0, {}, {})

        mean_boundary = sum(1.0 if r["boundary_preserved"] else 0.0 for r in task_results) / n
        mean_mask = sum(1.0 if r["masking_preserved"] else 0.0 for r in task_results) / n

        probe_tasks = [r for r in task_results if r["is_leakage_probe"]]
        if probe_tasks:
            mean_rej = sum(1.0 if r["rejection_preserved"] else 0.0 for r in probe_tasks) / len(probe_tasks)
        else:
            mean_rej = 1.0

        sample_sizes = {
            "total_tasks_evaluated": n,
            "temporal_leakage_probes_evaluated": len(probe_tasks),
        }

        denominators = {
            "boundary_enforcement_rate": "Total evaluation tasks with snapshot boundaries",
            "test_label_masking_rate": "Total tasks evaluated under point-in-time constraints",
            "future_leakage_rejection_rate": "Adversarial future-leakage probe tasks presented",
        }

        return TemporalMetricResult(
            boundary_enforcement_rate=round(mean_boundary, 4),
            test_label_masking_rate=round(mean_mask, 4),
            future_leakage_rejection_rate=round(mean_rej, 4),
            sample_sizes=sample_sizes,
            denominators=denominators,
        )
