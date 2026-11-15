"""
Temporal Prediction Drift Detection Module.
Monitors shifts in predicted probability distributions, predicted class proportions,
threshold crossing rates, and multi-model disagreement across chronological snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SnapshotPredictionSummary:
    snapshot_id: int
    sample_size: int
    mean_probability: float
    std_probability: float
    median_probability: float
    p90_probability: float
    predicted_illicit_rate_050: float
    predicted_illicit_rate_tuned: float  # e.g. at 0.42 threshold
    model_disagreement_rate: float  # e.g. ETGNN vs RF


@dataclass
class PredictionDriftReport:
    model_name: str
    baseline_window: str
    target_window: str
    baseline_mean_prob: float
    target_mean_prob: float
    probability_shift_delta: float
    baseline_predicted_illicit_pct: float
    target_predicted_illicit_pct: float
    threshold_crossing_shift: float
    max_disagreement_snapshot: int
    max_disagreement_rate: float
    snapshots_analyzed: List[SnapshotPredictionSummary]
    scientific_distinction: str


class PredictionDriftDetector:
    """Evaluates prediction shifts and cross-model divergence over time."""

    def __init__(self, tuned_threshold: float = 0.42):
        self.tuned_threshold = tuned_threshold

    def analyze_snapshot_predictions(
        self,
        snapshot_ids: np.ndarray,
        primary_probs: np.ndarray,
        secondary_probs: Optional[np.ndarray] = None,
        model_name: str = "ETGNN",
        baseline_snapshots: Tuple[int, int] = (1, 32),
        target_snapshots: Tuple[int, int] = (41, 49),
    ) -> PredictionDriftReport:
        """
        Analyzes probability shift across chronological snapshots and computes disagreement against secondary model.
        """
        unique_snaps = sorted(list(set(snapshot_ids)))
        summaries: List[SnapshotPredictionSummary] = []

        baseline_mask = (snapshot_ids >= baseline_snapshots[0]) & (snapshot_ids <= baseline_snapshots[1])
        target_mask = (snapshot_ids >= target_snapshots[0]) & (snapshot_ids <= target_snapshots[1])

        base_mean = float(np.mean(primary_probs[baseline_mask])) if np.any(baseline_mask) else 0.0
        tgt_mean = float(np.mean(primary_probs[target_mask])) if np.any(target_mask) else 0.0

        base_rate_05 = float(np.mean(primary_probs[baseline_mask] >= 0.50)) if np.any(baseline_mask) else 0.0
        tgt_rate_05 = float(np.mean(primary_probs[target_mask] >= 0.50)) if np.any(target_mask) else 0.0

        max_disagree_snap = -1
        max_disagree_val = 0.0

        for snap in unique_snaps:
            snap_mask = (snapshot_ids == snap)
            s_probs = primary_probs[snap_mask]
            if len(s_probs) == 0:
                continue

            # Compute model disagreement if secondary model provided
            if secondary_probs is not None:
                sec_s_probs = secondary_probs[snap_mask]
                disagree_binary = (s_probs >= 0.50) != (sec_s_probs >= 0.50)
                disagree_rate = float(np.mean(disagree_binary))
            else:
                disagree_rate = 0.0

            if disagree_rate > max_disagree_val:
                max_disagree_val = disagree_rate
                max_disagree_snap = snap

            summary = SnapshotPredictionSummary(
                snapshot_id=int(snap),
                sample_size=int(len(s_probs)),
                mean_probability=round(float(np.mean(s_probs)), 4),
                std_probability=round(float(np.std(s_probs)), 4),
                median_probability=round(float(np.median(s_probs)), 4),
                p90_probability=round(float(np.percentile(s_probs, 90)), 4),
                predicted_illicit_rate_050=round(float(np.mean(s_probs >= 0.50)), 4),
                predicted_illicit_rate_tuned=round(float(np.mean(s_probs >= self.tuned_threshold)), 4),
                model_disagreement_rate=round(disagree_rate, 4),
            )
            summaries.append(summary)

        return PredictionDriftReport(
            model_name=model_name,
            baseline_window=f"Snapshots {baseline_snapshots[0]}-{baseline_snapshots[1]}",
            target_window=f"Snapshots {target_snapshots[0]}-{target_snapshots[1]}",
            baseline_mean_prob=round(base_mean, 4),
            target_mean_prob=round(tgt_mean, 4),
            probability_shift_delta=round(tgt_mean - base_mean, 4),
            baseline_predicted_illicit_pct=round(base_rate_05, 4),
            target_predicted_illicit_pct=round(tgt_rate_05, 4),
            threshold_crossing_shift=round(tgt_rate_05 - base_rate_05, 4),
            max_disagreement_snapshot=max_disagree_snap,
            max_disagreement_rate=round(max_disagree_val, 4),
            snapshots_analyzed=summaries,
            scientific_distinction="Prediction drift measures shifts in model outputs P(Y_hat). It must not be conflated with true concept drift P(Y|X), which requires ground-truth label changes.",
        )
