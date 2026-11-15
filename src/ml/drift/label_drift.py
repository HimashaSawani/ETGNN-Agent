"""
Temporal Label and Performance Drift Detection Module.
Tracks illicit prevalence, unlabelled transaction proportions,
and model performance (AUPRC, Recall, Precision, Brier calibration) across snapshots.
Enforces strict preservation of unknown labels (never assumed licit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, brier_score_loss


@dataclass
class SnapshotLabelSummary:
    snapshot_id: int
    total_nodes: int
    labelled_count: int
    unlabelled_unknown_count: int
    unlabelled_proportion: float
    illicit_count: int
    licit_count: int
    illicit_prevalence_labelled: float
    auprc: Optional[float]
    recall: Optional[float]
    precision: Optional[float]
    brier_score: Optional[float]
    insufficient_samples: bool


@dataclass
class LabelDriftReport:
    total_snapshots: int
    baseline_window: str
    target_window: str
    baseline_illicit_prevalence: float
    target_illicit_prevalence: float
    prevalence_shift_delta: float
    baseline_mean_auprc: float
    target_mean_auprc: float
    auprc_shift_delta: float
    overall_unlabelled_count: int
    overall_unlabelled_pct: float
    snapshots_analyzed: List[SnapshotLabelSummary]
    unknown_label_safeguard: str


class LabelDriftDetector:
    """Analyzes class distribution shifts and snapshot-level model performance."""

    UNKNOWN_SAFEGUARD = (
        "Unknown labels (label -1 / raw 2) represent unlabelled Bitcoin transactions. "
        "They are strictly excluded from supervised evaluation denominators and NEVER treated as licit."
    )

    def analyze_label_and_performance(
        self,
        nodes_df: pd.DataFrame,
        probabilities: Optional[np.ndarray] = None,
        baseline_snapshots: Tuple[int, int] = (1, 32),
        target_snapshots: Tuple[int, int] = (41, 49),
    ) -> LabelDriftReport:
        """
        Computes illicit prevalence and snapshot-level AUPRC/metrics.
        """
        snapshots = sorted(nodes_df["time_step"].unique())
        summaries: List[SnapshotLabelSummary] = []

        total_nodes = len(nodes_df)
        total_unknown = int((nodes_df["label"] == -1).sum())
        overall_unlabelled_pct = (total_unknown / max(1, total_nodes))

        has_probs = probabilities is not None and len(probabilities) == len(nodes_df)

        base_auprcs: List[float] = []
        tgt_auprcs: List[float] = []

        base_mask_labelled = (nodes_df["time_step"] >= baseline_snapshots[0]) & (nodes_df["time_step"] <= baseline_snapshots[1]) & (nodes_df["label"] != -1)
        tgt_mask_labelled = (nodes_df["time_step"] >= target_snapshots[0]) & (nodes_df["time_step"] <= target_snapshots[1]) & (nodes_df["label"] != -1)

        base_prev = float(nodes_df.loc[base_mask_labelled, "label"].mean()) if base_mask_labelled.any() else 0.0
        tgt_prev = float(nodes_df.loc[tgt_mask_labelled, "label"].mean()) if tgt_mask_labelled.any() else 0.0

        for s in snapshots:
            s_mask = (nodes_df["time_step"] == s)
            s_df = nodes_df[s_mask]
            tot = len(s_df)
            if tot == 0:
                continue

            unlabelled = int((s_df["label"] == -1).sum())
            labelled_df = s_df[s_df["label"] != -1]
            n_lab = len(labelled_df)
            illicit = int((labelled_df["label"] == 1).sum())
            licit = int((labelled_df["label"] == 0).sum())

            prev = (illicit / max(1, n_lab)) if n_lab > 0 else 0.0

            # If fewer than 2 illicit or licit samples, AUPRC cannot be reliably calculated
            insufficient = (illicit < 2 or licit < 2)

            auprc_val: Optional[float] = None
            rec_val: Optional[float] = None
            prec_val: Optional[float] = None
            brier_val: Optional[float] = None

            if has_probs and not insufficient:
                lab_indices = labelled_df.index.values
                y_true = labelled_df["label"].values.astype(int)
                y_scores = probabilities[lab_indices]

                try:
                    auprc_val = round(float(average_precision_score(y_true, y_scores)), 4)
                    y_pred = (y_scores >= 0.50).astype(int)
                    rec_val = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
                    prec_val = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
                    brier_val = round(float(brier_score_loss(y_true, y_scores)), 4)

                    if baseline_snapshots[0] <= s <= baseline_snapshots[1]:
                        base_auprcs.append(auprc_val)
                    elif target_snapshots[0] <= s <= target_snapshots[1]:
                        tgt_auprcs.append(auprc_val)
                except Exception:
                    pass

            summaries.append(
                SnapshotLabelSummary(
                    snapshot_id=int(s),
                    total_nodes=tot,
                    labelled_count=n_lab,
                    unlabelled_unknown_count=unlabelled,
                    unlabelled_proportion=round(unlabelled / tot, 4),
                    illicit_count=illicit,
                    licit_count=licit,
                    illicit_prevalence_labelled=round(prev, 4),
                    auprc=auprc_val,
                    recall=rec_val,
                    precision=prec_val,
                    brier_score=brier_val,
                    insufficient_samples=insufficient,
                )
            )

        mean_base_auprc = float(np.mean(base_auprcs)) if base_auprcs else 0.0
        mean_tgt_auprc = float(np.mean(tgt_auprcs)) if tgt_auprcs else 0.0

        return LabelDriftReport(
            total_snapshots=len(snapshots),
            baseline_window=f"Snapshots {baseline_snapshots[0]}-{baseline_snapshots[1]}",
            target_window=f"Snapshots {target_snapshots[0]}-{target_snapshots[1]}",
            baseline_illicit_prevalence=round(base_prev, 4),
            target_illicit_prevalence=round(tgt_prev, 4),
            prevalence_shift_delta=round(tgt_prev - base_prev, 4),
            baseline_mean_auprc=round(mean_base_auprc, 4),
            target_mean_auprc=round(mean_tgt_auprc, 4),
            auprc_shift_delta=round(mean_tgt_auprc - mean_base_auprc, 4),
            overall_unlabelled_count=total_unknown,
            overall_unlabelled_pct=round(overall_unlabelled_pct, 4),
            snapshots_analyzed=summaries,
            unknown_label_safeguard=self.UNKNOWN_SAFEGUARD,
        )
