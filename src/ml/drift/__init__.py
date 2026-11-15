"""
Temporal Drift Detection and Concept Shift Analysis Module.
"""

from __future__ import annotations

from src.ml.drift.feature_drift import (
    FeatureDriftDetector,
    SingleFeatureDriftResult,
    WindowFeatureDriftReport,
    calculate_js_divergence,
    calculate_ks,
    calculate_psi,
    calculate_wasserstein,
)
from src.ml.drift.prediction_drift import (
    PredictionDriftDetector,
    PredictionDriftReport,
    SnapshotPredictionSummary,
)
from src.ml.drift.graph_drift import (
    GraphTopologyDriftDetector,
    GraphTopologyDriftReport,
    SnapshotGraphTopologySummary,
)
from src.ml.drift.label_drift import (
    LabelDriftDetector,
    LabelDriftReport,
    SnapshotLabelSummary,
)
from src.ml.drift.drift_report import (
    ConsolidatedTemporalDriftReport,
    generate_and_save_drift_report,
)

__all__ = [
    "FeatureDriftDetector",
    "SingleFeatureDriftResult",
    "WindowFeatureDriftReport",
    "calculate_psi",
    "calculate_ks",
    "calculate_wasserstein",
    "calculate_js_divergence",
    "PredictionDriftDetector",
    "PredictionDriftReport",
    "SnapshotPredictionSummary",
    "GraphTopologyDriftDetector",
    "GraphTopologyDriftReport",
    "SnapshotGraphTopologySummary",
    "LabelDriftDetector",
    "LabelDriftReport",
    "SnapshotLabelSummary",
    "ConsolidatedTemporalDriftReport",
    "generate_and_save_drift_report",
]
