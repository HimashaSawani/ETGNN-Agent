"""
Continual Learning and Model Promotion Framework.
"""

from __future__ import annotations

from src.ml.continual.candidate_trainer import (
    CandidateExperimentManifest,
    SafeCandidateTrainer,
)
from src.ml.continual.promotion_gate import (
    ModelMetrics,
    ModelPromotionGate,
    PromotionDecision,
    PromotionEvaluationResult,
)

__all__ = [
    "CandidateExperimentManifest",
    "SafeCandidateTrainer",
    "ModelMetrics",
    "ModelPromotionGate",
    "PromotionDecision",
    "PromotionEvaluationResult",
]

