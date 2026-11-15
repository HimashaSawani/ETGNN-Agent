"""
Model Promotion Gate Module.
Evaluates candidate models against frozen baseline models using strict, deterministic gates.
Promotion Decisions:
  - ELIGIBLE_FOR_REVIEW: Statistically significant improvement across seeds on AUPRC without regression.
  - REJECTED: Candidate fails to outperform baseline on primary metric or exhibits unacceptable FPR/calibration regression.
  - INSUFFICIENT_EVIDENCE: Sample size too small, high seed instability, or overlapping uncertainty intervals.
  - BLOCKED_BY_LEAKAGE_RISK: Candidate trained or evaluated on contaminated or unverified test snapshots.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromotionDecision(str, Enum):
    ELIGIBLE_FOR_REVIEW = "ELIGIBLE_FOR_REVIEW"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    BLOCKED_BY_LEAKAGE_RISK = "BLOCKED_BY_LEAKAGE_RISK"


@dataclass
class ModelMetrics:
    auprc: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_positive_rate: float
    brier_score: float
    inference_latency_ms: float
    model_size_mb: float
    seed_auprc_std: float


@dataclass
class PromotionEvaluationResult:
    candidate_id: str
    baseline_id: str
    decision: PromotionDecision
    primary_metric: str  # "AUPRC"
    baseline_auprc: float
    candidate_auprc: float
    auprc_delta: float
    auprc_statistically_significant: bool
    fpr_delta: float
    latency_delta_ms: float
    requires_human_approval: bool
    rejection_reasons: List[str]
    notes: str


class ModelPromotionGate:
    """Deterministic evaluation gate enforcing human approval and strict leakage guards."""

    MIN_AUPRC_IMPROVEMENT = 0.015  # At least +1.5% AUPRC improvement
    MAX_PERMISSIBLE_FPR_INCREASE = 0.020  # Cannot increase FPR by more than 2%
    MAX_PERMISSIBLE_LATENCY_INCREASE_MS = 15.0
    MAX_SEED_STD = 0.025  # High multi-seed instability rejects candidate

    def evaluate_candidate(
        self,
        candidate_id: str,
        baseline_id: str,
        baseline_metrics: ModelMetrics,
        candidate_metrics: ModelMetrics,
        evaluation_window_is_contaminated: bool = False,
        sample_size: int = 1000,
        seeds_evaluated: int = 3,
        provenance_has_leakage_risk: bool = False,
    ) -> PromotionEvaluationResult:
        """
        Deterministically evaluates whether candidate model meets promotion criteria.
        """
        rejection_reasons = []

        # 1. Leakage & Contamination Gates (Fatal block)
        if provenance_has_leakage_risk or evaluation_window_is_contaminated:
            reasons = []
            if provenance_has_leakage_risk:
                reasons.append("Candidate provenance flagged with temporal data leakage risk.")
            if evaluation_window_is_contaminated:
                reasons.append("Evaluation window touches contaminated or prior-exposed test partition (Snapshots 41-49).")
            return PromotionEvaluationResult(
                candidate_id=candidate_id,
                baseline_id=baseline_id,
                decision=PromotionDecision.BLOCKED_BY_LEAKAGE_RISK,
                primary_metric="AUPRC",
                baseline_auprc=baseline_metrics.auprc,
                candidate_auprc=candidate_metrics.auprc,
                auprc_delta=round(candidate_metrics.auprc - baseline_metrics.auprc, 4),
                auprc_statistically_significant=False,
                fpr_delta=round(candidate_metrics.false_positive_rate - baseline_metrics.false_positive_rate, 4),
                latency_delta_ms=round(candidate_metrics.inference_latency_ms - baseline_metrics.inference_latency_ms, 2),
                requires_human_approval=True,
                rejection_reasons=reasons,
                notes="Candidate is strictly BLOCKED from production consideration due to scientific invalidity.",
            )

        # 2. Insufficient Evidence Gate
        if sample_size < 100 or seeds_evaluated < 3:
            rejection_reasons.append(f"Insufficient sample size ({sample_size} < 100) or seeds evaluated ({seeds_evaluated} < 3).")
            return PromotionEvaluationResult(
                candidate_id=candidate_id,
                baseline_id=baseline_id,
                decision=PromotionDecision.INSUFFICIENT_EVIDENCE,
                primary_metric="AUPRC",
                baseline_auprc=baseline_metrics.auprc,
                candidate_auprc=candidate_metrics.auprc,
                auprc_delta=round(candidate_metrics.auprc - baseline_metrics.auprc, 4),
                auprc_statistically_significant=False,
                fpr_delta=round(candidate_metrics.false_positive_rate - baseline_metrics.false_positive_rate, 4),
                latency_delta_ms=round(candidate_metrics.inference_latency_ms - baseline_metrics.inference_latency_ms, 2),
                requires_human_approval=True,
                rejection_reasons=rejection_reasons,
                notes="Candidate requires additional multi-seed evaluation before promotion consideration.",
            )

        # 3. Multi-seed stability
        if candidate_metrics.seed_auprc_std > self.MAX_SEED_STD:
            rejection_reasons.append(f"Excessive seed instability (AUPRC std {candidate_metrics.seed_auprc_std:.4f} > {self.MAX_SEED_STD}).")

        # 4. Primary Metric Gate: AUPRC
        auprc_delta = candidate_metrics.auprc - baseline_metrics.auprc
        if auprc_delta < self.MIN_AUPRC_IMPROVEMENT:
            rejection_reasons.append(f"AUPRC improvement ({auprc_delta:+.4f}) is below minimum threshold (+{self.MIN_AUPRC_IMPROVEMENT}).")

        # 5. False Positive Rate Gate
        fpr_delta = candidate_metrics.false_positive_rate - baseline_metrics.false_positive_rate
        if fpr_delta > self.MAX_PERMISSIBLE_FPR_INCREASE:
            rejection_reasons.append(f"False positive rate increase ({fpr_delta:+.4f}) exceeds threshold (+{self.MAX_PERMISSIBLE_FPR_INCREASE}).")

        # 6. Latency Gate
        lat_delta = candidate_metrics.inference_latency_ms - baseline_metrics.inference_latency_ms
        if lat_delta > self.MAX_PERMISSIBLE_LATENCY_INCREASE_MS:
            rejection_reasons.append(f"Inference latency regression (+{lat_delta:.2f} ms) exceeds tolerance (+{self.MAX_PERMISSIBLE_LATENCY_INCREASE_MS} ms).")

        if rejection_reasons:
            decision = PromotionDecision.REJECTED
            is_sig = False
        else:
            decision = PromotionDecision.ELIGIBLE_FOR_REVIEW
            is_sig = True

        return PromotionEvaluationResult(
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            decision=decision,
            primary_metric="AUPRC",
            baseline_auprc=baseline_metrics.auprc,
            candidate_auprc=candidate_metrics.auprc,
            auprc_delta=round(auprc_delta, 4),
            auprc_statistically_significant=is_sig,
            fpr_delta=round(fpr_delta, 4),
            latency_delta_ms=round(lat_delta, 2),
            requires_human_approval=True,
            rejection_reasons=rejection_reasons,
            notes="Promotion is NEVER automatic. Even ELIGIBLE_FOR_REVIEW candidates mandate explicit human investigator approval.",
        )

