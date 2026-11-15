"""
Temporal Feature Drift Detection Module.
Computes Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test,
Wasserstein distance, and Jensen-Shannon divergence across chronological windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats
import scipy.spatial.distance


@dataclass
class SingleFeatureDriftResult:
    feature_index: int
    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    ks_significant_bonferroni: bool
    wasserstein_distance: float
    js_divergence: float
    drift_level: str  # "LOW", "MODERATE", "HIGH"


@dataclass
class WindowFeatureDriftReport:
    reference_window: str
    target_window: str
    reference_sample_size: int
    target_sample_size: int
    features_analyzed: int
    mean_psi: float
    median_psi: float
    max_psi: float
    high_drift_features_count: int
    moderate_drift_features_count: int
    low_drift_features_count: int
    top_drifting_features: List[SingleFeatureDriftResult]
    multiple_testing_correction: str
    notes: str


def calculate_psi(
    reference: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between reference and target distributions.
    Uses quantile-based binning on reference with Laplace/epsilon smoothing.
    """
    ref = reference[~np.isnan(reference)]
    tgt = target[~np.isnan(target)]
    if len(ref) < 10 or len(tgt) < 10:
        return 0.0

    # Quantile bin edges from reference
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    tgt_counts, _ = np.histogram(tgt, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * len(ref_counts))
    tgt_pct = (tgt_counts + epsilon) / (np.sum(tgt_counts) + epsilon * len(tgt_counts))

    psi = np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct))
    return float(max(0.0, psi))


def calculate_ks(reference: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Computes Kolmogorov-Smirnov statistic and p-value."""
    ref = reference[~np.isnan(reference)]
    tgt = target[~np.isnan(target)]
    if len(ref) < 5 or len(tgt) < 5:
        return 0.0, 1.0
    stat, pval = scipy.stats.ks_2samp(ref, tgt)
    return float(stat), float(pval)


def calculate_wasserstein(reference: np.ndarray, target: np.ndarray) -> float:
    """Computes Earth Mover's Distance / 1-Wasserstein distance."""
    ref = reference[~np.isnan(reference)]
    tgt = target[~np.isnan(target)]
    if len(ref) < 5 or len(tgt) < 5:
        return 0.0
    return float(scipy.stats.wasserstein_distance(ref, tgt))


def calculate_js_divergence(reference: np.ndarray, target: np.ndarray, num_bins: int = 20) -> float:
    """Computes Jensen-Shannon divergence between empirical distributions."""
    ref = reference[~np.isnan(reference)]
    tgt = target[~np.isnan(target)]
    if len(ref) < 10 or len(tgt) < 10:
        return 0.0

    min_val = min(float(ref.min()), float(tgt.min()))
    max_val = max(float(ref.max()), float(tgt.max()))
    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, num_bins + 1)
    ref_counts, _ = np.histogram(ref, bins=bins, density=True)
    tgt_counts, _ = np.histogram(tgt, bins=bins, density=True)

    epsilon = 1e-6
    p = (ref_counts + epsilon) / np.sum(ref_counts + epsilon)
    q = (tgt_counts + epsilon) / np.sum(tgt_counts + epsilon)

    return float(scipy.spatial.distance.jensenshannon(p, q))


class FeatureDriftDetector:
    """Detects and reports distribution shift across feature dimensions."""

    def __init__(self, significance_alpha: float = 0.01):
        self.significance_alpha = significance_alpha

    def analyze_window_drift(
        self,
        ref_matrix: np.ndarray,
        tgt_matrix: np.ndarray,
        feature_names: Optional[List[str]] = None,
        reference_window_name: str = "Training (Snapshots 1-32)",
        target_window_name: str = "Test (Snapshots 41-49)",
        max_features_to_report: int = 10,
    ) -> WindowFeatureDriftReport:
        """
        Analyzes drift for every feature column in ref_matrix vs tgt_matrix.
        """
        num_features = ref_matrix.shape[1]
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(num_features)]

        bonferroni_threshold = self.significance_alpha / max(1, num_features)

        results: List[SingleFeatureDriftResult] = []
        psi_values: List[float] = []

        for i in range(num_features):
            ref_col = ref_matrix[:, i]
            tgt_col = tgt_matrix[:, i]

            psi = calculate_psi(ref_col, tgt_col)
            ks_stat, ks_pval = calculate_ks(ref_col, tgt_col)
            wass = calculate_wasserstein(ref_col, tgt_col)
            js = calculate_js_divergence(ref_col, tgt_col)

            is_sig = ks_pval < bonferroni_threshold

            if psi >= 0.25:
                level = "HIGH"
            elif psi >= 0.10:
                level = "MODERATE"
            else:
                level = "LOW"

            psi_values.append(psi)
            results.append(
                SingleFeatureDriftResult(
                    feature_index=i,
                    feature_name=feature_names[i],
                    psi=round(psi, 4),
                    ks_statistic=round(ks_stat, 4),
                    ks_pvalue=float(ks_pval),
                    ks_significant_bonferroni=is_sig,
                    wasserstein_distance=round(wass, 4),
                    js_divergence=round(js, 4),
                    drift_level=level,
                )
            )

        # Sort by PSI descending
        results.sort(key=lambda x: x.psi, reverse=True)

        high_count = sum(1 for r in results if r.drift_level == "HIGH")
        mod_count = sum(1 for r in results if r.drift_level == "MODERATE")
        low_count = sum(1 for r in results if r.drift_level == "LOW")

        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        median_psi = float(np.median(psi_values)) if psi_values else 0.0
        max_psi = float(np.max(psi_values)) if psi_values else 0.0

        return WindowFeatureDriftReport(
            reference_window=reference_window_name,
            target_window=target_window_name,
            reference_sample_size=int(ref_matrix.shape[0]),
            target_sample_size=int(tgt_matrix.shape[0]),
            features_analyzed=num_features,
            mean_psi=round(mean_psi, 4),
            median_psi=round(median_psi, 4),
            max_psi=round(max_psi, 4),
            high_drift_features_count=high_count,
            moderate_drift_features_count=mod_count,
            low_drift_features_count=low_count,
            top_drifting_features=results[:max_features_to_report],
            multiple_testing_correction=f"Bonferroni adjusted alpha = {bonferroni_threshold:.6f}",
            notes="PSI thresholds: LOW (<0.10), MODERATE (0.10-0.25), HIGH (>=0.25). Arbitrary thresholds do not prove model degradation.",
        )
