"""
Analyst Feedback Quality and Epistemic Governance Evaluator.
Analyzes analyst feedback in data/analyst_feedback.db while enforcing strict separation
between analyst opinions, factual corrections, and verified fraud ground-truth labels.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FeedbackQualityReport:
    total_feedback_records: int
    unique_investigations: int
    unique_transactions: int
    unique_reviewers: int
    mean_rating: float
    rating_distribution: Dict[int, int]
    category_distribution: Dict[str, int]
    disagreement_incidents: int
    citation_issue_frequency: float
    missing_evidence_frequency: float
    numerical_error_frequency: float
    scientific_safeguard: str
    provenance_classification: str


class AnalystFeedbackEvaluator:
    """Evaluates analyst feedback as an indicator of investigation report usability."""

    SCIENTIFIC_SAFEGUARD = (
        "Analyst feedback represents qualitative subjective evaluation of explanation clarity and utility. "
        "It MUST NEVER be treated as verified ground-truth fraud labels for model training or promotion."
    )

    def __init__(self, db_path: str = "data/analyst_feedback.db"):
        self.db_path = Path(db_path)

    def analyze_feedback(self) -> FeedbackQualityReport:
        """Reads records from the feedback database and aggregates governance metrics."""
        if not self.db_path.exists():
            return FeedbackQualityReport(
                total_feedback_records=0,
                unique_investigations=0,
                unique_transactions=0,
                unique_reviewers=0,
                mean_rating=0.0,
                rating_distribution={},
                category_distribution={},
                disagreement_incidents=0,
                citation_issue_frequency=0.0,
                missing_evidence_frequency=0.0,
                numerical_error_frequency=0.0,
                scientific_safeguard=self.SCIENTIFIC_SAFEGUARD,
                provenance_classification="ANALYST_OPINION_ONLY",
            )

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT feedback_id, investigation_id, transaction_id, feedback_category, rating, reviewer_id FROM analyst_feedback"
        )
        rows = cursor.fetchall()
        conn.close()

        total = len(rows)
        if total == 0:
            return FeedbackQualityReport(
                total_feedback_records=0,
                unique_investigations=0,
                unique_transactions=0,
                unique_reviewers=0,
                mean_rating=0.0,
                rating_distribution={},
                category_distribution={},
                disagreement_incidents=0,
                citation_issue_frequency=0.0,
                missing_evidence_frequency=0.0,
                numerical_error_frequency=0.0,
                scientific_safeguard=self.SCIENTIFIC_SAFEGUARD,
                provenance_classification="ANALYST_OPINION_ONLY",
            )

        inv_ids = set()
        tx_ids = set()
        reviewers = set()
        ratings = []
        rating_counts: Dict[int, int] = {i: 0 for i in range(1, 6)}
        cat_counts: Dict[str, int] = {}
        tx_ratings: Dict[int, List[int]] = {}

        for _, inv_id, tx_id, cat, rating, reviewer in rows:
            inv_ids.add(inv_id)
            tx_ids.add(tx_id)
            reviewers.add(reviewer)
            ratings.append(rating)
            rating_counts[int(rating)] = rating_counts.get(int(rating), 0) + 1
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

            if tx_id not in tx_ratings:
                tx_ratings[tx_id] = []
            tx_ratings[tx_id].append(int(rating))

        # Check for disagreement (variance in rating on same transaction)
        disagreement_count = sum(1 for tx, r_list in tx_ratings.items() if len(set(r_list)) > 1)

        mean_rat = sum(ratings) / max(1, total)
        cit_issues = cat_counts.get("citation_error", 0) / max(1, total)
        missing_ev = cat_counts.get("missing_evidence", 0) / max(1, total)
        num_err = cat_counts.get("numerical_error", 0) / max(1, total)

        return FeedbackQualityReport(
            total_feedback_records=total,
            unique_investigations=len(inv_ids),
            unique_transactions=len(tx_ids),
            unique_reviewers=len(reviewers),
            mean_rating=round(mean_rat, 2),
            rating_distribution=rating_counts,
            category_distribution=cat_counts,
            disagreement_incidents=disagreement_count,
            citation_issue_frequency=round(cit_issues, 4),
            missing_evidence_frequency=round(missing_ev, 4),
            numerical_error_frequency=round(num_err, 4),
            scientific_safeguard=self.SCIENTIFIC_SAFEGUARD,
            provenance_classification="ANALYST_OPINION_ONLY",
        )

