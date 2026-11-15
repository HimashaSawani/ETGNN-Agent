"""
Analyst Feedback Storage and Isolation Service.
Provides persistent SQLite storage for human review feedback on AI forensic investigations.
Strict Safety Invariant: Completely isolated from ML checkpoints, raw datasets, and training labels.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
from pydantic import BaseModel, Field, field_validator


DEFAULT_DB_PATH = Path("data/analyst_feedback.db")

ALLOWED_CATEGORIES = {
    "helpful",
    "unhelpful",
    "citation_issue",
    "missing_evidence",
    "incorrect_numerical",
    "needs_further_review",
}


class FeedbackSubmission(BaseModel):
    investigation_id: str = Field(..., min_length=3, max_length=100)
    transaction_id: int = Field(..., ge=0)
    feedback_category: str = Field(
        ...,
        description="Category: helpful, unhelpful, citation_issue, missing_evidence, incorrect_numerical, needs_further_review"
    )
    rating: int = Field(default=3, ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=2000)
    reviewer_id: str = Field(default="analyst_local", max_length=50)
    report_version: str = Field(default="ETGNN-Agent-v3.5.0", max_length=50)
    evidence_references: List[str] = Field(default_factory=list)

    @field_validator("feedback_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ALLOWED_CATEGORIES:
            raise ValueError(f"Invalid feedback category '{v}'. Allowed: {sorted(ALLOWED_CATEGORIES)}")
        return clean

    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer_id(cls, v: str) -> str:
        clean = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-\.]{1,50}$", clean):
            raise ValueError("reviewer_id must be 1-50 alphanumeric characters (or _, -, .)")
        return clean

    @field_validator("comment")
    @classmethod
    def sanitize_comment(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Strip null bytes and non-printable control chars
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", v).strip()
        return sanitized[:2000]

    @field_validator("evidence_references")
    @classmethod
    def validate_evidence_refs(cls, v: List[str]) -> List[str]:
        if len(v) > 50:
            raise ValueError("Maximum 50 evidence references allowed per submission")
        return [str(ref).strip()[:100] for ref in v if str(ref).strip()]


class FeedbackRecord(BaseModel):
    feedback_id: str
    investigation_id: str
    transaction_id: int
    feedback_category: str
    rating: int
    comment: Optional[str] = None
    reviewer_id: str
    report_version: str
    evidence_references: List[str] = Field(default_factory=list)
    created_at: str


class AnalystFeedbackStore:
    """
    Thread-safe SQLite storage for human analyst review feedback.
    Enforces isolation from model weights, checkpoints, and raw Elliptic data.
    """

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analyst_feedback (
                        feedback_id TEXT PRIMARY KEY,
                        investigation_id TEXT NOT NULL,
                        transaction_id INTEGER NOT NULL,
                        feedback_category TEXT NOT NULL,
                        rating INTEGER NOT NULL,
                        comment TEXT,
                        reviewer_id TEXT NOT NULL,
                        report_version TEXT NOT NULL,
                        evidence_references TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_inv_id 
                    ON analyst_feedback(investigation_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feedback_tx_id 
                    ON analyst_feedback(transaction_id)
                """)
                conn.commit()

    def submit_feedback(self, submission: FeedbackSubmission) -> FeedbackRecord:
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        evidence_str = ",".join(submission.evidence_references)

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO analyst_feedback (
                        feedback_id, investigation_id, transaction_id,
                        feedback_category, rating, comment, reviewer_id,
                        report_version, evidence_references, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback_id,
                        submission.investigation_id,
                        submission.transaction_id,
                        submission.feedback_category,
                        submission.rating,
                        submission.comment,
                        submission.reviewer_id,
                        submission.report_version,
                        evidence_str,
                        now_iso,
                    ),
                )
                conn.commit()

        return FeedbackRecord(
            feedback_id=feedback_id,
            investigation_id=submission.investigation_id,
            transaction_id=submission.transaction_id,
            feedback_category=submission.feedback_category,
            rating=submission.rating,
            comment=submission.comment,
            reviewer_id=submission.reviewer_id,
            report_version=submission.report_version,
            evidence_references=submission.evidence_references,
            created_at=now_iso,
        )

    def get_feedback_for_investigation(self, investigation_id: str) -> List[FeedbackRecord]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT feedback_id, investigation_id, transaction_id,
                           feedback_category, rating, comment, reviewer_id,
                           report_version, evidence_references, created_at
                    FROM analyst_feedback
                    WHERE investigation_id = ?
                    ORDER BY created_at DESC
                    """,
                    (investigation_id,),
                )
                rows = cursor.fetchall()

        results: List[FeedbackRecord] = []
        for r in rows:
            ev_refs = [x.strip() for x in r["evidence_references"].split(",") if x.strip()] if r["evidence_references"] else []
            results.append(
                FeedbackRecord(
                    feedback_id=r["feedback_id"],
                    investigation_id=r["investigation_id"],
                    transaction_id=r["transaction_id"],
                    feedback_category=r["feedback_category"],
                    rating=r["rating"],
                    comment=r["comment"],
                    reviewer_id=r["reviewer_id"],
                    report_version=r["report_version"],
                    evidence_references=ev_refs,
                    created_at=r["created_at"],
                )
            )
        return results

    def list_all_feedback(self, limit: int = 50, offset: int = 0) -> List[FeedbackRecord]:
        actual_limit = int(getattr(limit, "default", limit))
        actual_offset = int(getattr(offset, "default", offset))
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT feedback_id, investigation_id, transaction_id,
                           feedback_category, rating, comment, reviewer_id,
                           report_version, evidence_references, created_at
                    FROM analyst_feedback
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (actual_limit, actual_offset),
                )
                rows = cursor.fetchall()

        results: List[FeedbackRecord] = []
        for r in rows:
            ev_refs = [x.strip() for x in r["evidence_references"].split(",") if x.strip()] if r["evidence_references"] else []
            results.append(
                FeedbackRecord(
                    feedback_id=r["feedback_id"],
                    investigation_id=r["investigation_id"],
                    transaction_id=r["transaction_id"],
                    feedback_category=r["feedback_category"],
                    rating=r["rating"],
                    comment=r["comment"],
                    reviewer_id=r["reviewer_id"],
                    report_version=r["report_version"],
                    evidence_references=ev_refs,
                    created_at=r["created_at"],
                )
            )
        return results

    def get_feedback_stats(self) -> Dict[str, Any]:
        with self._lock:
            with self._get_connection() as conn:
                total_count = conn.execute("SELECT COUNT(*) FROM analyst_feedback").fetchone()[0]
                avg_rating = conn.execute("SELECT AVG(rating) FROM analyst_feedback").fetchone()[0] or 0.0
                categories_cur = conn.execute(
                    "SELECT feedback_category, COUNT(*) as cnt FROM analyst_feedback GROUP BY feedback_category"
                )
                categories = {row["feedback_category"]: row["cnt"] for row in categories_cur.fetchall()}

        return {
            "total_feedback_count": total_count,
            "average_rating": round(float(avg_rating), 2),
            "category_distribution": categories,
            "storage_engine": "SQLite (Isolated)",
            "safety_invariant": "Feedback isolated from model weights and training labels",
        }


# Singleton instance
_store_instance: Optional[AnalystFeedbackStore] = None


def get_feedback_store() -> AnalystFeedbackStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = AnalystFeedbackStore()
    return _store_instance
