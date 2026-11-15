"""
Phase 6 Application Security & Input Validation Tests.
Validates defenses against SQL injection, path traversal, identifier overflow,
feature dimension tampering, and malicious parameter injections.
"""

import math
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.agent.feedback.store import AnalystFeedbackStore, FeedbackSubmission
from src.web.app import (
    PredictRequest,
    api_export_investigation_pdf,
    api_get_agent_graph,
    api_get_agent_transaction,
    get_transaction,
    get_transactions,
)


def test_sql_injection_defense_in_feedback(tmp_path):
    """Test that SQLite feedback store parameterized queries neutralize SQL injection payloads."""
    db_file = tmp_path / "test_feedback_sec.db"
    store = AnalystFeedbackStore(db_path=db_file)

    malicious_payload = "'; DROP TABLE analyst_feedback; --"
    submission = FeedbackSubmission(
        investigation_id="inv_test_sec_01",
        transaction_id=288717023,
        feedback_category="helpful",
        rating=5,
        comment=malicious_payload,
        reviewer_id="analyst_test",
    )

    record = store.submit_feedback(submission)
    assert record.feedback_id.startswith("fb_")
    assert record.comment == malicious_payload

    # Verify table was NOT dropped and record is queryable
    items = store.get_feedback_for_investigation("inv_test_sec_01")
    assert len(items) == 1
    assert items[0].comment == malicious_payload

    # Direct query to ensure table still exists
    stats = store.get_feedback_stats()
    assert stats["total_feedback_count"] == 1


def test_predict_request_transaction_id_bounds():
    """Verify transaction ID must be between 0 and 2^63 - 1."""
    with pytest.raises(ValidationError):
        PredictRequest(transaction_id=-1)

    with pytest.raises(ValidationError):
        PredictRequest(transaction_id=2**64)

    req = PredictRequest(transaction_id=288717023)
    assert req.transaction_id == 288717023


def test_predict_request_feature_override_dimension_and_finite_checks():
    """Verify feature overrides cannot exceed 165 dims or contain NaN/Inf."""
    # Exceeding 165 features
    oversized = {f"feat_{i}": 0.5 for i in range(166)}
    with pytest.raises(ValueError, match="cannot exceed 165"):
        PredictRequest(transaction_id=1, features_override=oversized)

    # NaN / Inf protection
    with pytest.raises(ValueError, match="must be a finite float"):
        PredictRequest(transaction_id=1, features_override={"feat_0": float("nan")})

    with pytest.raises(ValueError, match="must be a finite float"):
        PredictRequest(transaction_id=1, features_override={"feat_0": float("inf")})


def test_path_traversal_rejection_in_pdf_export():
    """Verify path traversal sequences in investigation_id are strictly rejected."""
    traversal_ids = [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "inv/../../../secret",
        "inv%2e%2e%2fetc",
    ]
    for tid in traversal_ids:
        with pytest.raises(HTTPException) as exc_info:
            api_export_investigation_pdf(investigation_id=tid)
        assert exc_info.value.status_code == 400


def test_feedback_reviewer_id_sanitization():
    """Verify reviewer_id rejects shell or injection characters."""
    with pytest.raises(ValidationError):
        FeedbackSubmission(
            investigation_id="inv_test",
            transaction_id=1,
            feedback_category="helpful",
            reviewer_id="analyst; rm -rf /",
        )


def test_feedback_category_whitelist():
    """Verify unknown or malicious feedback categories are rejected."""
    with pytest.raises(ValidationError):
        FeedbackSubmission(
            investigation_id="inv_test",
            transaction_id=1,
            feedback_category="<script>alert(1)</script>",
        )


def test_spa_path_traversal_protection():
    """Verify SPA route protection against directory escape."""
    from src.web.app import DIST_DIR, serve_spa
    if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
        with pytest.raises(HTTPException) as exc_info:
            serve_spa("../../../etc/passwd")
        assert exc_info.value.status_code == 403
