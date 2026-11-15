"""
Phase 4.5 Test Suite: Provider Reliability, Snapshot 43 Integrity, Topology Semantics,
Feedback Security, and Report Export Verification.
"""

import hashlib
import json
import os
import sqlite3
import pytest

from src.agent.llm.provider import (
    get_llm_provider,
    MockLLMProvider,
    EvidenceOnlyProvider,
    LLMAuthenticationError,
    LLMProviderError,
)
from src.agent.llm.prompts import SYSTEM_PROMPT_FORENSIC_INVESTIGATOR
from src.agent.feedback.store import (
    AnalystFeedbackStore,
    FeedbackSubmission,
)
from src.agent.reports.pdf_exporter import generate_investigation_pdf
from src.agent.validation.report_validator import ReportValidator
from src.agent.orchestration.nodes import generate_explanation_node


# ---------------------------------------------------------------------------
# 1. Provider Reliability & Fallback Tests
# ---------------------------------------------------------------------------

def test_missing_gemini_credentials_raises_authentication_error():
    """Unconfigured Gemini must raise typed LLMAuthenticationError, never silent mock fallback."""
    original_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    try:
        with pytest.raises(LLMAuthenticationError) as exc_info:
            get_llm_provider("gemini")
        assert "GEMINI_API_KEY" in str(exc_info.value)
    finally:
        if original_key is not None:
            os.environ["GEMINI_API_KEY"] = original_key


def test_missing_claude_credentials_raises_authentication_error():
    """Unconfigured Anthropic must raise typed LLMAuthenticationError, never silent mock fallback."""
    original_key = os.environ.get("ANTHROPIC_API_KEY")
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    try:
        with pytest.raises(LLMAuthenticationError) as exc_info:
            get_llm_provider("anthropic")
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)
    finally:
        if original_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = original_key


def test_unsupported_provider_raises_provider_error():
    """Unsupported provider names must raise LLMProviderError."""
    with pytest.raises(LLMProviderError) as exc_info:
        get_llm_provider("unsupported_model_provider_xyz")
    assert "Unsupported or unconfigured LLM provider" in str(exc_info.value)


def test_explicit_mock_mode_metadata():
    """Explicit mock mode must declare simulated non-live metadata."""
    provider = get_llm_provider("mock")
    assert isinstance(provider, MockLLMProvider)

    raw_json = provider.generate(
        prompt="Test prompt",
        system_prompt=SYSTEM_PROMPT_FORENSIC_INVESTIGATOR,
        context_data={"tx_id": 230425980, "snapshot_id": 43},
    )
    parsed = json.loads(raw_json)
    assert parsed["generation_mode"] == "SIMULATED"
    assert parsed["live_llm_verified"] is False
    assert parsed["provider_type"] == "mock"


def test_generate_explanation_node_strict_live_failure():
    """Live provider failure with allow_evidence_only_fallback=False must raise, not switch to mock."""
    state = {
        "investigation_id": "inv_test_strict_fail",
        "tx_id": 230425980,
        "as_of_snapshot": 43,
        "evidence_mode": "RETROSPECTIVE_INVESTIGATION",
        "llm_provider": "gemini",
        "allow_evidence_only_fallback": False,
        "evidence_bundle": {"transaction": {"snapshot_id": 43, "in_degree": 0, "out_degree": 2}},
        "provenance_chain": ["EV-TX-LOOKUP-230425980"],
    }

    original_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    try:
        with pytest.raises(LLMAuthenticationError):
            generate_explanation_node(state)
    finally:
        if original_key is not None:
            os.environ["GEMINI_API_KEY"] = original_key


def test_generate_explanation_node_evidence_only_fallback():
    """Live provider failure with allow_evidence_only_fallback=True must route to DETERMINISTIC_EVIDENCE_ONLY."""
    state = {
        "investigation_id": "inv_test_evidence_fallback",
        "tx_id": 230425980,
        "as_of_snapshot": 43,
        "evidence_mode": "RETROSPECTIVE_INVESTIGATION",
        "llm_provider": "gemini",
        "allow_evidence_only_fallback": True,
        "evidence_bundle": {"transaction": {"snapshot_id": 43, "in_degree": 0, "out_degree": 2}},
        "provenance_chain": ["EV-TX-LOOKUP-230425980"],
    }

    original_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    try:
        new_state = generate_explanation_node(state)
        report = new_state.get("raw_explanation")
        assert report is not None
        assert report["generation_mode"] == "DETERMINISTIC_EVIDENCE_ONLY"
        assert report["live_llm_verified"] is False
        assert report["provider_type"] == "evidence_only"
        limitations_text = " ".join(report.get("limitations", []))
        assert "fallen back to deterministic evidence-only mode" in limitations_text
    finally:
        if original_key is not None:
            os.environ["GEMINI_API_KEY"] = original_key


# ---------------------------------------------------------------------------
# 2. Snapshot 43 Scientific Integrity Tests
# ---------------------------------------------------------------------------

def test_snapshot_43_prompt_template_integrity():
    """System prompt must avoid asserting unproven causality for Snapshot 43."""
    forbidden_terms = ["caused by Silk Road", "caused by Hydra", "darknet shutdown caused", "market closure caused"]
    for term in forbidden_terms:
        assert term.lower() not in SYSTEM_PROMPT_FORENSIC_INVESTIGATOR.lower(), f"Forbidden causal claim found: {term}"

    # Must contain scientifically defensible wording
    assert "underlying cause has not been established" in SYSTEM_PROMPT_FORENSIC_INVESTIGATOR


def test_mock_provider_snapshot_43_qualification():
    """Mock provider limitations must qualify Snapshot 43 as an observed shift without causality."""
    provider = MockLLMProvider()
    raw_json = provider.generate(
        prompt="Investigate transaction at snapshot 43",
        system_prompt=SYSTEM_PROMPT_FORENSIC_INVESTIGATOR,
        context_data={"tx_id": 288717023, "snapshot_id": 43, "evidence_bundle": {"transaction": {"in_degree": 0, "out_degree": 2}}},
    )
    parsed = json.loads(raw_json)
    limitations = " ".join(parsed.get("evidence_limitations", []))
    assert "observed structural distribution shift" in limitations
    assert "underlying cause has not been established" in limitations
    assert "caused by silk road" not in limitations.lower()


# ---------------------------------------------------------------------------
# 3. Directed Graph Topology Semantics Tests
# ---------------------------------------------------------------------------

def test_graph_topology_degree_distinctions():
    """ReportValidator and Mock provider must distinguish 0 in-degree vs 0 out-degree vs isolated."""
    validator = ReportValidator()

    # Case 1: 0 in-degree, 2 out-degree
    res_in0 = validator.validate_report(
        report_dict={"supported_findings": [], "limitations": []},
        evidence_bundle={"transaction": {"in_degree": 0, "out_degree": 2}, "provenance_chain": []},
        snapshot_id=43,
        target_tx_id=288717023,
    )
    lims_in0 = " ".join(res_in0.sanitized_report["limitations"])
    assert "absence of observed upstream edges in the dataset sample" in lims_in0
    assert "not proof that the transaction originated funds" in lims_in0

    # Case 2: 2 in-degree, 0 out-degree
    res_out0 = validator.validate_report(
        report_dict={"supported_findings": [], "limitations": []},
        evidence_bundle={"transaction": {"in_degree": 2, "out_degree": 0}, "provenance_chain": []},
        snapshot_id=10,
        target_tx_id=230425980,
    )
    lims_out0 = " ".join(res_out0.sanitized_report["limitations"])
    assert "absence of observed downstream spending edges" in lims_out0
    assert "not proof of unspent funds or final accumulation" in lims_out0

    # Case 3: Fully isolated (0 in, 0 out)
    res_iso = validator.validate_report(
        report_dict={"supported_findings": [], "limitations": []},
        evidence_bundle={"transaction": {"in_degree": 0, "out_degree": 0}, "provenance_chain": []},
        snapshot_id=10,
        target_tx_id=230425980,
    )
    lims_iso = " ".join(res_iso.sanitized_report["limitations"])
    assert "fully isolated in snapshot graph (both in-degree and out-degree are 0)" in lims_iso


# ---------------------------------------------------------------------------
# 4. Analyst Feedback Store Security & SQLite WAL Tests
# ---------------------------------------------------------------------------

def test_analyst_feedback_store_wal_mode_enabled(tmp_path):
    """AnalystFeedbackStore must enable WAL mode for concurrency and isolation."""
    db_file = tmp_path / "test_feedback.db"
    store = AnalystFeedbackStore(db_path=str(db_file))

    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"


def test_analyst_feedback_store_input_validation(tmp_path):
    """AnalystFeedbackStore must validate categories, sanitize comments, and enforce rating ranges."""
    db_file = tmp_path / "test_feedback.db"
    store = AnalystFeedbackStore(db_path=str(db_file))

    # Invalid rating (> 5)
    with pytest.raises(ValueError):
        FeedbackSubmission(
            investigation_id="inv_test",
            transaction_id=123,
            feedback_category="helpful",
            rating=6,
        )

    # Invalid rating (< 1)
    with pytest.raises(ValueError):
        FeedbackSubmission(
            investigation_id="inv_test",
            transaction_id=123,
            feedback_category="helpful",
            rating=0,
        )

    # Invalid category
    with pytest.raises(ValueError):
        FeedbackSubmission(
            investigation_id="inv_test",
            transaction_id=123,
            feedback_category="malicious_category_injection",
            rating=4,
        )

    # Valid submission with null-byte stripping
    valid_sub = FeedbackSubmission(
        investigation_id="inv_test",
        transaction_id=123,
        feedback_category="citation_issue",
        rating=4,
        comment="Citation verified.\x00",
        reviewer_id="analyst_qa",
    )
    rec = store.submit_feedback(valid_sub)
    assert rec.comment == "Citation verified."
    assert rec.reviewer_id == "analyst_qa"


def test_analyst_feedback_checkpoint_isolation(tmp_path):
    """Feedback store writes must be completely isolated from model checkpoints and raw Elliptic data."""
    db_file = tmp_path / "test_feedback.db"
    store = AnalystFeedbackStore(db_path=str(db_file))

    sub = FeedbackSubmission(
        investigation_id="inv_iso",
        transaction_id=230425980,
        feedback_category="helpful",
        rating=5,
    )
    store.submit_feedback(sub)

    assert "models/checkpoints" not in str(store.db_path)
    assert "data/raw" not in str(store.db_path)


# ---------------------------------------------------------------------------
# 5. Deterministic PDF Export Checksum Tests
# ---------------------------------------------------------------------------

def test_pdf_export_deterministic_sha256_checksum():
    """PDF export must generate valid bytes and compute deterministic canonical SHA256."""
    mock_report = {
        "investigation_id": "inv_pdf_test",
        "target_transaction_id": 230425980,
        "snapshot_id": 43,
        "status": "COMPLETED",
        "consensus_verdict": "HIGH_RISK_ILLICIT",
        "predicted_class": 1,
        "model_probability": 0.885,
        "generation_mode": "SIMULATED",
        "provider_type": "mock",
        "live_llm_verified": False,
        "narrative_summary": "Test narrative summary for deterministic SHA256 checksum.",
        "supported_findings": [],
        "uncertain_findings": [],
        "limitations": [
            "Snapshot 43 Observation: An observed structural distribution shift occurs around Snapshot 43.",
        ],
        "research_citations": ["Weber et al. (2019)"],
        "provenance_chain": ["EV-TX-LOOKUP-230425980"],
    }

    pdf_bytes = generate_investigation_pdf(mock_report)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")

    canonical_repr = json.dumps({
        "investigation_id": mock_report["investigation_id"],
        "target_transaction_id": mock_report["target_transaction_id"],
        "snapshot_id": mock_report["snapshot_id"],
        "consensus_verdict": mock_report["consensus_verdict"],
        "model_probability": mock_report["model_probability"],
        "provenance_chain": mock_report["provenance_chain"],
    }, sort_keys=True)
    expected_hash = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()
    assert len(expected_hash) == 64
