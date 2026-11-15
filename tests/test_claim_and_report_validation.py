import pytest
from src.agent.validation.citation_validator import CitationValidator
from src.agent.validation.claim_validator import ClaimValidator, ClaimClassification
from src.agent.validation.report_validator import ReportValidator


def test_citation_validator_valid():
    """Verify valid citation IDs matching provenance chain are accepted."""
    validator = CitationValidator(min_coverage=0.80)
    provenance = ["EV-TX-100", "EV-PRED-ETGNN-100", "EV-SUB-100-2HOP"]
    cited = ["EV-TX-100", "EV-PRED-ETGNN-100"]

    res = validator.validate_citations(cited, provenance)
    assert res.is_valid is True
    assert res.citation_coverage == 1.0


def test_citation_validator_invalid_and_missing():
    """Verify malformed and missing IDs are rejected."""
    validator = CitationValidator()
    provenance = ["EV-TX-100"]
    cited = ["EV-TX-100", "INVALID_FORMAT_ID", "EV-PRED-UNKNOWN-999"]

    res = validator.validate_citations(cited, provenance)
    assert res.is_valid is False
    assert "INVALID_FORMAT_ID" in res.invalid_citations
    assert "EV-PRED-UNKNOWN-999" in res.missing_evidence_ids


def test_claim_validator_prohibited_certainty():
    """Verify claims with definitive guilt or conviction phrases are marked UNSUPPORTED."""
    validator = ClaimValidator()
    bundle = {"provenance_chain": ["EV-TX-100"]}

    finding = "This transaction proves guilt and convicts the user as a confirmed money launderer."
    rec = validator.validate_claim(finding, "EV-TX-100", bundle)
    assert rec.classification == ClaimClassification.UNSUPPORTED
    assert any("prohibited" in r.lower() for r in rec.reasons)


def test_claim_validator_hallucinated_features():
    """Verify claims with invented feature semantics are rejected."""
    validator = ClaimValidator()
    bundle = {"provenance_chain": ["EV-TX-100"]}

    finding = "The wallet_balance and mixer_fee indicate high risk."
    rec = validator.validate_claim(finding, "EV-TX-100", bundle)
    assert rec.classification == ClaimClassification.UNSUPPORTED
    assert any("hallucinated" in r.lower() for r in rec.reasons)


def test_claim_validator_numerical_drift():
    """Verify claims with numerical probabilities conflicting with evidence are flagged."""
    validator = ClaimValidator(numerical_tolerance=0.05)
    bundle = {
        "provenance_chain": ["EV-PRED-ETGNN-100"],
        "predictions": {
            "etgnn": {"illicit_probability": 0.20, "evidence_id": "EV-PRED-ETGNN-100"}
        }
    }

    # Claim says 85% illicit, but evidence has 0.20 (20%)
    finding = "ETGNN predicted 85% probability of fraud."
    rec = validator.validate_claim(finding, "EV-PRED-ETGNN-100", bundle)
    assert rec.classification == ClaimClassification.UNSUPPORTED
    assert any("differs from" in r.lower() for r in rec.reasons)


def test_report_validator_snapshot43_qualification():
    """Strict contract: missing Snapshot 43 qualification invalidates the original report.

    The validator may still append qualification text to the sanitized report, but that
    auto-appended text must not retroactively make the original report valid.
    """
    validator = ReportValidator()
    bundle = {
        "provenance_chain": ["EV-TX-100"],
        # Explicit, verified non-isolated topology so only Snapshot 43 behavior is exercised.
        "transaction": {"transaction_id": 100, "in_degree": 2, "out_degree": 1, "snapshot_id": 43},
    }
    report = {
        "supported_findings": [],
        "limitations": ["Standard evaluation."]
    }

    res = validator.validate_report(report, bundle, snapshot_id=43)
    assert res.is_valid is False
    assert any("Snapshot 43" in r for r in res.reasons)
    # Qualification text is appended to the sanitized report only as remediation
    limitations_text = " ".join(report["limitations"]).lower()
    assert "snapshot 43" in limitations_text
    assert "observed structural distribution shift" in limitations_text
    assert "underlying cause has not been established" in limitations_text

