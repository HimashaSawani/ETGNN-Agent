import pytest
from src.agent.validation.claim_validator import ClaimValidator
from src.agent.validation.citation_validator import CitationValidator
from src.agent.validation.report_validator import ReportValidator


def test_metric_specific_numeric_tolerances():
    validator = ClaimValidator()
    bundle = {
        "transaction": type("Tx", (), {"transaction_id": 230425980, "in_degree": 2, "out_degree": 1, "snapshot_id": 5})(),
        "predictions": {
            "etgnn": type("Pred", (), {"illicit_probability": 0.8523, "predicted_class": 1, "evidence_id": "pred_etgnn_1"})(),
        },
        "metrics": [
            type("Metric", (), {"model_name": "ETGNN", "metric_name": "AUPRC", "metric_value": 0.824, "evidence_id": "met_1"})()
        ],
        "provenance_chain": ["pred_etgnn_1", "met_1", "tx_230425980"],
    }

    # 1. Probability within tolerance 0.8523 vs 0.85 (diff 0.0023 <= 0.01) -> PASS
    findings_valid_prob = [
        {"claim_text": "ETGNN model predicted illicit probability of 0.85.", "evidence_citations": ["pred_etgnn_1"]}
    ]
    res1 = validator.validate_findings(findings_valid_prob, bundle, target_tx_id=230425980)
    assert res1.is_valid is True

    # 2. Probability outside tolerance 0.8523 vs 0.90 (diff 0.0477 > 0.01) -> FAIL
    findings_invalid_prob = [
        {"claim_text": "ETGNN model predicted illicit probability of 0.90.", "evidence_citations": ["pred_etgnn_1"]}
    ]
    res2 = validator.validate_findings(findings_invalid_prob, bundle, target_tx_id=230425980)
    assert res2.is_valid is False
    assert any("Numeric discrepancy" in err for err in res2.critical_errors)

    # 3. Benchmark metric within tolerance 0.824 vs 0.824 -> PASS
    findings_valid_met = [
        {"claim_text": "ETGNN achieved test AUPRC of 0.824 on the evaluation split.", "evidence_citations": ["met_1"]}
    ]
    res3 = validator.validate_findings(findings_valid_met, bundle, target_tx_id=230425980)
    assert res3.is_valid is True

    # 4. Benchmark metric outside tolerance 0.824 vs 0.835 (diff 0.011 > 0.001) -> FAIL
    findings_invalid_met = [
        {"claim_text": "ETGNN achieved test AUPRC of 0.835 on the evaluation split.", "evidence_citations": ["met_1"]}
    ]
    res4 = validator.validate_findings(findings_invalid_met, bundle, target_tx_id=230425980)
    assert res4.is_valid is False


def test_citation_target_transaction_mismatch():
    validator = CitationValidator()
    bundle = {
        "provenance_chain": ["tx_111111111", "tx_222222222"],
    }
    # Evaluated target tx is 111111111, but claim cites 222222222
    claim_text = "Transaction 222222222 shows illicit patterns."
    citations = ["tx_222222222"]
    
    result = validator.validate_claim_citations(
        claim_text=claim_text,
        evidence_citations=citations,
        evidence_bundle=bundle,
        target_tx_id=111111111,
    )
    assert result.is_valid is False
    assert len(result.mismatched_target_ids) > 0
    assert 222222222 in result.mismatched_target_ids


def test_citation_claim_type_mismatch():
    validator = CitationValidator()
    bundle = {
        "provenance_chain": ["subgraph_230425980_h2"],
    }
    # Claim is strictly about model prediction probability, but citations only cite graph structure
    claim_text = "The model predicted illicit probability of 0.88 for this transaction."
    citations = ["subgraph_230425980_h2"]

    result = validator.validate_claim_citations(
        claim_text=claim_text,
        evidence_citations=citations,
        evidence_bundle=bundle,
        target_tx_id=230425980,
    )
    assert result.is_valid is False
    assert any("Evidence type mismatch" in err for err in result.validation_errors)


def test_report_validator_topology_distinctions():
    validator = ReportValidator()
    # Test fully isolated node (in_degree=0, out_degree=0)
    bundle_isolated = {
        "transaction": type("Tx", (), {"transaction_id": 99999, "in_degree": 0, "out_degree": 0, "snapshot_id": 10})(),
        "provenance_chain": ["tx_99999"],
    }
    report_missing_qualification = {
        "target_transaction_id": 99999,
        "supported_findings": [],
        "evidence_limitations": [],
    }
    res = validator.validate_report(report_missing_qualification, bundle_isolated, snapshot_id=10, target_tx_id=99999)
    assert res.is_valid is False
    assert any("fully isolated" in r for r in res.reasons)

    # Adding appropriate qualification satisfies the validator
    report_qualified = {
        "target_transaction_id": 99999,
        "supported_findings": [],
        "evidence_limitations": ["Node is fully isolated in snapshot graph (0 in-degree, 0 out-degree); structural message passing is inactive."],
    }
    res_ok = validator.validate_report(report_qualified, bundle_isolated, snapshot_id=10, target_tx_id=99999)
    assert res_ok.is_valid is True


def test_report_validator_snapshot_43_qualification():
    validator = ReportValidator()
    bundle = {
        "transaction": type("Tx", (), {"transaction_id": 12345, "in_degree": 2, "out_degree": 1, "snapshot_id": 43})(),
        "provenance_chain": ["tx_12345"],
    }
    report_no_s43 = {
        "target_transaction_id": 12345,
        "supported_findings": [],
        "evidence_limitations": [],
    }
    res1 = validator.validate_report(report_no_s43, bundle, snapshot_id=43, target_tx_id=12345)
    assert res1.is_valid is False
    assert any("Snapshot 43" in r for r in res1.reasons)

    report_with_s43 = {
        "target_transaction_id": 12345,
        "supported_findings": [],
        "evidence_limitations": ["Snapshot 43 Historical Qualification: Darknet market distribution shift."],
    }
    res2 = validator.validate_report(report_with_s43, bundle, snapshot_id=43, target_tx_id=12345)
    assert res2.is_valid is True
