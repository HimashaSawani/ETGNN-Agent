import pytest
from src.agent.schemas import (
    EvidenceMode,
    EvidenceRecord,
    GraphEdge,
    GraphEvidence,
    LabelStatus,
    ModelDisagreement,
    ModelMetadataEvidence,
    ModelMetricEvidence,
    PredictionEvidence,
    ResearchDocumentEvidence,
    RetrievalResult,
    SourceAuthorityTier,
    TransactionEvidence,
    VerificationStatus,
)

def test_evidence_record_immutability():
    rec = EvidenceRecord(
        evidence_id="EV-TEST-1",
        source_type="test",
        source_artifact="test.csv",
        citation="Test (2026)",
    )
    assert rec.evidence_id == "EV-TEST-1"
    # Frozen model prevents mutation
    with pytest.raises(Exception):
        rec.evidence_id = "EV-MUTATED"

def test_transaction_evidence_serialization():
    tx_ev = TransactionEvidence(
        evidence_id="EV-TX-100",
        source_type="transaction",
        source_artifact="nodes.csv",
        citation="Weber et al. (2019)",
        node_index=5,
        label_status=LabelStatus.ILLICIT,
        raw_label_code=1,
        in_degree=3,
        out_degree=2,
        is_isolated_incoming=False,
        is_completely_isolated=False,
        safe_feature_summary={"local_mean": 0.5, "anonymized_feature_notice": "No semantic names."},
    )
    d = tx_ev.model_dump()
    assert d["label_status"] == "illicit"
    assert d["in_degree"] == 3
    assert "safe_feature_summary" in d

def test_model_disagreement_schema():
    dis = ModelDisagreement(
        is_divergent=True,
        model_probabilities={"random_forest": 0.85, "etgnn": 0.22},
        probability_spread=0.63,
        divergence_reason="Topological smearing in heterophilic graph.",
    )
    assert dis.is_divergent is True
    assert dis.probability_spread == 0.63

