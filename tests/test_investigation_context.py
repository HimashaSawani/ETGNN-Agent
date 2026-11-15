import pytest
from src.agent.services.investigation_context_service import InvestigationContextService
from src.agent.schemas import EvidenceMode

@pytest.fixture(scope="module")
def context_service():
    return InvestigationContextService()

def test_build_investigation_context_e2e(context_service):
    # Use first transaction in dataset
    sample_tx = int(context_service.tools.loader.nodes_df["txId"].iloc[0])

    ctx = context_service.build_investigation_context(
        tx_id=sample_tx,
        question="Why might models diverge on this transaction?",
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
    )

    assert ctx.tx_id == sample_tx
    assert ctx.transaction_evidence is not None
    assert ctx.graph_evidence is not None
    assert "random_forest" in ctx.predictions
    assert "etgnn" in ctx.predictions
    assert "xgboost" in ctx.predictions
    assert ctx.disagreement is not None
    assert len(ctx.research_context) > 0
    assert len(ctx.reproduced_metrics) >= 4
    assert len(ctx.provenance_chain) >= 5

def test_investigation_context_provenance_chain_integrity(context_service):
    sample_tx = int(context_service.tools.loader.nodes_df["txId"].iloc[0])

    ctx = context_service.build_investigation_context(
        tx_id=sample_tx,
        question="Ablation evidence and edge impact",
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
    )

    # Assert provenance chain contains deterministic IDs
    assert any(p.startswith("EV-TX-") for p in ctx.provenance_chain)
    assert any(p.startswith("EV-GRAPH-") for p in ctx.provenance_chain)
    assert any(p.startswith("EV-PRED-") for p in ctx.provenance_chain)
    assert any(p.startswith("EV-DOC-") for p in ctx.provenance_chain)
    assert any(p.startswith("EV-METRIC-") for p in ctx.provenance_chain)

