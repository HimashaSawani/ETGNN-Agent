import pytest
from src.agent.tools.investigation_tools import InvestigationToolRegistry
from src.agent.schemas import EvidenceMode, LabelStatus, TemporalBoundaryViolationError

@pytest.fixture(scope="module")
def tool_registry():
    return InvestigationToolRegistry.get_instance()

def test_future_transaction_rejection_in_historical_mode(tool_registry):
    # Find a transaction at step 45
    t45_nodes = tool_registry.loader.nodes_df[tool_registry.loader.nodes_df["time_step"] == 45]
    assert len(t45_nodes) > 0
    t45_tx = int(t45_nodes["txId"].iloc[0])

    # Query with as_of_snapshot = 32 (should be rejected)
    with pytest.raises(TemporalBoundaryViolationError):
        tool_registry.get_transaction(
            t45_tx,
            evidence_mode=EvidenceMode.HISTORICAL_EVALUATION,
            as_of_snapshot=32,
        )

def test_test_label_masking_in_historical_mode(tool_registry):
    # Find a test transaction (time_step >= 41)
    test_nodes = tool_registry.loader.nodes_df[tool_registry.loader.nodes_df["time_step"] >= 41]
    test_tx = int(test_nodes["txId"].iloc[0])

    ev = tool_registry.get_transaction(
        test_tx,
        evidence_mode=EvidenceMode.HISTORICAL_EVALUATION,
        as_of_snapshot=45,
    )
    # Ground truth is restricted in historical evaluation mode
    assert ev.label_status == LabelStatus.RESTRICTED
    assert ev.raw_label_code is None

def test_retrospective_mode_reveals_ground_truth(tool_registry):
    test_nodes = tool_registry.loader.nodes_df[
        (tool_registry.loader.nodes_df["time_step"] >= 41) & (tool_registry.loader.nodes_df["label"] == 1)
    ]
    test_tx = int(test_nodes["txId"].iloc[0])

    ev = tool_registry.get_transaction(
        test_tx,
        evidence_mode=EvidenceMode.RETROSPECTIVE_INVESTIGATION,
    )
    assert ev.label_status == LabelStatus.ILLICIT
    assert ev.raw_label_code == 1

