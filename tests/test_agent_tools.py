import pytest
import numpy as np
import pandas as pd
from src.agent.tools.investigation_tools import (
    InvestigationToolRegistry,
    get_transaction,
    get_transaction_neighbors,
    get_prediction,
    get_explanation,
    get_model_metadata,
    get_evaluation_metrics,
)
from src.agent.schemas import (
    EvidenceMode,
    LabelStatus,
    TransactionNotFoundError,
    ModelNotSupportedError,
    ResourceLimitExceededError,
    TemporalBoundaryViolationError,
)

@pytest.fixture(scope="module")
def tool_registry():
    return InvestigationToolRegistry.get_instance()

def test_valid_transaction_lookup(tool_registry):
    # Retrieve first node
    sample_tx = int(tool_registry.loader.nodes_df["txId"].iloc[0])
    ev = tool_registry.get_transaction(sample_tx, evidence_mode=EvidenceMode.RETROSPECTIVE_INVESTIGATION)

    assert ev.tx_id == sample_tx
    assert ev.evidence_id == f"EV-TX-{sample_tx}"
    assert ev.feature_dimension == 165
    assert ev.local_features_count == 93
    assert ev.aggregated_features_count == 72
    assert "local_mean" in ev.safe_feature_summary
    assert "anonymized_feature_notice" in ev.safe_feature_summary

def test_invalid_transaction_id(tool_registry):
    with pytest.raises(TransactionNotFoundError):
        tool_registry.get_transaction(9999999999999)

def test_unknown_label_preservation(tool_registry):
    # Find transaction with label == -1
    unknown_rows = tool_registry.loader.nodes_df[tool_registry.loader.nodes_df["label"] == -1]
    assert len(unknown_rows) > 0
    sample_unknown_tx = int(unknown_rows["txId"].iloc[0])

    ev = tool_registry.get_transaction(sample_unknown_tx, evidence_mode=EvidenceMode.RETROSPECTIVE_INVESTIGATION)
    assert ev.label_status == LabelStatus.UNKNOWN
    assert ev.raw_label_code == -1

def test_zero_in_degree_transaction(tool_registry):
    # Find transaction with zero in-degree
    zero_in_indices = np.where(tool_registry.in_degrees == 0)[0]
    assert len(zero_in_indices) > 0
    sample_tx = int(tool_registry.loader.nodes_df["txId"].iloc[zero_in_indices[0]])

    ev = tool_registry.get_transaction(sample_tx, evidence_mode=EvidenceMode.RETROSPECTIVE_INVESTIGATION)
    assert ev.in_degree == 0
    assert ev.is_isolated_incoming is True

def test_directed_one_hop_graph_traversal(tool_registry):
    sample_tx = int(tool_registry.loader.nodes_df["txId"].iloc[0])
    graph_ev = tool_registry.get_transaction_neighbors(
        sample_tx, hops=1, max_nodes=50, evidence_mode=EvidenceMode.RETROSPECTIVE_INVESTIGATION
    )

    assert graph_ev.tx_id == sample_tx
    assert graph_ev.hops == 1
    assert graph_ev.total_subgraph_nodes >= 1
    assert sample_tx in graph_ev.subgraph_nodes
    # Verify all edges are intra-snapshot
    for edge in graph_ev.directed_edges:
        assert edge.is_intra_snapshot is True

def test_bounded_two_hop_resource_limits(tool_registry):
    sample_tx = int(tool_registry.loader.nodes_df["txId"].iloc[0])
    
    # 2 hops valid
    graph_ev = tool_registry.get_transaction_neighbors(
        sample_tx, hops=2, max_nodes=20, evidence_mode=EvidenceMode.RETROSPECTIVE_INVESTIGATION
    )
    assert graph_ev.hops == 2
    assert len(graph_ev.subgraph_nodes) <= 20

    # 3 hops rejected
    with pytest.raises(ResourceLimitExceededError):
        tool_registry.get_transaction_neighbors(sample_tx, hops=3)

    # max_nodes > 100 rejected
    with pytest.raises(ResourceLimitExceededError):
        tool_registry.get_transaction_neighbors(sample_tx, hops=1, max_nodes=150)

def test_model_prediction_mapping(tool_registry):
    sample_tx = int(tool_registry.loader.nodes_df["txId"].iloc[0])

    for m in ("etgnn", "random_forest", "xgboost"):
        pred = tool_registry.get_prediction(sample_tx, model_name=m)
        assert pred.tx_id == sample_tx
        assert 0.0 <= pred.illicit_probability <= 1.0
        assert pred.predicted_class in (0, 1)
        assert pred.decision_threshold == 0.50
        assert pred.is_calibrated_confidence is False
        assert pred.training_cutoff_step == 32

def test_model_not_supported(tool_registry):
    sample_tx = int(tool_registry.loader.nodes_df["txId"].iloc[0])
    with pytest.raises(ModelNotSupportedError):
        tool_registry.get_prediction(sample_tx, model_name="unsupported_deep_network")

def test_model_metadata_and_checksums(tool_registry):
    meta_list = tool_registry.get_model_metadata()
    assert len(meta_list) >= 4
    
    rf_meta = [m for m in meta_list if m.model_name == "random_forest"][0]
    assert rf_meta.input_dimension == 165
    assert len(rf_meta.source_checksum) == 64  # SHA-256 length

def test_reproduced_metrics_retrieval(tool_registry):
    metrics = tool_registry.get_evaluation_metrics()
    assert len(metrics) >= 4
    
    rf_auprc = [m for m in metrics if m.model_name == "random_forest" and m.metric_name == "AUPRC"][0]
    assert rf_auprc.metric_value == 0.632263
    assert rf_auprc.reproduction_status == "INDEPENDENTLY_REPRODUCED"

def test_explanation_with_zero_degree_fallback(tool_registry):
    # Find node with 0 incoming candidate edges
    zero_in_indices = np.where(tool_registry.in_degrees == 0)[0]
    sample_tx = int(tool_registry.loader.nodes_df["txId"].iloc[zero_in_indices[0]])

    exp = tool_registry.get_explanation(sample_tx)
    assert exp.tx_id == sample_tx
    assert "fidelity_disclaimer" in exp.model_dump()
    assert exp.explanation_fidelity_drop == 0.0

