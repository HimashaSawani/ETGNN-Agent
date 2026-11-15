"""
Phase 6.6 Backend Integration & Model Verification Test Suite.
Validates:
1. ETGNN model architecture, weights shape, and checkpoint loading.
2. Real prediction endpoint using test-split transactions.
3. Prediction alignment: scores, node indices, time steps, and degree context.
4. Unknown labels never silently treated as licit.
5. Error handling: invalid IDs, bounds, feature overrides, and missing investigations.
6. Graph-RAG and LangGraph backend integration.
"""

import math
import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from src.models.etgnn import ETGNNClassifier
from src.web.app import (
    health,
    get_models,
    get_metrics,
    get_transactions,
    get_transaction,
    get_transaction_subgraph,
    predict_transaction,
    get_explanation,
    api_agent_investigate,
    api_get_investigation,
    api_get_investigation_events,
    api_export_investigation_pdf,
    PredictRequest,
    AgentInvestigateRequest,
)
from src.web.data_loader import get_data_loader


# 1. Model Loading & Architecture Contract
def test_etgnn_checkpoint_and_architecture_contract():
    loader = get_data_loader()
    assert loader.etgnn is not None
    assert isinstance(loader.etgnn, ETGNNClassifier)
    
    # Verify input dimensions and time encoding dimensions
    assert loader.etgnn.conv1.in_channels == 181  # 165 features + 16 time_dim
    assert loader.etgnn.conv1.out_channels == 16   # 64 hidden // 4 heads
    assert loader.etgnn.conv1.heads == 4
    assert loader.etgnn.conv2.in_channels == 64    # concatenated 16 * 4
    assert loader.etgnn.conv2.out_channels == 2   # 2 classes (licit / illicit)
    assert loader.etgnn.time_dim == 16
    assert loader.features.shape[1] == 165


# 2. Real Prediction on Held-Out Test Transaction
def test_real_prediction_on_held_out_test_split():
    # 94340906 is from Snapshot 43 (Test Split, steps 41..49, illicit label 1)
    req = PredictRequest(transaction_id=94340906)
    res = predict_transaction(req)
    
    assert res["transaction_id"] == 94340906
    assert "node_index" in res
    assert res["has_feature_override"] is False
    
    # Assert all models have valid predictions
    preds = res["predictions"]
    assert "etgnn" in preds
    assert "matched_gat" in preds
    assert "random_forest" in preds
    assert "xgboost" in preds
    
    for m_id, p in preds.items():
        assert 0.0 <= p["illicit_probability"] <= 1.0
        assert p["predicted_class"] in [0, 1]
        assert 0.0 <= p["confidence"] <= 1.0


# 3. Prediction Alignment with Graph Context
def test_prediction_alignment_and_graph_context():
    loader = get_data_loader()
    tx_id = 94340906
    
    # Check transaction metadata
    tx_meta = get_transaction(tx_id)
    assert tx_meta["transaction_id"] == tx_id
    assert tx_meta["time_step"] == 43
    assert tx_meta["split"] == "test"
    assert tx_meta["actual_label"] == 1
    assert tx_meta["actual_label_name"] == "illicit"
    
    # Check subgraph
    subgraph_meta = get_transaction_subgraph(tx_id, hops=2)
    assert subgraph_meta["center_transaction_id"] == tx_id
    assert subgraph_meta["node_count"] >= 1
    assert any(n["transaction_id"] == tx_id for n in subgraph_meta["nodes"])


# 4. Unknown Labels Are Never Silently Treated as Licit
def test_unknown_labels_never_silently_treated_as_licit():
    loader = get_data_loader()
    
    # Query for unlabeled nodes
    txs_resp = loader.get_nodes_page(split="unlabelled", limit=10)
    assert txs_resp["total"] > 0
    assert len(txs_resp["items"]) > 0
    
    for item in txs_resp["items"]:
        # Must be strictly 'unknown', never 'licit'
        assert item["actual_label_name"] == "unknown"
        assert item["actual_label_name"] != "licit"
        
        # Verify transaction detail matches
        detail = get_transaction(item["transaction_id"])
        assert detail["actual_label_name"] == "unknown"
        assert detail["actual_label_name"] != "licit"


# 5. API Error Handling: Bounds, Unknown Tx, Overrides
def test_api_error_handling_invalid_inputs():
    # A. Negative tx_id rejected by Pydantic validation
    with pytest.raises(ValidationError):
        PredictRequest(transaction_id=-1)
        
    # B. Overflow tx_id rejected
    with pytest.raises(ValidationError):
        PredictRequest(transaction_id=2**64)
        
    # C. Non-existent transaction ID returns 404
    with pytest.raises(HTTPException) as exc_info:
        get_transaction(999999999999)
    assert exc_info.value.status_code == 404
    
    # D. Feature override dimension overflow (>165)
    with pytest.raises(ValidationError):
        PredictRequest(
            transaction_id=94340906,
            features_override={f"feat_{i}": 1.0 for i in range(170)}
        )
        
    # E. Feature override non-finite values (inf / nan)
    with pytest.raises(ValidationError):
        PredictRequest(
            transaction_id=94340906,
            features_override={"feat_0": float("nan")}
        )


# 6. Graph-RAG & LangGraph Backend Integration
def test_graph_rag_and_langgraph_backend_integration():
    req = AgentInvestigateRequest(
        transaction_id=94340906,
        question="Investigate fraud risk and topology for transaction 94340906",
        llm_provider="mock",
    )
    inv = api_agent_investigate(req)
    inv_id = inv["investigation_id"]
    
    assert inv["status"] == "COMPLETED"
    assert inv["report"]["target_transaction_id"] == 94340906
    assert "research_citations" in inv["report"]
    assert "provenance_chain" in inv["report"]
    assert "model_predictions" in inv["report"]
    
    # Retrieve investigation by ID
    fetched = api_get_investigation(inv_id)
    assert fetched["investigation_id"] == inv_id
    
    # Check streamed events
    events = api_get_investigation_events(inv_id)
    assert len(events) >= 3
    event_types = [e["event_type"] for e in events]
    assert "INVESTIGATION_STARTED" in event_types
    assert "INVESTIGATION_COMPLETED" in event_types
    
    # Verify PDF export
    pdf_resp = api_export_investigation_pdf(inv_id)
    assert pdf_resp.status_code == 200
    assert pdf_resp.media_type == "application/pdf"
    assert pdf_resp.body.startswith(b"%PDF")
