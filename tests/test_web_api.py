"""
Automated backend HTTP/ASGI transport tests for ETGNN Research Demonstration Platform API.
Uses Starlette/FastAPI TestClient (which requires asyncio event loop -> socket.socketpair() -> 127.0.0.1 TCP loopback).
If executed inside a network-isolated Windows sandbox that blocks 127.0.0.1 TCP loopback,
these ASGI transport tests are explicitly marked ENVIRONMENT-BLOCKED (skipped), while direct
handler and data-loader contract verification executes in tests/test_research_integrity.py.
"""

import pytest
from fastapi.testclient import TestClient

from src.web.app import app
from tests.conftest import ENVIRONMENT_BLOCKED_REASON, LOOPBACK_SOCKETPAIR_AVAILABLE

pytestmark = pytest.mark.skipif(
    not LOOPBACK_SOCKETPAIR_AVAILABLE,
    reason=ENVIRONMENT_BLOCKED_REASON,
)

client = TestClient(app) if LOOPBACK_SOCKETPAIR_AVAILABLE else None

# Known exemplar transaction ID from results/explanations/connected_tp.json
EXEMPLAR_TP_TX = 288717023
INVALID_TX = 999999999999


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ETGNN" in data["models_loaded"]
    assert "RandomForest" in data["models_loaded"]
    assert "203,769" in data["dataset"]


def test_models_metadata_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 4
    model_ids = [m["id"] for m in models]
    assert "etgnn" in model_ids
    assert "matched_gat" in model_ids
    assert "random_forest" in model_ids
    assert "xgboost" in model_ids


def test_benchmark_metrics_parity():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["test_nodes_count"] == 9973
    metrics = {m["model"]: m for m in data["metrics"]}

    # Check parity with final_test_results.csv
    assert "Random Forest" in metrics
    assert "ETGNN" in metrics
    rf_auprc = round(metrics["Random Forest"]["auprc"], 4)
    et_auprc = round(metrics["ETGNN"]["auprc"], 4)
    assert rf_auprc == 0.6323
    assert et_auprc == 0.2250
    assert rf_auprc > et_auprc


def test_transactions_pagination_and_filtering():
    response = client.get("/api/transactions?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 203769
    assert len(data["items"]) == 10
    assert data["limit"] == 10
    assert data["offset"] == 0

    res_test = client.get("/api/transactions?limit=5&split=test")
    assert res_test.status_code == 200
    for item in res_test.json()["items"]:
        assert item["split"] == "test"

    res_illicit = client.get("/api/transactions?limit=5&label=1")
    assert res_illicit.status_code == 200
    for item in res_illicit.json()["items"]:
        assert item["actual_label"] == 1
        assert item["actual_label_name"] == "illicit"


def test_get_single_transaction_and_404():
    response = client.get(f"/api/transactions/{EXEMPLAR_TP_TX}")
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == EXEMPLAR_TP_TX
    assert data["actual_label"] == 1
    assert "predictions" in data
    assert "etgnn" in data["predictions"]
    assert "random_forest" in data["predictions"]
    assert "graph_degree" in data

    res_invalid = client.get(f"/api/transactions/{INVALID_TX}")
    assert res_invalid.status_code == 404
    assert "not found" in res_invalid.json()["detail"].lower()


def test_get_subgraph():
    response = client.get(f"/api/transactions/{EXEMPLAR_TP_TX}/subgraph?hops=2")
    assert response.status_code == 200
    data = response.json()
    assert data["center_transaction_id"] == EXEMPLAR_TP_TX
    assert data["node_count"] > 0
    assert data["edge_count"] > 0
    assert len(data["nodes"]) == data["node_count"]
    assert len(data["edges"]) == data["edge_count"]

    center_nodes = [n for n in data["nodes"] if n["is_center"]]
    assert len(center_nodes) == 1
    assert center_nodes[0]["transaction_id"] == EXEMPLAR_TP_TX


def test_predict_endpoint():
    payload = {"transaction_id": EXEMPLAR_TP_TX}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == EXEMPLAR_TP_TX
    preds = data["predictions"]
    for model_key in ["etgnn", "matched_gat", "random_forest", "xgboost"]:
        prob = preds[model_key]["illicit_probability"]
        assert 0.0 <= prob <= 1.0
        assert preds[model_key]["predicted_class"] in (0, 1)


def test_predict_with_feature_override():
    payload = {
        "transaction_id": EXEMPLAR_TP_TX,
        "features_override": {"feat_0": 5.0, "feat_1": -2.0},
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["has_feature_override"] is True
    assert "predictions" in data


def test_get_explanation_frozen_exemplar():
    response = client.get(f"/api/explanations/{EXEMPLAR_TP_TX}")
    assert response.status_code == 200
    data = response.json()
    assert data["is_frozen_research_artifact"] is True
    assert data["case"] == "TP"
    assert "top_edges" in data
    assert len(data["top_edges"]) > 0
    assert "scientific_safety_notice" in data


def test_research_c1_endpoint():
    response = client.get("/api/research/c1")
    assert response.status_code == 200
    data = response.json()
    assert "C1" in data["experiment"]
    assert "INCONCLUSIVE" in data["findings_summary"]
    assert len(data["data"]) >= 5


def test_research_c2_endpoint():
    response = client.get("/api/research/c2")
    assert response.status_code == 200
    data = response.json()
    assert data["registers_count"] == 8
    assert len(data["registers"]) == 8


def test_research_ablations_endpoint():
    response = client.get("/api/research/ablations")
    assert response.status_code == 200
    data = response.json()
    assert "multi_seed_comparison" in data
    assert "edge_ablation_anomaly" in data
    assert "+0.2468" in data["edge_ablation_anomaly"]["phenomenon"]


def test_spa_index_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "ETGNN" in response.text
    assert "<div id=\"root\">" in response.text


def test_subgraph_truncation_disclosure():
    response = client.get(f"/api/transactions/{EXEMPLAR_TP_TX}/subgraph?hops=2&max_nodes=10")
    assert response.status_code == 200
    data = response.json()
    assert data["is_truncated"] is True
    assert data["total_untruncated_nodes"] > 10
    assert data["node_count"] <= 10
    assert "truncated" in data["truncation_notice"].lower()


def test_security_input_validation_and_path_traversal():
    res_neg = client.post("/api/predict", json={"transaction_id": -5})
    assert res_neg.status_code == 422

    res_bad = client.post(
        "/api/predict",
        content=b'{"transaction_id": 288717023, "features_override": {"feat_0": "not_a_number"}}',
        headers={"Content-Type": "application/json"},
    )
    assert res_bad.status_code == 422

    huge_overrides = {f"feat_{i}": 1.0 for i in range(200)}
    res_huge = client.post(
        "/api/predict",
        json={"transaction_id": EXEMPLAR_TP_TX, "features_override": huge_overrides},
    )
    assert res_huge.status_code == 422

    res_traversal = client.get("/..%2f..%2fsecret.txt")
    assert res_traversal.status_code == 403


def test_research_external_validation_endpoint():
    response = client.get("/api/research/external_validation")
    assert response.status_code == 200
    data = response.json()
    assert "Elliptic++ Actors" in data["experiment"]
    assert "validation_summary" in data
    assert "test_summary" in data
    assert "external_xai_summary" in data
    assert "conditions" in data["validation_summary"]
    assert "B" in data["validation_summary"]["conditions"]
    assert data["validation_summary"]["conditions"]["B"]["param_count"] == 9736
    assert data["validation_summary"]["contrasts_auprc"][0]["contrast_id"] == "B_vs_D_rewired"
    assert data["validation_summary"]["contrasts_auprc"][0]["statistically_significant_holm_005"] is True


def test_research_temporal_stability_endpoint():
    response = client.get("/api/research/temporal_stability")
    assert response.status_code == 200
    data = response.json()
    assert "Phase 12 Experiment A" in data["experiment"]
    assert "census_summary" in data
    assert data["census_summary"]["total_nodes"] == 9973
    assert data["census_summary"]["preshift_illicit_zero_edge_pct"] == 13.52
    assert data["census_summary"]["postshift_illicit_zero_edge_pct"] == 76.92
    assert "stability_summary" in data
    assert data["stability_summary"]["stratum_2_matched_pairs"] == 59


def test_research_error_mitigation_endpoint():
    response = client.get("/api/research/error_mitigation")
    assert response.status_code == 200
    data = response.json()
    assert "Phase 12 Experiment B" in data["experiment"]
    assert data["strategies_count"] == 21
    assert len(data["mitigation_results"]) == 63
    full_strats = [s for s in data["mitigation_results"] if "Full_Test" in s["Period"]]
    assert len(full_strats) == 21
    strat_ids = [s["Strategy_ID"] for s in full_strats]
    assert any("13_C2b" in s for s in strat_ids)
    assert any("19_C3c" in s for s in strat_ids)
    assert "paired_bootstrap_contrasts" in data
    assert len(data["paired_bootstrap_contrasts"]) >= 10


def test_residual_etgnn_prediction_in_transaction_and_predict():
    response = client.get(f"/api/transactions/{EXEMPLAR_TP_TX}")
    assert response.status_code == 200
    data = response.json()
    assert "residual_etgnn" in data["predictions"]
    res_pred = data["predictions"]["residual_etgnn"]
    assert res_pred["model_id"] == "Residual_ETGNN_93f"
    assert 0.0 <= res_pred["illicit_probability"] <= 1.0

    res_post = client.post("/api/predict", json={"transaction_id": EXEMPLAR_TP_TX})
    assert res_post.status_code == 200
    post_data = res_post.json()
    assert "residual_etgnn" in post_data["predictions"]
    assert 0.0 <= post_data["predictions"]["residual_etgnn"]["illicit_probability"] <= 1.0


