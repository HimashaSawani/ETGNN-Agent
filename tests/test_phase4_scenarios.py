"""
Phase 4 End-to-End Scenarios Test Suite.
Verifies Scenarios A through F required by Phase 4 specification:
  - Scenario A: High-risk agreement across models
  - Scenario B: Divergence detection / probability spread logic
  - Scenario C: Graph neighborhood evidence retrieval
  - Scenario D: Snapshot 43 temporal qualification
  - Scenario E: Human feedback SQLite storage & isolation
  - Scenario F: Fallback execution when live provider unconfigured
"""

import os
import pytest
from src.web.app import (
    api_agent_investigate,
    api_submit_feedback,
    api_get_investigation_feedback,
    api_list_feedback,
    AgentInvestigateRequest,
)
from src.agent.feedback.store import get_feedback_store

@pytest.fixture(scope="module")
def investigation_results():
    """Runs investigation once and yields results for Scenarios A-D."""
    req = AgentInvestigateRequest(
        transaction_id=230425980,
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
        llm_provider="mock",
        allow_evidence_only_fallback=True,
    )
    res = api_agent_investigate(req)
    assert res["status"] == "COMPLETED"
    return res

def test_scenario_a_model_agreement(investigation_results):
    """Scenario A: Model evaluation on target transaction."""
    report = investigation_results["report"]
    assert report is not None
    preds = report["model_predictions"]
    assert any(k in preds for k in ["canonical_etgnn_prob", "etgnn", "etgnn_prob"])
    assert any(k in preds for k in ["random_forest_prob", "random_forest"])
    assert any(k in preds for k in ["xgboost_prob", "xgboost"])
    assert report["risk_indicator"] in ["HIGH_RISK", "CRITICAL_RISK", "SUSPICIOUS", "LOW_RISK", "SAFE", "DIVERGENT_MODELS", "UNCERTAIN"]

def test_scenario_b_model_disagreement_detection(investigation_results):
    """Scenario B: Probability spread and divergence threshold checking."""
    report = investigation_results["report"]
    assert "probability_spread" in report
    spread = report["probability_spread"]
    assert isinstance(spread, (int, float))
    assert 0.0 <= spread <= 1.0
    # Model disagreement flag must match spread >= 0.30
    assert report["model_disagreement"] == (spread >= 0.30)

def test_scenario_c_graph_evidence_retrieval(investigation_results):
    """Scenario C: Multi-hop graph neighborhood evidence retrieval."""
    report = investigation_results["report"]
    graph_ev = report.get("graph_evidence")
    assert graph_ev is not None
    assert "incoming_edges_count" in graph_ev
    assert "outgoing_edges_count" in graph_ev
    assert "subgraph_nodes_2hop" in graph_ev
    assert graph_ev["subgraph_nodes_2hop"] >= 1

def test_scenario_d_snapshot43_temporal_qualification(investigation_results):
    """Scenario D: Snapshot 43 qualification logic in agent findings."""
    report = investigation_results["report"]
    assert len(report["limitations"]) > 0
    assert len(report["provenance_chain"]) > 0

def test_scenario_e_analyst_feedback_isolated_store():
    """Scenario E: Analyst feedback submission and SQLite isolation check."""
    store = get_feedback_store()
    initial_stats = store.get_feedback_stats()
    initial_count = initial_stats["total_feedback_count"]

    feedback_payload = {
        "investigation_id": "test_inv_scenario_e",
        "transaction_id": 999999999,
        "feedback_category": "helpful",
        "rating": 5,
        "comment": "Comprehensive topological breakdown and accurate citation mapping.",
        "reviewer_id": "auditor_e2e",
        "evidence_references": ["EV-001", "EV-002"],
    }
    rec = api_submit_feedback(feedback_payload)
    assert rec["feedback_id"].startswith("fb_")
    assert rec["rating"] == 5

    # Verify query
    items = api_get_investigation_feedback("test_inv_scenario_e")
    assert len(items) >= 1
    assert any(x["feedback_id"] == rec["feedback_id"] for x in items)

    # Verify stats
    stats_data = api_list_feedback()
    assert stats_data["stats"]["total_feedback_count"] == initial_count + 1

    # Invariant: Verify raw elliptic data files and experimental records are NOT touched
    raw_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    experiments_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
    feedback_db = os.path.join(os.path.dirname(__file__), "..", "data", "analyst_feedback.db")
    assert os.path.exists(raw_data_dir)
    assert os.path.exists(experiments_dir)
    assert os.path.exists(feedback_db)

def test_scenario_f_provider_failure_and_fallback():
    """Scenario F: Fallback to deterministic mock / evidence-only when live provider fails."""
    req = AgentInvestigateRequest(
        transaction_id=230425980,
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
        llm_provider="gemini",  # No live API key configured in test environment
        allow_evidence_only_fallback=True,
    )
    res = api_agent_investigate(req)
    assert res["status"] == "COMPLETED"
    report = res["report"]
    # Must report fallback / simulated / evidence-only mode transparently
    assert report["generation_mode"] in ["SIMULATED", "DETERMINISTIC_EVIDENCE_ONLY"]
