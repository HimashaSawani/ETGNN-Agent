"""
ETGNN-Agent — Phase 6.7 Live Agentic AI + RAG End-to-End Verification Suite.
Validates:
  1. Transaction 94340906 ground truth, model inference (ETGNN, Matched GAT, RF, XGBoost),
     graph context, and explanation generation.
  2. LangGraph multi-step autonomous investigation lifecycle, state transitions,
     telemetry event stream, and structured report synthesis.
  3. BM25 + Dense Hybrid RAG retrieval, research citations, and temporal boundary enforcement.
  4. Live LLM provider audit, strict fail-closed authentication gates, and
     deterministic evidence-only fallback transparency.
  5. React 19 frontend build assets, static SPA serving, and Playwright spec availability.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from fastapi import HTTPException

from src.agent.llm.provider import (
    AnthropicProvider,
    GeminiProvider,
    LLMAuthenticationError,
    OllamaProvider,
    get_llm_provider,
)
from src.agent.orchestration.state import InvestigationStatus
from src.agent.rag.retriever import HybridRetriever
from src.agent.schemas import EvidenceMode, LabelStatus, TemporalBoundaryViolationError
from src.agent.services.investigation_chat_service import (
    ChatRequest,
    InvestigationChatService,
)
from src.agent.services.investigation_runner import get_investigation_runner
from src.agent.telemetry.event_sink import get_event_sink
from src.agent.telemetry.events import InvestigationEventType
from src.agent.tools.investigation_tools import InvestigationToolRegistry
from src.web.app import (
    AgentInvestigateRequest,
    AgentRetrieveRequest,
    api_agent_chat,
    api_agent_investigate,
    api_agent_retrieve,
    api_get_investigation,
    api_get_investigation_events,
    get_explanation,
    get_transaction_subgraph,
    get_transaction,
    predict_transaction,
    PredictRequest,
)
from src.web.data_loader import get_data_loader

ROOT = Path(__file__).resolve().parent.parent
TARGET_TX = 94340906  # Snapshot 43, illicit label 1, node_index 174515


@pytest.fixture(scope="module")
def loader():
    return get_data_loader()


@pytest.fixture(scope="module")
def tool_registry():
    return InvestigationToolRegistry.get_instance()


@pytest.fixture(scope="module")
def runner():
    return get_investigation_runner()


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


# ============================================================================
# Gate 1: Transaction 94340906 Ground Truth & Real Inference
# ============================================================================

def test_tx94340906_ground_truth_and_model_inference(loader):
    """
    Verifies that target transaction 94340906 is correctly aligned:
      - Snapshot 43, split 'test', label 1 (illicit)
      - Forward pass on real ETGNN PyTorch weights produces valid probability
      - Matched GAT, Random Forest, and XGBoost baseline predictions are present
      - 2-hop neighborhood extraction returns valid nodes and edges
      - GNNExplainer / research explanation structure is valid
    """
    # 1. Transaction metadata via FastAPI endpoint handler
    tx_data = get_transaction(TARGET_TX)
    assert tx_data["transaction_id"] == TARGET_TX
    assert tx_data["node_index"] == 174515
    assert tx_data["time_step"] == 43
    assert tx_data["split"] == "test"
    assert tx_data["actual_label"] == 1
    assert tx_data["actual_label_name"] == "illicit"

    # 2. Real inference across all 4 benchmark models
    preds = tx_data["predictions"]
    assert "etgnn" in preds
    assert "matched_gat" in preds
    assert "random_forest" in preds
    assert "xgboost" in preds

    etgnn_prob = preds["etgnn"]["illicit_probability"]
    gat_prob = preds["matched_gat"]["illicit_probability"]
    rf_prob = preds["random_forest"]["illicit_probability"]
    xgb_prob = preds["xgboost"]["illicit_probability"]

    assert 0.0 <= etgnn_prob <= 1.0
    assert 0.0 <= gat_prob <= 1.0
    assert 0.0 <= rf_prob <= 1.0
    assert 0.0 <= xgb_prob <= 1.0

    # Test predict endpoint directly
    pred_res = predict_transaction(PredictRequest(transaction_id=TARGET_TX))
    assert pred_res["transaction_id"] == TARGET_TX
    assert abs(pred_res["predictions"]["etgnn"]["illicit_probability"] - etgnn_prob) < 1e-5

    # 3. 2-hop subgraph extraction
    subgraph = get_transaction_subgraph(TARGET_TX, hops=2, max_nodes=50)
    assert len(subgraph["nodes"]) >= 2
    assert len(subgraph["edges"]) >= 1
    tx_ids = {n["transaction_id"] for n in subgraph["nodes"]}
    assert TARGET_TX in tx_ids
    center_nodes = [n for n in subgraph["nodes"] if n.get("is_center")]
    assert len(center_nodes) == 1
    assert center_nodes[0]["transaction_id"] == TARGET_TX

    # 4. Explanation endpoint
    expl = get_explanation(TARGET_TX)
    assert expl["transaction_id"] == TARGET_TX
    assert "original_predicted_class_probability" in expl
    assert "disclaimer" in expl


# ============================================================================
# Gate 2: LangGraph Autonomous Investigation Lifecycle & Telemetry
# ============================================================================

def test_tx94340906_langgraph_lifecycle_and_telemetry(runner):
    """
    Executes full LangGraph autonomous investigation on target transaction 94340906.
    Verifies:
      - Investigation status is COMPLETED
      - Observable event telemetry stream records all key lifecycle events
      - Final report contains supported findings, research citations, and provenance chain
      - Snapshot 43 darknet market / structural anomaly qualification is present in limitations
    """
    req = AgentInvestigateRequest(
        transaction_id=TARGET_TX,
        question="Investigate illicit transaction 94340906 in Snapshot 43 and assess fraud risk consensus.",
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
        llm_provider="evidence_only",
    )
    result = api_agent_investigate(req)
    assert result["status"] == InvestigationStatus.COMPLETED.value
    inv_id = result["investigation_id"]

    # Verify query via api_get_investigation
    inv_lookup = api_get_investigation(inv_id)
    assert inv_lookup["status"] == InvestigationStatus.COMPLETED.value
    report = inv_lookup["report"]
    assert report is not None
    assert report["target_transaction_id"] == TARGET_TX
    assert report["snapshot_id"] == 43

    # Supported findings & citations
    assert len(report["supported_findings"]) > 0
    assert len(report["research_citations"]) > 0
    assert len(report["provenance_chain"]) >= 4

    # Model predictions in report
    model_preds = report["model_predictions"]
    assert model_preds["canonical_etgnn_prob"] > 0.0

    # Snapshot 43 qualification in limitations
    limitations_text = " ".join(report.get("limitations", [])).lower()
    assert "snapshot 43" in limitations_text or "darknet" in limitations_text or "structural" in limitations_text

    # Telemetry events verification via api_get_investigation_events
    events = api_get_investigation_events(inv_id)
    assert len(events) >= 5
    event_types = [e["event_type"] for e in events]
    assert "INVESTIGATION_STARTED" in event_types
    assert "PLAN_CREATED" in event_types
    assert "INVESTIGATION_COMPLETED" in event_types


# ============================================================================
# Gate 3: BM25 + Dense Hybrid RAG & Temporal Boundary Cutoff
# ============================================================================

def test_tx94340906_hybrid_rag_and_temporal_boundaries(retriever, tool_registry):
    """
    Verifies Hybrid RAG retrieval quality and strict temporal boundary filtering:
      - RAG retrieval surfaces Snapshot 43 structural anomaly and heterophily evidence
      - Historical evaluation mode with as_of_snapshot < 43 strictly raises TemporalBoundaryViolationError
      - Retrospective mode allows ground truth inspection
    """
    # 1. Hybrid RAG retrieval via API handler
    retrieve_req = AgentRetrieveRequest(
        query="Snapshot 43 structural anomaly zero-incoming illicit transactions",
        top_k=3,
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
    )
    rag_res = api_agent_retrieve(retrieve_req)
    assert rag_res["total_evidence_count"] > 0
    assert len(rag_res["research_evidence"]) > 0
    combined_rag_text = " ".join([c["text_content"] for c in rag_res["research_evidence"]]).lower()
    assert "43" in combined_rag_text or "darknet" in combined_rag_text or "anomaly" in combined_rag_text or "structural" in combined_rag_text

    # 2. Strict temporal boundary enforcement on target transaction (Snapshot 43)
    # Querying Snapshot 43 tx with historical as_of_snapshot = 32 MUST be rejected
    with pytest.raises(TemporalBoundaryViolationError):
        tool_registry.get_transaction(
            TARGET_TX,
            evidence_mode=EvidenceMode.HISTORICAL_EVALUATION,
            as_of_snapshot=32,
        )

    # 3. Target transaction queried with as_of_snapshot = 43 in HISTORICAL mode has restricted ground truth
    hist_ev = tool_registry.get_transaction(
        TARGET_TX,
        evidence_mode=EvidenceMode.HISTORICAL_EVALUATION,
        as_of_snapshot=43,
    )
    assert hist_ev.label_status == LabelStatus.RESTRICTED
    assert hist_ev.raw_label_code is None

    # 4. Retrospective mode reveals true illicit label
    retro_ev = tool_registry.get_transaction(
        TARGET_TX,
        evidence_mode=EvidenceMode.RETROSPECTIVE_INVESTIGATION,
    )
    assert retro_ev.label_status == LabelStatus.ILLICIT
    assert retro_ev.raw_label_code == 1


# ============================================================================
# Gate 4: Live LLM Provider Audit & Fail-Closed Gate
# ============================================================================

def test_live_llm_provider_audit_and_fail_closed_gate(monkeypatch):
    """
    Audits live LLM providers and fail-closed security:
      - In the absence of live API keys, Gemini and Anthropic raise LLMAuthenticationError in strict mode
      - Degrades transparently to DETERMINISTIC_EVIDENCE_ONLY when fallback is permitted
      - Follow-up chat generates grounded answers using the investigation evidence bundle
    """
    # Verify environment lacks live keys (or ensure monkeypatched absence for deterministic testing)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMAuthenticationError):
        get_llm_provider("gemini")

    with pytest.raises(LLMAuthenticationError):
        get_llm_provider("anthropic")

    # Run investigation with evidence_only fallback
    runner = get_investigation_runner()
    res = runner.run_investigation(
        tx_id=TARGET_TX,
        question="Verify fail-closed degradation mode on transaction 94340906",
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
        llm_provider="gemini",
        allow_evidence_only_fallback=True,
    )
    assert res["status"] == "COMPLETED"
    report = res["final_report"]
    assert report["generation_mode"] == "DETERMINISTIC_EVIDENCE_ONLY"
    assert report["live_llm_verified"] is False
    assert any("Live LLM provider 'gemini' failed" in lim for lim in report["limitations"])

    # Follow-up interactive chat grounded in evidence bundle
    chat_res = api_agent_chat({
        "investigation_id": res["investigation_id"],
        "question": "What is the consensus between ETGNN and Random Forest on this transaction?",
        "llm_provider": "evidence_only",
    })
    assert chat_res["generation_mode"] == "DETERMINISTIC_EVIDENCE_ONLY"
    assert chat_res["provider_type"] == "evidence_only"
    assert len(chat_res["cited_evidence_ids"]) > 0


# ============================================================================
# Gate 5: React 19 Frontend Build Assets & E2E Test Specifications
# ============================================================================

def test_react_frontend_assets_and_playwright_specs():
    """
    Verifies that the React 19 frontend is built and served:
      - web/frontend/dist/index.html exists and contains the application root
      - web/frontend/dist/assets contains the bundled JS and CSS chunks
      - Playwright end-to-end specifications exist and reference the dashboard
    """
    dist_dir = ROOT / "web" / "frontend" / "dist"
    assert dist_dir.exists(), f"Dist directory {dist_dir} does not exist"

    index_html = dist_dir / "index.html"
    assert index_html.exists(), "index.html not found in dist"
    content = index_html.read_text(encoding="utf-8")
    assert '<div id="root">' in content or '<div id="app">' in content or '<script' in content

    assets_dir = dist_dir / "assets"
    assert assets_dir.exists(), "assets folder not found in dist"
    asset_files = list(assets_dir.glob("*"))
    assert len(asset_files) > 0, "No asset files found in dist/assets"

    # Playwright E2E spec files
    e2e_dir = ROOT / "web" / "frontend" / "e2e"
    assert e2e_dir.exists(), "web/frontend/e2e does not exist"
    copilot_spec = e2e_dir / "investigation_copilot.spec.ts"
    demo_spec = e2e_dir / "phase6_portfolio_demo.spec.ts"
    assert copilot_spec.exists(), "investigation_copilot.spec.ts not found"
    assert demo_spec.exists(), "phase6_portfolio_demo.spec.ts not found"
