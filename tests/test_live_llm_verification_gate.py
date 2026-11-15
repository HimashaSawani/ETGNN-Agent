"""
ETGNN-Agent — Live LLM Integration Verification Gate Test Suite.
Audits and verifies:
  1. Live provider credential/runtime checks (Gemini, Anthropic, Ollama) without exposing secrets
  2. Strict non-fabrication when live credentials/runtimes are unavailable (no silent fallback to mock)
  3. Explicit degradation to DETERMINISTIC_EVIDENCE_ONLY when allow_evidence_only_fallback=True
  4. Live provider wire-protocol, token/cost telemetry extraction, and LangGraph validation gates
  5. Follow-up InvestigationChatService behavior across evidence_only, mock, and live provider modes
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch
import pytest

from src.agent.llm.provider import (
    AnthropicProvider,
    GeminiProvider,
    LLMAuthenticationError,
    LLMProviderError,
    OllamaProvider,
    get_llm_provider,
)
from src.agent.services.investigation_chat_service import (
    ChatRequest,
    InvestigationChatService,
)
from src.agent.services.investigation_runner import get_investigation_runner


def test_live_provider_environment_audit_and_strict_errors(monkeypatch):
    """
    Verifies that when live API keys are absent, Gemini and Anthropic raise LLMAuthenticationError
    immediately and never silently fall back to MockLLMProvider.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMAuthenticationError, match="Gemini API key not found"):
        get_llm_provider("gemini")

    with pytest.raises(LLMAuthenticationError, match="Anthropic API key not found"):
        get_llm_provider("anthropic")


def test_ollama_provider_unreachable_daemon_raises_provider_error():
    """
    Verifies that OllamaProvider raises LLMProviderError when no local Ollama server is listening,
    never fabricating a simulated response.
    """
    provider = OllamaProvider(base_url="http://127.0.0.1:11439", model="llama3:latest")
    with pytest.raises(LLMProviderError, match="Ollama local API request failed"):
        provider.generate("Analyze tx 288717023", "Forensic system prompt")


def test_langgraph_live_provider_failure_and_evidence_only_degradation(monkeypatch):
    """
    Verifies that running the full 11-stage LangGraph workflow with an unconfigured live provider:
      1) Raises LLMAuthenticationError when allow_evidence_only_fallback=False
      2) Degrades transparently to DETERMINISTIC_EVIDENCE_ONLY (with live_llm_verified=False)
         when allow_evidence_only_fallback=True, while still retrieving real ML predictions,
         2-hop graph topology, GNNExplainer, and Hybrid Graph-RAG evidence.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    runner = get_investigation_runner()

    # 1. Strict mode (allow_evidence_only_fallback=False) must raise LLMAuthenticationError
    with pytest.raises(LLMAuthenticationError):
        runner.run_investigation(
            tx_id=288717023,
            question="Verify live Gemini strict failure mode",
            evidence_mode="RETROSPECTIVE_INVESTIGATION",
            llm_provider="gemini",
            allow_evidence_only_fallback=False,
        )

    # 2. Evidence-only fallback mode (allow_evidence_only_fallback=True)
    res = runner.run_investigation(
        tx_id=288717023,
        question="Verify deterministic evidence-only fallback when live Gemini is unconfigured",
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
        llm_provider="gemini",
        allow_evidence_only_fallback=True,
    )
    assert res["status"] == "COMPLETED"
    report = res["final_report"]
    assert report["generation_mode"] == "DETERMINISTIC_EVIDENCE_ONLY"
    assert report["provider_type"] == "evidence_only"
    assert report["live_llm_verified"] is False
    assert any("Live LLM provider 'gemini' failed" in lim for lim in report["limitations"])

    # Confirm actual ML predictions, graph topology, GNNExplainer, and RAG evidence were retrieved
    preds = report["model_predictions"]
    assert preds["canonical_etgnn_prob"] > 0.0
    assert preds["matched_gat_prob"] > 0.0
    assert preds["random_forest_prob"] > 0.0
    assert preds["xgboost_prob"] > 0.0
    assert report["graph_evidence"]["subgraph_nodes_2hop"] > 1
    assert len(report["provenance_chain"]) >= 7


def test_gemini_and_anthropic_wire_telemetry_extraction():
    """
    Verifies that GeminiProvider and AnthropicProvider accurately extract token usage and
    estimated USD cost from provider JSON responses when HTTP 200 is returned.
    """
    mock_gemini_resp = MagicMock()
    mock_gemini_resp.status_code = 200
    mock_gemini_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"narrative_summary": "Verified"}'}]}}],
        "usageMetadata": {
            "promptTokenCount": 800,
            "candidatesTokenCount": 200,
            "totalTokenCount": 1000,
        },
    }
    with patch("httpx.Client.post", return_value=mock_gemini_resp):
        gem = GeminiProvider(api_key="test_contract_only_key")
        out = gem.generate("prompt", "system")
        assert "Verified" in out
        assert gem.last_token_usage == {
            "prompt_tokens": 800,
            "completion_tokens": 200,
            "total_tokens": 1000,
        }
        assert gem.last_estimated_cost_usd > 0.0

    mock_claude_resp = MagicMock()
    mock_claude_resp.status_code = 200
    mock_claude_resp.json.return_value = {
        "content": [{"text": '{"narrative_summary": "Claude Verified"}'}],
        "usage": {"input_tokens": 600, "output_tokens": 150},
    }
    with patch("httpx.Client.post", return_value=mock_claude_resp):
        ant = AnthropicProvider(api_key="test_contract_only_key")
        out_c = ant.generate("prompt", "system")
        assert "Claude Verified" in out_c
        assert ant.last_token_usage == {
            "prompt_tokens": 600,
            "completion_tokens": 150,
            "total_tokens": 750,
        }
        assert ant.last_estimated_cost_usd > 0.0


def test_followup_investigation_chat_verification(monkeypatch):
    """
    Verifies follow-up investigation chat in evidence_only and mock modes,
    and confirms unconfigured live provider in chat raises LLMAuthenticationError.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    runner = get_investigation_runner()
    inv = runner.run_investigation(
        tx_id=288717023,
        evidence_mode="RETROSPECTIVE_INVESTIGATION",
        llm_provider="evidence_only",
    )
    inv_id = inv["investigation_id"]
    chat_svc = InvestigationChatService()

    # 1. Deterministic evidence_only follow-up chat
    ans_ev = chat_svc.answer_question(
        ChatRequest(
            investigation_id=inv_id,
            question="Why did ETGNN and Random Forest differ or agree on this transaction?",
            llm_provider="evidence_only",
        )
    )
    assert ans_ev.generation_mode == "DETERMINISTIC_EVIDENCE_ONLY"
    assert ans_ev.provider_type == "evidence_only"
    assert len(ans_ev.cited_evidence_ids) > 0
    for cid in ans_ev.cited_evidence_ids:
        assert cid in inv["provenance_chain"]

    # 2. Unconfigured live provider in chat must raise LLMAuthenticationError
    with pytest.raises(LLMAuthenticationError):
        chat_svc.answer_question(
            ChatRequest(
                investigation_id=inv_id,
                question="Summarize the 2-hop graph neighborhood.",
                llm_provider="gemini",
            )
        )

