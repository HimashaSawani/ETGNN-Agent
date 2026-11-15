import os
import json
import pytest
from unittest.mock import patch, MagicMock
import httpx

from src.agent.llm.provider import (
    get_llm_provider,
    BaseLLMProvider,
    MockLLMProvider,
    EvidenceOnlyProvider,
    GeminiProvider,
    AnthropicProvider,
    OllamaProvider,
    LLMProviderError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMOutputFormatError,
)
from src.agent.services.investigation_runner import InvestigationRunner


def test_mock_provider_metadata_and_simulated_mode():
    """Verify MockLLMProvider is strictly marked as simulated generation."""
    provider = MockLLMProvider()
    assert provider.provider_name == "mock"
    assert provider.model_id == "deterministic_mock_v3"
    
    ctx = {
        "tx_id": 230425980,
        "snapshot_id": 1,
        "investigation_id": "inv_test_mock",
        "evidence_bundle": {
            "transaction": MagicMock(in_degree=1, out_degree=2, snapshot_id=1),
            "predictions": {
                "etgnn": {"illicit_probability": 0.05, "predicted_class": 0},
                "random_forest": {"illicit_probability": 0.04, "predicted_class": 0},
                "xgboost": {"illicit_probability": 0.06, "predicted_class": 0},
            },
            "subgraph": MagicMock(total_subgraph_nodes=3, total_subgraph_edges=2, illicit_neighbors_count=0),
            "provenance_chain": ["tx_230425980"],
        },
    }
    raw = provider.generate("Investigate", "System prompt", context_data=ctx)
    report = json.loads(raw)
    assert report["generation_mode"] == "SIMULATED"
    assert report["live_llm_verified"] is False
    assert report["provider_type"] == "mock"
    assert "epistemic_notice" in report


def test_evidence_only_provider_factual_mode():
    """Verify EvidenceOnlyProvider produces non-generative factual summary."""
    provider = EvidenceOnlyProvider()
    assert provider.provider_name == "evidence_only"
    assert provider.model_id == "deterministic_rules_v1"

    ctx = {
        "tx_id": 230425980,
        "snapshot_id": 1,
        "investigation_id": "inv_test_ev_only",
        "evidence_bundle": {
            "transaction": MagicMock(in_degree=2, out_degree=3, snapshot_id=1),
            "predictions": {
                "etgnn": {"illicit_probability": 0.02, "predicted_class": 0},
                "random_forest": {"illicit_probability": 0.03, "predicted_class": 0},
                "xgboost": {"illicit_probability": 0.02, "predicted_class": 0},
            },
            "subgraph": MagicMock(total_subgraph_nodes=5, total_subgraph_edges=4, illicit_neighbors_count=0),
            "provenance_chain": ["tx_230425980"],
        },
    }
    raw = provider.generate("Investigate", "System prompt", context_data=ctx)
    report = json.loads(raw)
    assert report["generation_mode"] == "DETERMINISTIC_EVIDENCE_ONLY"
    assert report["live_llm_verified"] is False
    assert report["provider_type"] == "evidence_only"
    assert "deterministic non-generative factual summary" in report["narrative_summary"].lower()


def test_live_provider_missing_credentials_raises_typed_error():
    """Verify live providers raise LLMAuthenticationError when credentials are missing, never silent mock fallback."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        with pytest.raises(LLMAuthenticationError) as exc_info:
            GeminiProvider(api_key="")
        assert "API key" in str(exc_info.value)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
        with pytest.raises(LLMAuthenticationError) as exc_info:
            AnthropicProvider(api_key="")
        assert "API key" in str(exc_info.value)


def test_live_provider_timeout_raises_llm_timeout_error():
    """Verify network timeouts map to LLMTimeoutError."""
    prov = GeminiProvider(api_key="test_fake_key_123")
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
        with pytest.raises(LLMTimeoutError):
            prov.generate("prompt", "system")


def test_live_provider_rate_limit_raises_llm_rate_limit_error():
    """Verify HTTP 429 status codes map to LLMRateLimitError."""
    prov = GeminiProvider(api_key="test_fake_key_123")
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=mock_resp)

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(LLMRateLimitError):
            prov.generate("prompt", "system")


def test_runner_fallback_to_evidence_only_when_allowed():
    """Verify runner gracefully falls back to EvidenceOnlyProvider (NOT Mock) when live provider fails and fallback is enabled."""
    runner = InvestigationRunner()
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        result = runner.run_investigation(
            tx_id=230425980,
            llm_provider="gemini",
            allow_evidence_only_fallback=True,
        )
        assert result.get("status") == "COMPLETED"
        report = result.get("final_report", {})
        assert report.get("generation_mode") == "DETERMINISTIC_EVIDENCE_ONLY"
        assert report.get("provider_type") == "evidence_only"
        assert any("fallen back to deterministic evidence-only mode" in str(lim) for lim in report.get("evidence_limitations", []))


def test_runner_raises_when_fallback_disabled():
    """Verify runner fails explicitly when live provider fails and allow_evidence_only_fallback=False."""
    runner = InvestigationRunner()
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        with pytest.raises(LLMAuthenticationError):
            runner.run_investigation(
                tx_id=230425980,
                llm_provider="gemini",
                allow_evidence_only_fallback=False,
            )


def test_get_llm_provider_factory():
    """Verify factory returns appropriate provider types."""
    assert isinstance(get_llm_provider("mock"), MockLLMProvider)
    assert isinstance(get_llm_provider("evidence_only"), EvidenceOnlyProvider)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_key_123"}):
        assert isinstance(get_llm_provider("gemini"), GeminiProvider)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy_key_456"}):
        assert isinstance(get_llm_provider("anthropic"), AnthropicProvider)
    assert isinstance(get_llm_provider("ollama"), OllamaProvider)
