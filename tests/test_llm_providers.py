import pytest
import json
from src.agent.llm.provider import MockLLMProvider, get_llm_provider
from src.agent.llm.structured_output import extract_json_str, parse_and_validate_json, StructuredOutputError
from src.agent.reports.report_builder import ForensicInvestigationReport


def test_extract_json_str():
    """Verify markdown code fences and whitespace are stripped properly."""
    raw1 = "```json\n{\"key\": \"val\"}\n```"
    assert extract_json_str(raw1) == "{\"key\": \"val\"}"

    raw2 = "Prefix text {\"a\": 1, \"b\": 2,} suffix"
    cleaned = extract_json_str(raw2)
    assert json.loads(cleaned) == {"a": 1, "b": 2}


def test_parse_and_validate_json_with_model():
    """Verify Pydantic model validation on extracted JSON."""
    raw = """
    ```json
    {
        "investigation_id": "inv_123",
        "target_transaction_id": 23042587,
        "snapshot_id": 35,
        "status": "COMPLETED",
        "consensus_verdict": "LOW_RISK_LICIT",
        "supported_findings": [],
        "uncertain_findings": [],
        "limitations": []
    }
    ```
    """
    report = parse_and_validate_json(raw, ForensicInvestigationReport)
    assert isinstance(report, ForensicInvestigationReport)
    assert report.investigation_id == "inv_123"
    assert report.target_transaction_id == 23042587


def test_mock_llm_provider_deterministic_generation():
    """Verify MockLLMProvider produces valid ForensicInvestigationReport JSON using supplied context."""
    provider = MockLLMProvider()
    context = {
        "tx_id": 23042587,
        "snapshot_id": 35,
        "investigation_id": "inv_test_mock",
        "evidence_bundle": {
            "transaction": {"in_degree": 2, "out_degree": 1},
            "predictions": {
                "etgnn": {"illicit_probability": 0.85, "predicted_class": 1, "evidence_id": "EV-PRED-ETGNN-23042587"},
                "random_forest": {"illicit_probability": 0.80, "predicted_class": 1, "evidence_id": "EV-PRED-RF-23042587"},
                "xgboost": {"illicit_probability": 0.78, "predicted_class": 1, "evidence_id": "EV-PRED-XGB-23042587"},
            },
            "subgraph": {"total_subgraph_nodes": 12, "illicit_neighbors_count": 1, "evidence_id": "EV-SUB-23042587-2HOP"},
            "provenance_chain": [
                "EV-TX-23042587",
                "EV-PRED-ETGNN-23042587",
                "EV-PRED-RF-23042587",
                "EV-SUB-23042587-2HOP",
            ]
        }
    }

    raw = provider.generate("Investigate tx", "System prompt", context_data=context)
    report = parse_and_validate_json(raw, ForensicInvestigationReport)
    assert report.investigation_id == "inv_test_mock"
    assert report.consensus_verdict.value == "HIGH_RISK_ILLICIT"
    assert len(report.supported_findings) >= 3


def test_get_llm_provider_default():
    """Verify get_llm_provider defaults to MockLLMProvider when unset."""
    provider = get_llm_provider()
    assert isinstance(provider, MockLLMProvider)

