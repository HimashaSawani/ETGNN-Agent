import pytest
import importlib.metadata
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from src.agent.orchestration.graph import (
    build_investigation_graph,
    CompiledInvestigationGraph,
)
from src.agent.orchestration.state import InvestigationState, InvestigationStatus
from src.agent.orchestration.routing import (
    route_after_request_validation,
    route_after_claim_validation,
    route_after_report_validation,
)
from src.agent.services.investigation_runner import InvestigationRunner


def test_official_langgraph_package_installed_and_version():
    """Verify official langgraph package is installed and accessible via importlib."""
    version = importlib.metadata.version("langgraph")
    assert version is not None
    assert version.startswith("1.2") or version.startswith("1.")


def test_official_langgraph_compiled_type():
    """Verify build_investigation_graph returns official CompiledStateGraph."""
    graph = build_investigation_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert issubclass(CompiledInvestigationGraph, CompiledStateGraph) or CompiledInvestigationGraph is CompiledStateGraph


def test_official_langgraph_nodes_registered():
    """Verify all 8 lifecycle nodes are registered in the CompiledStateGraph."""
    graph = build_investigation_graph()
    nodes = set(graph.nodes.keys())
    expected_nodes = {
        "validate_request",
        "plan_investigation",
        "execute_tools",
        "consolidate_evidence",
        "generate_explanation",
        "validate_claims",
        "validate_final_report",
        "finalize_investigation",
    }
    assert expected_nodes.issubset(nodes)


def test_routing_after_request_validation():
    """Test conditional routing logic after request validation."""
    valid_state = {"status": InvestigationStatus.PLANNING.value}
    abstained_state = {"status": InvestigationStatus.ABSTAINED.value}
    
    assert route_after_request_validation(valid_state) == "plan_investigation"
    assert route_after_request_validation(abstained_state) == "finalize_investigation"


def test_routing_after_claim_validation():
    """Test conditional routing logic after claim validation with retry bounds."""
    # Under retry limit and valid -> proceed
    valid_state = {
        "claim_validation_result": {"is_valid": True},
        "retries_count": 0,
    }
    assert route_after_claim_validation(valid_state) == "validate_final_report"

    # Under retry limit and invalid -> regenerate
    invalid_state = {
        "claim_validation_result": {"is_valid": False},
        "retries_count": 0,
    }
    assert route_after_claim_validation(invalid_state) == "generate_explanation"

    # Over retry limit and invalid -> proceed anyway (bounded retry)
    exhausted_state = {
        "claim_validation_result": {"is_valid": False},
        "retries_count": 2,
    }
    assert route_after_claim_validation(exhausted_state) == "validate_final_report"


def test_routing_after_report_validation():
    """Test conditional routing logic after report validation with retry bounds."""
    valid_state = {
        "report_validation_result": {"is_valid": True},
        "retries_count": 0,
    }
    assert route_after_report_validation(valid_state) == "finalize_investigation"

    invalid_state = {
        "report_validation_result": {"is_valid": False, "should_regenerate": True},
        "retries_count": 0,
    }
    assert route_after_report_validation(invalid_state) == "generate_explanation"

    exhausted_state = {
        "report_validation_result": {"is_valid": False, "should_regenerate": True},
        "retries_count": 2,
    }
    assert route_after_report_validation(exhausted_state) == "finalize_investigation"


def test_official_langgraph_end_to_end_execution():
    """Verify end-to-end execution of the official LangGraph state machine."""
    runner = InvestigationRunner()
    result = runner.run_investigation(
        tx_id=230425980,
        question="Verify transaction classification and risk status.",
        llm_provider="mock",
    )
    assert result is not None
    assert result.get("status") == "COMPLETED"
    report = result.get("final_report")
    assert report is not None
    assert report.get("target_transaction_id") == 230425980
    assert report.get("status") == "COMPLETED"
    assert report.get("generation_mode") == "SIMULATED"
    assert report.get("live_llm_verified") is False
    assert len(report.get("supported_findings", [])) >= 1
