import pytest
from src.agent.orchestration.graph import build_investigation_graph, StateGraph, START, END
from src.agent.orchestration.state import InvestigationState, InvestigationStatus
from src.agent.orchestration.execution_limits import (
    ExecutionBudgetTracker,
    MAX_TOOL_CALLS,
    ResourceLimitExceededError,
)
from src.agent.telemetry.event_sink import get_event_sink


def test_stategraph_basic_execution():
    """Verify StateGraph engine executes nodes in sequence."""
    graph = StateGraph(dict)
    graph.add_node("step1", lambda s: {"val": s.get("val", 0) + 1})
    graph.add_node("step2", lambda s: {"val": s.get("val", 0) * 2})
    graph.set_entry_point("step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)

    compiled = graph.compile()
    result = compiled.invoke({"val": 5})
    assert result["val"] == 12


def test_stategraph_conditional_routing():
    """Verify conditional branching routes according to condition callable."""
    graph = StateGraph(dict)
    graph.add_node("check", lambda s: s)
    graph.add_node("branch_a", lambda s: {"route": "A"})
    graph.add_node("branch_b", lambda s: {"route": "B"})

    graph.set_entry_point("check")
    graph.add_conditional_edges(
        "check",
        lambda s: "branch_a" if s.get("flag") else "branch_b",
        {"branch_a": "branch_a", "branch_b": "branch_b"}
    )
    graph.add_edge("branch_a", END)
    graph.add_edge("branch_b", END)

    compiled = graph.compile()
    res_a = compiled.invoke({"flag": True})
    assert res_a["route"] == "A"

    res_b = compiled.invoke({"flag": False})
    assert res_b["route"] == "B"


def test_execution_budget_tracker_enforces_limits():
    """Verify ExecutionBudgetTracker halts execution when budget is breached."""
    tracker = ExecutionBudgetTracker(max_tool_calls=3)
    tracker.record_tool_call("tool1")
    tracker.record_tool_call("tool2")
    tracker.record_tool_call("tool3")

    with pytest.raises(ResourceLimitExceededError, match="Tool call limit"):
        tracker.record_tool_call("tool4")


def test_compiled_investigation_graph_structure():
    """Verify production compiled graph contains all 8 required nodes and valid entry point."""
    compiled = build_investigation_graph()
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
    assert expected_nodes.issubset(set(compiled.nodes.keys()))
    assert (
        ("__start__", "validate_request") in compiled.builder.edges
        or (START, "validate_request") in compiled.builder.edges
    )

