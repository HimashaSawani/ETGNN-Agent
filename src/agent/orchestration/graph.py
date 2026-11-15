from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

# Backward-compatible type alias pointing directly to official LangGraph CompiledStateGraph
CompiledInvestigationGraph = CompiledStateGraph

from src.agent.orchestration.state import (
    InvestigationState,
    InvestigationStatus,
)
from src.agent.orchestration.nodes import (
    validate_request_node,
    plan_investigation_node,
    execute_tools_node,
    consolidate_evidence_node,
    generate_explanation_node,
    validate_claims_node,
    validate_final_report_node,
    finalize_investigation_node,
)
from src.agent.orchestration.routing import (
    route_after_request_validation,
    route_after_claim_validation,
    route_after_report_validation,
)

logger = logging.getLogger(__name__)


def build_investigation_graph() -> CompiledStateGraph:
    """
    Builds and compiles the production state graph using the official LangGraph library.
    Implements multi-step Bitcoin transaction fraud investigation with bounded retries,
    conditional state transitions, and deterministic evidence collection.
    """
    workflow = StateGraph(InvestigationState)

    # 1. Register the 8 workflow execution nodes
    workflow.add_node("validate_request", validate_request_node)
    workflow.add_node("plan_investigation", plan_investigation_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("consolidate_evidence", consolidate_evidence_node)
    workflow.add_node("generate_explanation", generate_explanation_node)
    workflow.add_node("validate_claims", validate_claims_node)
    workflow.add_node("validate_final_report", validate_final_report_node)
    workflow.add_node("finalize_investigation", finalize_investigation_node)

    # 2. Define Entry Point via official START node
    workflow.add_edge(START, "validate_request")

    # 3. Add Edges & Conditional Routing
    workflow.add_conditional_edges(
        "validate_request",
        route_after_request_validation,
        {
            "plan_investigation": "plan_investigation",
            "finalize_investigation": "finalize_investigation",
        }
    )

    workflow.add_edge("plan_investigation", "execute_tools")
    workflow.add_edge("execute_tools", "consolidate_evidence")
    workflow.add_edge("consolidate_evidence", "generate_explanation")
    workflow.add_edge("generate_explanation", "validate_claims")

    workflow.add_conditional_edges(
        "validate_claims",
        route_after_claim_validation,
        {
            "validate_final_report": "validate_final_report",
            "generate_explanation": "generate_explanation",
        }
    )

    workflow.add_conditional_edges(
        "validate_final_report",
        route_after_report_validation,
        {
            "finalize_investigation": "finalize_investigation",
            "generate_explanation": "generate_explanation",
        }
    )

    workflow.add_edge("finalize_investigation", END)

    # Compile official LangGraph graph
    return workflow.compile()
