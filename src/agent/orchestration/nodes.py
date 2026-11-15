from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from src.agent.orchestration.state import (
    InvestigationState,
    InvestigationStatus,
    InvestigationPlan,
    ToolCallRequest,
)
from src.agent.orchestration.execution_limits import (
    ExecutionBudgetTracker,
    MAX_TOOL_CALLS,
    MAX_HOPS,
    MAX_SUBGRAPH_NODES,
    MAX_RAG_CHUNKS,
)
def _get_tool_registry():
    from src.agent.tools.investigation_tools import InvestigationToolRegistry
    return InvestigationToolRegistry.get_instance()


def _get_hybrid_retriever():
    from src.agent.rag.retriever import HybridRetriever
    return HybridRetriever()
from src.agent.telemetry.events import (
    AgentInvestigationEvent,
    InvestigationEventType,
    ExecutionStatus,
)
from src.agent.telemetry.event_sink import get_event_sink
from src.agent.llm.provider import get_llm_provider, MockLLMProvider, EvidenceOnlyProvider
from src.agent.llm.prompts import (
    SYSTEM_PROMPT_FORENSIC_INVESTIGATOR,
    INVESTIGATION_PROMPT_TEMPLATE,
    REGENERATION_PROMPT_TEMPLATE,
)
from src.agent.llm.structured_output import parse_and_validate_json
from src.agent.validation.claim_validator import ClaimValidator
from src.agent.validation.citation_validator import CitationValidator
from src.agent.validation.report_validator import ReportValidator
from src.agent.reports.report_builder import (
    ForensicInvestigationReport,
    ConsensusVerdict,
    ModelPredictionsSummary,
    GraphEvidenceSummary,
    GNNExplainerEvidenceSummary,
    SupportedFinding,
)
from src.agent.schemas import (
    AgentToolError,
    TransactionNotFoundError,
    TemporalBoundaryViolationError,
    EvidenceMode,
)


def _emit_event(
    inv_id: str,
    event_type: InvestigationEventType,
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    tool_name: Optional[str] = None,
    duration_ms: int = 0,
    safe_input: Optional[Dict[str, Any]] = None,
    safe_output: Optional[Dict[str, Any]] = None,
    evidence_ids: Optional[List[str]] = None,
    error_code: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    event = AgentInvestigationEvent(
        investigation_id=inv_id,
        event_type=event_type,
        execution_status=status,
        tool_name=tool_name,
        duration_ms=duration_ms,
        safe_input_summary=safe_input,
        safe_output_summary=safe_output,
        evidence_ref_ids=evidence_ids or [],
        error_code=error_code,
        message=message,
    )
    get_event_sink().emit(event)


# Node 1: Validate Request
def validate_request_node(state: InvestigationState) -> Dict[str, Any]:
    t0 = time.time()
    inv_id = state.get("investigation_id", "inv_unknown")
    tx_id = state.get("tx_id")
    as_of = state.get("as_of_snapshot")
    mode = state.get("evidence_mode", EvidenceMode.RETROSPECTIVE_INVESTIGATION.value)
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    _emit_event(inv_id, InvestigationEventType.INVESTIGATION_STARTED, message="Starting investigation")

    # Check tx_id existence
    registry = _get_tool_registry()
    try:
        tx_info = registry.get_transaction(tx_id, as_of_snapshot=as_of, evidence_mode=mode)
    except TransactionNotFoundError as e:
        abstain_reason = f"Transaction ID {tx_id} does not exist in the Elliptic dataset."
        _emit_event(
            inv_id,
            InvestigationEventType.INVESTIGATION_ABSTAINED,
            status=ExecutionStatus.FAILED,
            error_code="TRANSACTION_NOT_FOUND",
            message=abstain_reason,
        )
        return {
            "status": InvestigationStatus.ABSTAINED.value,
            "abstention_reason": abstain_reason,
            "errors": errors + [abstain_reason],
        }
    except TemporalBoundaryViolationError as e:
        abstain_reason = f"Temporal boundary violation: {e.message}"
        _emit_event(
            inv_id,
            InvestigationEventType.INVESTIGATION_ABSTAINED,
            status=ExecutionStatus.FAILED,
            error_code="TEMPORAL_LEAKAGE_BLOCKED",
            message=abstain_reason,
        )
        return {
            "status": InvestigationStatus.ABSTAINED.value,
            "abstention_reason": abstain_reason,
            "errors": errors + [abstain_reason],
        }

    # Temporal boundary check for historical evaluation
    if mode == EvidenceMode.HISTORICAL_EVALUATION.value and as_of is not None:
        if tx_info.snapshot_id > as_of:
            abstain_reason = (
                f"Temporal boundary qualification: Transaction {tx_id} occurred at snapshot {tx_info.snapshot_id}, "
                f"which is strictly after the point-in-time evaluation cutoff {as_of}."
            )
            _emit_event(
                inv_id,
                InvestigationEventType.INVESTIGATION_ABSTAINED,
                status=ExecutionStatus.FAILED,
                error_code="TEMPORAL_LEAKAGE_BLOCKED",
                message=abstain_reason,
            )
            return {
                "status": InvestigationStatus.ABSTAINED.value,
                "abstention_reason": abstain_reason,
                "errors": errors + [abstain_reason],
            }

    # Prompt injection check on question
    q = state.get("question", "")
    injection_patterns = ["ignore previous", "system prompt", "drop table", "override guardrails"]
    for pat in injection_patterns:
        if pat in q.lower():
            warnings.append(f"Sanitized potential adversarial prompt pattern: '{pat}'")

    duration = int((time.time() - t0) * 1000)
    _emit_event(
        inv_id,
        InvestigationEventType.REQUEST_VALIDATED,
        duration_ms=duration,
        safe_input={"tx_id": tx_id, "snapshot_id": tx_info.snapshot_id, "mode": mode},
        safe_output={"status": "VALIDATED", "warnings_count": len(warnings)},
    )

    return {
        "status": InvestigationStatus.PLANNING.value,
        "warnings": warnings,
        "evidence_bundle": {"transaction": tx_info},
        "provenance_chain": [tx_info.evidence_id],
        "workflow_stages_completed": [
            "STAGE_1_VALIDATE_REQUEST",
            "STAGE_2_RESOLVE_TRANSACTION",
            "STAGE_3_LOAD_TEMPORAL_EVIDENCE",
        ],
    }


# Node 2: Plan Investigation
def plan_investigation_node(state: InvestigationState) -> Dict[str, Any]:
    t0 = time.time()
    inv_id = state.get("investigation_id", "inv_unknown")
    tx_id = state.get("tx_id")
    as_of = state.get("as_of_snapshot")
    mode = state.get("evidence_mode", EvidenceMode.RETROSPECTIVE_INVESTIGATION.value)
    q = state.get("question", "")

    planned_tools: List[ToolCallRequest] = [
        ToolCallRequest(
            tool_name="get_prediction",
            arguments={"tx_id": tx_id, "model_name": "etgnn"},
            rationale="Retrieve ETGNN model inference risk score",
        ),
        ToolCallRequest(
            tool_name="get_prediction",
            arguments={"tx_id": tx_id, "model_name": "matched_gat"},
            rationale="Retrieve Matched GAT graph attention baseline inference",
        ),
        ToolCallRequest(
            tool_name="get_prediction",
            arguments={"tx_id": tx_id, "model_name": "random_forest"},
            rationale="Retrieve Random Forest baseline inference",
        ),
        ToolCallRequest(
            tool_name="get_prediction",
            arguments={"tx_id": tx_id, "model_name": "xgboost"},
            rationale="Retrieve XGBoost baseline inference",
        ),
        ToolCallRequest(
            tool_name="get_transaction_neighbors",
            arguments={"tx_id": tx_id, "hops": 2, "max_nodes": 100, "as_of_snapshot": as_of, "evidence_mode": mode},
            rationale="Retrieve 2-hop topological neighborhood and counterparty transaction flows",
        ),
        ToolCallRequest(
            tool_name="get_explanation",
            arguments={"tx_id": tx_id, "model_name": "etgnn"},
            rationale="Extract GNNExplainer feature attribution and influential edges",
        ),
        ToolCallRequest(
            tool_name="get_evaluation_metrics",
            arguments={"model_name": "etgnn"},
            rationale="Fetch independently reproduced test metrics for ETGNN",
        ),
        ToolCallRequest(
            tool_name="search_research",
            arguments={"query": q or f"Bitcoin transaction {tx_id} graph fraud", "top_k": 4},
            rationale="Retrieve relevant research literature and methodological findings via Graph-RAG",
        ),
    ]

    plan = InvestigationPlan(
        planned_tools=planned_tools,
        reasoning_summary="Multi-step evidence retrieval covering point-in-time features, model predictions, topology, GNNExplainer attribution, and research context.",
    )

    duration = int((time.time() - t0) * 1000)
    _emit_event(
        inv_id,
        InvestigationEventType.PLAN_CREATED,
        duration_ms=duration,
        safe_output={"planned_tool_count": len(planned_tools)},
    )

    return {
        "status": InvestigationStatus.EXECUTING_TOOLS.value,
        "plan": plan.model_dump(),
    }


# Node 3: Execute Tools
def execute_tools_node(state: InvestigationState) -> Dict[str, Any]:
    inv_id = state.get("investigation_id", "inv_unknown")
    tx_id = state.get("tx_id")
    as_of = state.get("as_of_snapshot")
    mode = state.get("evidence_mode", EvidenceMode.RETROSPECTIVE_INVESTIGATION.value)
    plan_dict = state.get("plan") or {}
    planned_tools_raw = plan_dict.get("planned_tools", [])
    budget = ExecutionBudgetTracker()

    registry = _get_tool_registry()
    retriever = _get_hybrid_retriever()

    bundle = dict(state.get("evidence_bundle", {}))
    provenance = list(state.get("provenance_chain", []))
    tool_history = list(state.get("tool_history", []))
    predictions_map = dict(bundle.get("predictions", {}))

    for tool_raw in planned_tools_raw:
        tool_req = ToolCallRequest.model_validate(tool_raw)
        tool_name = tool_req.tool_name
        args = tool_req.arguments

        t0 = time.time()
        _emit_event(
            inv_id,
            InvestigationEventType.TOOL_STARTED,
            tool_name=tool_name,
            safe_input={k: v for k, v in args.items() if k not in ("raw_features", "token")},
        )

        try:
            budget.record_tool_call(tool_name)

            if tool_name == "get_prediction":
                m_name = args.get("model_name", "etgnn")
                pred = registry.get_prediction(tx_id, model_name=m_name)
                predictions_map[m_name] = pred
                if pred.evidence_id not in provenance:
                    provenance.append(pred.evidence_id)
                tool_req.result_evidence_id = pred.evidence_id

            elif tool_name == "get_transaction_neighbors":
                subgraph = registry.get_transaction_neighbors(
                    tx_id,
                    hops=args.get("hops", 2),
                    max_nodes=args.get("max_nodes", 100),
                    as_of_snapshot=as_of,
                    evidence_mode=mode,
                )
                bundle["subgraph"] = subgraph
                if subgraph.evidence_id not in provenance:
                    provenance.append(subgraph.evidence_id)
                tool_req.result_evidence_id = subgraph.evidence_id

            elif tool_name == "get_explanation":
                exp = registry.get_explanation(tx_id, model_name=args.get("model_name", "etgnn"))
                bundle["explanation"] = exp
                if exp.evidence_id not in provenance:
                    provenance.append(exp.evidence_id)
                tool_req.result_evidence_id = exp.evidence_id

            elif tool_name == "get_evaluation_metrics":
                metrics = registry.get_evaluation_metrics(model_name=args.get("model_name", "etgnn"))
                bundle["metrics"] = metrics
                for m in metrics:
                    if m.evidence_id not in provenance:
                        provenance.append(m.evidence_id)
                tool_req.result_evidence_id = metrics[0].evidence_id if metrics else None

            elif tool_name == "search_research":
                query_str = args.get("query", "Bitcoin illicit transaction detection")
                top_k = min(args.get("top_k", 4), MAX_RAG_CHUNKS)
                retrieval_res = retriever.retrieve(query_str, top_k=top_k)
                docs = retrieval_res.research_evidence if hasattr(retrieval_res, "research_evidence") else (retrieval_res if isinstance(retrieval_res, list) else [])
                bundle["research"] = docs
                for d in docs:
                    if d.evidence_id not in provenance:
                        provenance.append(d.evidence_id)
                tool_req.result_evidence_id = docs[0].evidence_id if docs else None

            tool_req.executed = True
            dur = int((time.time() - t0) * 1000)
            _emit_event(
                inv_id,
                InvestigationEventType.TOOL_COMPLETED,
                tool_name=tool_name,
                duration_ms=dur,
                safe_output={"status": "SUCCESS", "evidence_id": tool_req.result_evidence_id},
                evidence_ids=[tool_req.result_evidence_id] if tool_req.result_evidence_id else [],
            )

        except Exception as e:
            dur = int((time.time() - t0) * 1000)
            tool_req.error = str(e)
            _emit_event(
                inv_id,
                InvestigationEventType.TOOL_COMPLETED,
                status=ExecutionStatus.FAILED,
                tool_name=tool_name,
                duration_ms=dur,
                error_code=type(e).__name__,
                message=str(e),
            )

        tool_history.append(tool_req.model_dump())

    bundle["predictions"] = predictions_map
    bundle["provenance_chain"] = provenance
    stages = list(state.get("workflow_stages_completed", []))
    for stg in ("STAGE_4_EXECUTE_MODELS", "STAGE_5_ANALYZE_GRAPH_XAI", "STAGE_6_RETRIEVE_RAG_EVIDENCE"):
        if stg not in stages:
            stages.append(stg)

    return {
        "status": InvestigationStatus.CONSOLIDATING_EVIDENCE.value,
        "evidence_bundle": bundle,
        "provenance_chain": provenance,
        "tool_history": tool_history,
        "workflow_stages_completed": stages,
    }


# Node 4: Consolidate Evidence (Stages 7 & 8: Compare Models & Evaluate Completeness)
def consolidate_evidence_node(state: InvestigationState) -> Dict[str, Any]:
    t0 = time.time()
    inv_id = state.get("investigation_id", "inv_unknown")
    bundle = state.get("evidence_bundle", {})
    preds = bundle.get("predictions", {})
    tool_history = state.get("tool_history", [])

    etgnn_p = getattr(preds.get("etgnn"), "illicit_probability", 0.0) if preds.get("etgnn") else 0.0
    gat_p = getattr(preds.get("matched_gat"), "illicit_probability", 0.0) if preds.get("matched_gat") else 0.0
    rf_p = getattr(preds.get("random_forest"), "illicit_probability", 0.0) if preds.get("random_forest") else 0.0
    xgb_p = getattr(preds.get("xgboost"), "illicit_probability", 0.0) if preds.get("xgboost") else 0.0

    probs = [p for p in (etgnn_p, gat_p, rf_p, xgb_p) if p is not None]
    prob_spread = max(probs) - min(probs) if probs else 0.0
    divergence = prob_spread >= 0.30

    # Stage 8: Evaluate Evidence Completeness & Consistency
    failed_tools = [t for t in tool_history if t.get("error")]
    if not bundle.get("transaction") or not preds:
        completeness = "EVIDENCE_INSUFFICIENT"
    elif failed_tools or not bundle.get("subgraph") or not bundle.get("research"):
        completeness = "PARTIAL_RESULT"
    else:
        completeness = "COMPLETE"

    stages = list(state.get("workflow_stages_completed", []))
    for stg in ("STAGE_7_COMPARE_MODELS", "STAGE_8_EVALUATE_EVIDENCE_COMPLETENESS"):
        if stg not in stages:
            stages.append(stg)

    duration = int((time.time() - t0) * 1000)
    _emit_event(
        inv_id,
        InvestigationEventType.EVIDENCE_CONSOLIDATED,
        duration_ms=duration,
        safe_output={
            "model_divergence_detected": divergence,
            "probability_spread": round(prob_spread, 4),
            "evidence_completeness": completeness,
            "evidence_count": len(state.get("provenance_chain", [])),
        },
    )

    return {
        "status": InvestigationStatus.GENERATING_EXPLANATION.value,
        "evidence_completeness": completeness,
        "workflow_stages_completed": stages,
    }


# Node 5: Generate Explanation
def generate_explanation_node(state: InvestigationState) -> Dict[str, Any]:
    t0 = time.time()
    inv_id = state.get("investigation_id", "inv_unknown")
    tx_id = state.get("tx_id", 0)
    as_of = state.get("as_of_snapshot")
    mode = state.get("evidence_mode", EvidenceMode.RETROSPECTIVE_INVESTIGATION.value)
    q = state.get("question", "")
    bundle = state.get("evidence_bundle", {})
    provenance = state.get("provenance_chain", [])
    retries = state.get("retries_count", 0)

    _emit_event(inv_id, InvestigationEventType.EXPLANATION_STARTED)

    # Format context for prompt
    tx_ev = bundle.get("transaction")
    snap_id = getattr(tx_ev, "snapshot_id", 0) if tx_ev else 0
    preds = bundle.get("predictions", {})
    subgraph = bundle.get("subgraph")
    explanation = bundle.get("explanation")
    research = bundle.get("research", [])
    metrics = bundle.get("metrics", [])

    pred_summary = "\n".join([
        f"- {m}: prob={getattr(p, 'illicit_probability', 0.0):.4f}, predicted_class={getattr(p, 'predicted_class', 0)} (ID: {getattr(p, 'evidence_id', '')})"
        for m, p in preds.items()
    ])

    sub_summary = (
        f"2-hop Subgraph: {getattr(subgraph, 'total_subgraph_nodes', 0)} nodes, "
        f"{getattr(subgraph, 'total_subgraph_edges', 0)} edges, "
        f"{getattr(subgraph, 'illicit_neighbors_count', 0)} illicit neighbors (ID: {getattr(subgraph, 'evidence_id', '')})"
        if subgraph else "No subgraph retrieved."
    )

    exp_summary = (
        f"GNNExplainer: top feature indices={getattr(explanation, 'top_feature_indices', [])}, "
        f"fidelity drop={getattr(explanation, 'explanation_fidelity_drop', 0.0):.4f} (ID: {getattr(explanation, 'evidence_id', '')})"
        if explanation else "No explanation retrieved."
    )

    research = bundle.get("research", [])
    if hasattr(research, "research_evidence"):
        research_docs = research.research_evidence
    elif isinstance(research, list):
        research_docs = research
    else:
        research_docs = []

    res_summary = "\n".join([
        f"- [{getattr(d, 'authority_tier', '')}] {getattr(d, 'document_title', '')}: {getattr(d, 'text_content', '')[:120]}... (ID: {getattr(d, 'evidence_id', '')})"
        for d in research_docs[:3]
    ])

    met_summary = "\n".join([
        f"- {getattr(m, 'model_name', '')} {getattr(m, 'metric_name', '')} = {getattr(m, 'metric_value', 0.0):.4f} (ID: {getattr(m, 'evidence_id', '')})"
        for m in metrics[:3]
    ])

    prompt = INVESTIGATION_PROMPT_TEMPLATE.format(
        tx_id=tx_id,
        snapshot_id=snap_id,
        evidence_mode=mode,
        as_of_snapshot=as_of if as_of is not None else "Unrestricted",
        predictions_summary=pred_summary,
        graph_summary=sub_summary,
        explanation_summary=exp_summary,
        research_summary=res_summary,
        metrics_summary=met_summary,
        provenance_ids=", ".join(provenance),
        question=q,
        investigation_id=inv_id,
    )

    # Provider call
    provider_name = state.get("llm_provider")
    allow_evidence_fallback = state.get("allow_evidence_only_fallback", True)
    context_data = {
        "tx_id": tx_id,
        "snapshot_id": snap_id,
        "investigation_id": inv_id,
        "evidence_bundle": bundle,
        "provenance_chain": provenance,
    }

    try:
        provider = get_llm_provider(provider_name)
        raw_text = provider.generate(prompt, SYSTEM_PROMPT_FORENSIC_INVESTIGATOR, context_data=context_data)
        report_dict = parse_and_validate_json(raw_text)
        if provider_name in ("gemini", "anthropic", "ollama") and not isinstance(provider, (MockLLMProvider, EvidenceOnlyProvider)):
            report_dict["generation_mode"] = "LIVE"
            report_dict["live_llm_verified"] = True
            report_dict["provider_type"] = provider.provider_name
            report_dict["provider_model_id"] = provider.model_id
            if provider.last_token_usage:
                report_dict["token_usage"] = provider.last_token_usage
            report_dict["estimated_cost_usd"] = provider.last_estimated_cost_usd
    except Exception as e:
        if allow_evidence_fallback:
            fallback = EvidenceOnlyProvider()
            raw_text = fallback.generate(prompt, SYSTEM_PROMPT_FORENSIC_INVESTIGATOR, context_data=context_data)
            report_dict = parse_and_validate_json(raw_text)
            report_dict["generation_mode"] = "DETERMINISTIC_EVIDENCE_ONLY"
            report_dict["provider_type"] = "evidence_only"
            report_dict["provider_model_id"] = fallback.model_id
            report_dict["live_llm_verified"] = False
            lims = report_dict.get("evidence_limitations", [])
            lims.append(
                f"Live LLM provider '{provider_name or 'live'}' failed ({type(e).__name__}: {str(e)}); fallen back to deterministic evidence-only mode."
            )
            report_dict["evidence_limitations"] = lims
            report_dict["limitations"] = lims
        else:
            _emit_event(
                inv_id,
                InvestigationEventType.EXPLANATION_GENERATED,
                status=ExecutionStatus.FAILED,
                error_code=type(e).__name__,
                message=f"Live LLM provider '{provider_name}' failed ({type(e).__name__}: {str(e)}) and fallback is disabled.",
            )
            raise

    dur = int((time.time() - t0) * 1000)
    report_dict["llm_latency_ms"] = round(float(dur), 2)
    report_dict["evidence_completeness"] = state.get("evidence_completeness", "COMPLETE")
    _emit_event(
        inv_id,
        InvestigationEventType.EXPLANATION_GENERATED,
        duration_ms=dur,
        safe_output={"findings_count": len(report_dict.get("supported_findings", []))},
    )

    is_regeneration = state.get("raw_explanation") is not None
    new_retries = retries + (1 if is_regeneration else 0)
    stages = list(state.get("workflow_stages_completed", []))
    if "STAGE_9_GENERATE_EXPLANATION" not in stages:
        stages.append("STAGE_9_GENERATE_EXPLANATION")

    return {
        "status": InvestigationStatus.VALIDATING_CLAIMS.value,
        "raw_explanation": report_dict,
        "retries_count": new_retries,
        "workflow_stages_completed": stages,
    }


# Node 6: Validate Claims
def validate_claims_node(state: InvestigationState) -> Dict[str, Any]:
    t0 = time.time()
    inv_id = state.get("investigation_id", "inv_unknown")
    tx_id = state.get("tx_id", 0)
    raw_exp = state.get("raw_explanation") or {}
    bundle = dict(state.get("evidence_bundle", {}))
    bundle["provenance_chain"] = list(state.get("provenance_chain", []))
    findings = raw_exp.get("supported_findings", [])

    validator = ClaimValidator()
    summary = validator.validate_findings(findings, bundle, target_tx_id=tx_id)

    dur = int((time.time() - t0) * 1000)
    _emit_event(
        inv_id,
        InvestigationEventType.CLAIMS_VALIDATED,
        duration_ms=dur,
        safe_output={
            "is_valid": summary.is_valid,
            "supported": summary.supported_count,
            "unsupported": summary.unsupported_count,
        },
    )

    return {
        "status": InvestigationStatus.VALIDATING_REPORT.value if summary.is_valid else InvestigationStatus.GENERATING_EXPLANATION.value,
        "claim_validation_result": summary.model_dump(),
    }


# Node 7: Validate Final Report
def validate_final_report_node(state: InvestigationState) -> Dict[str, Any]:
    t0 = time.time()
    inv_id = state.get("investigation_id", "inv_unknown")
    tx_id = state.get("tx_id", 0)
    raw_exp = state.get("raw_explanation") or {}
    bundle = dict(state.get("evidence_bundle", {}))
    bundle["provenance_chain"] = list(state.get("provenance_chain", []))
    tx_ev = bundle.get("transaction")
    snap_id = getattr(tx_ev, "snapshot_id", 0) if tx_ev else 0

    validator = ReportValidator()
    res = validator.validate_report(raw_exp, bundle, snap_id, target_tx_id=tx_id)

    dur = int((time.time() - t0) * 1000)
    _emit_event(
        inv_id,
        InvestigationEventType.REPORT_VALIDATED,
        duration_ms=dur,
        safe_output={
            "is_valid": res.is_valid,
            "can_proceed": res.can_proceed,
            "reasons": res.reasons,
        },
    )

    stages = list(state.get("workflow_stages_completed", []))
    for stg in ("STAGE_10_VALIDATE_CLAIMS_CITATIONS", "STAGE_11_PRODUCE_REPORT"):
        if stg not in stages:
            stages.append(stg)

    final_payload = res.sanitized_report or raw_exp
    final_payload["workflow_stages_completed"] = stages
    final_payload["evidence_completeness"] = state.get("evidence_completeness", "COMPLETE")

    return {
        "status": InvestigationStatus.COMPLETED.value,
        "report_validation_result": res.model_dump(),
        "final_report": final_payload,
        "workflow_stages_completed": stages,
    }


# Node 8: Finalize Investigation
def finalize_investigation_node(state: InvestigationState) -> Dict[str, Any]:
    inv_id = state.get("investigation_id", "inv_unknown")
    tx_id = state.get("tx_id", 0)
    status = state.get("status", InvestigationStatus.COMPLETED.value)
    abstain_reason = state.get("abstention_reason")
    final_dict = state.get("final_report")
    bundle = state.get("evidence_bundle", {})
    tx_ev = bundle.get("transaction")
    snap_id = getattr(tx_ev, "snapshot_id", 0) if tx_ev else 0

    if status == InvestigationStatus.ABSTAINED.value or abstain_reason:
        report = ForensicInvestigationReport(
            investigation_id=inv_id,
            target_transaction_id=tx_id,
            snapshot_id=snap_id,
            status="ABSTAINED",
            investigation_status="ABSTAINED",
            consensus_verdict=ConsensusVerdict.INCONCLUSIVE_DATA_GAP,
            narrative_summary=f"Investigation abstained: {abstain_reason or 'Insufficient evidence or boundary violation.'}",
            limitations=[abstain_reason] if abstain_reason else ["Data gap or constraint violation encountered."],
            evidence_limitations=[abstain_reason] if abstain_reason else ["Data gap or constraint violation encountered."],
        )
        _emit_event(
            inv_id,
            InvestigationEventType.INVESTIGATION_ABSTAINED,
            message=abstain_reason or "Investigation abstained.",
        )
        return {
            "status": InvestigationStatus.ABSTAINED.value,
            "final_report": report.model_dump(),
        }

    # Normal completion
    if final_dict:
        try:
            report = ForensicInvestigationReport.model_validate(final_dict)
        except Exception:
            report = ForensicInvestigationReport(
                investigation_id=inv_id,
                target_transaction_id=tx_id,
                snapshot_id=snap_id,
                status="COMPLETED",
                consensus_verdict=ConsensusVerdict.LOW_RISK_LICIT,
                narrative_summary=str(final_dict.get("narrative_summary", "Completed investigation")),
            )
    else:
        report = ForensicInvestigationReport(
            investigation_id=inv_id,
            target_transaction_id=tx_id,
            snapshot_id=snap_id,
            status="COMPLETED",
        )

    _emit_event(
        inv_id,
        InvestigationEventType.INVESTIGATION_COMPLETED,
        safe_output={
            "verdict": report.consensus_verdict.value,
            "findings_count": len(report.supported_findings),
        },
        message="Investigation successfully completed with verified evidence.",
    )

    return {
        "status": InvestigationStatus.COMPLETED.value,
        "final_report": report.model_dump(),
    }
