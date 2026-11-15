"""
FastAPI Application for ETGNN Research Demonstration Platform.
Backed strictly by real repository artifacts, checkpoints, and Elliptic dataset.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from fastapi import FastAPI, HTTPException, Path as FastAPIPath, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.web.data_loader import get_data_loader

ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = ROOT / "web" / "frontend" / "dist"

app = FastAPI(
    title="ETGNN Research Demonstration Platform API",
    description="Academic research demonstration platform for Graph Neural Networks in Bitcoin fraud detection.",
    version="1.0.0",
)

import math

# Restrict CORS to local development and local server instances
LOCAL_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers_middleware(request, call_next):
    """Enforce strict defensive security headers across all API endpoints."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


class PredictRequest(BaseModel):
    transaction_id: int = Field(
        ...,
        ge=0,
        le=2**63 - 1,
        description="Elliptic Bitcoin transaction ID",
    )
    features_override: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional feature adjustments (e.g. {'feat_0': 1.5}) to test model sensitivity",
    )

    @classmethod
    def validate_features(cls, values):
        return values

    def model_post_init(self, __context: Any) -> None:
        if self.features_override:
            if len(self.features_override) > 165:
                raise ValueError("features_override cannot exceed 165 feature dimensions")
            for k, val in self.features_override.items():
                if not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"Feature override value for {k} must be a finite float")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Health check and environment status."""
    loader = get_data_loader()
    return {
        "status": "healthy",
        "system": "ETGNN Research Demonstration Platform",
        "dataset": "Elliptic Bitcoin Dataset (203,769 transactions)",
        "models_loaded": ["ETGNN", "Matched_GAT", "Residual_ETGNN_93f", "RandomForest", "XGBoost"],
        "epistemic_notice": (
            "This platform is an academic research demonstration system for peer review and thesis defense. "
            "It is not a financial crime compliance service or automated decision-making tool."
        ),
    }


@app.get("/api/models")
def get_models() -> List[Dict[str, Any]]:
    """Get metadata and architectural specifications for evaluated models."""
    loader = get_data_loader()
    return loader.get_models_metadata()


@app.get("/api/metrics")
def get_metrics() -> Dict[str, Any]:
    """Get official held-out test evaluation results."""
    loader = get_data_loader()
    return loader.get_benchmark_metrics()


def _unwrap_param(val, default=None):
    """Safely unwrap FastAPI Query or Path parameter defaults when handlers are called in-process."""
    if hasattr(val, "default"):
        d = val.default
        return default if d is ... else d
    return default if val is None else val


@app.get("/api/transactions")
def get_transactions(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    split: Optional[str] = Query(None, description="train, validation, test, or unlabelled"),
    label: Optional[int] = Query(None, description="0 (licit), 1 (illicit), or -1 (unknown)"),
    time_step: Optional[int] = Query(None, ge=1, le=49),
    search: Optional[str] = Query(None, description="Substring search on transaction ID"),
) -> Dict[str, Any]:
    """Paginated listing of Elliptic transactions with filtering."""
    loader = get_data_loader()
    return loader.get_nodes_page(
        limit=int(_unwrap_param(limit, 25)),
        offset=int(_unwrap_param(offset, 0)),
        split=_unwrap_param(split, None),
        label=_unwrap_param(label, None),
        time_step=_unwrap_param(time_step, None),
        search=_unwrap_param(search, None),
    )


@app.get("/api/transactions/{transaction_id}")
def get_transaction(
    transaction_id: int = FastAPIPath(..., ge=0, le=2**63 - 1, description="Elliptic Bitcoin transaction ID")
) -> Dict[str, Any]:
    """Get complete details, degree, features, and model scores for a single transaction."""
    loader = get_data_loader()
    actual_tx = int(_unwrap_param(transaction_id, transaction_id))
    try:
        return loader.get_transaction(actual_tx)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@app.get("/api/transactions/{transaction_id}/subgraph")
def get_transaction_subgraph(
    transaction_id: int = FastAPIPath(..., ge=0, le=2**63 - 1, description="Elliptic Bitcoin transaction ID"),
    hops: int = Query(2, ge=1, le=2, description="Hop neighborhood radius (1 or 2)"),
    max_nodes: int = Query(150, ge=10, le=300),
) -> Dict[str, Any]:
    """Extract local k-hop transaction neighborhood for graph visualization."""
    loader = get_data_loader()
    actual_tx = int(_unwrap_param(transaction_id, transaction_id))
    actual_hops = int(_unwrap_param(hops, 2))
    actual_max = int(_unwrap_param(max_nodes, 150))
    try:
        return loader.get_subgraph(actual_tx, num_hops=actual_hops, max_nodes=actual_max)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@app.post("/api/predict")
def predict_transaction(payload: PredictRequest) -> Dict[str, Any]:
    """Execute live forward pass across all models for a transaction."""
    loader = get_data_loader()
    try:
        return loader.predict(
            tx_id=payload.transaction_id,
            features_override=payload.features_override,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@app.get("/api/explanations/{transaction_id}")
def get_explanation(
    transaction_id: int = FastAPIPath(..., ge=0, le=2**63 - 1, description="Elliptic Bitcoin transaction ID")
) -> Dict[str, Any]:
    """Retrieve frozen research explanation or compute demonstration explanation."""
    loader = get_data_loader()
    actual_tx = int(_unwrap_param(transaction_id, transaction_id))
    try:
        return loader.get_explanation(actual_tx)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@app.get("/api/transactions/{transaction_id}/intervention")
def get_transaction_intervention(
    transaction_id: int = FastAPIPath(..., ge=0, le=2**63 - 1, description="Elliptic Bitcoin transaction ID")
) -> Dict[str, Any]:
    """Execute controlled edge deletion intervention on frozen ETGNN model."""
    loader = get_data_loader()
    actual_tx = int(_unwrap_param(transaction_id, transaction_id))
    try:
        return loader.execute_intervention(actual_tx)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@app.get("/api/research/c1")
def get_research_c1() -> Dict[str, Any]:
    """Get frozen C1 experimental summary data (ETGNN vs Matched GAT)."""
    loader = get_data_loader()
    return loader.get_c1_summary()


@app.get("/api/research/c2")
def get_research_c2() -> Dict[str, Any]:
    """Get frozen C2 experimental summary data across 8 registers."""
    loader = get_data_loader()
    return loader.get_c2_summary()


@app.get("/api/research/ablations")
def get_research_ablations() -> Dict[str, Any]:
    """Get multi-seed ablation and edge-removal anomaly data."""
    loader = get_data_loader()
    return loader.get_ablations_data()


@app.get("/api/research/phase4")
def get_research_phase4() -> Dict[str, Any]:
    """Get verified 40-trial Phase 4 factorial results and Holm-adjusted paired contrasts."""
    loader = get_data_loader()
    return loader.get_phase4_summary()


@app.get("/api/research/xai_validation")
def get_research_xai_validation() -> Dict[str, Any]:
    """Get Phase H Explainable AI validation metrics on frozen 93f ResidualETGNNClassifier."""
    loader = get_data_loader()
    return loader.get_xai_validation_summary()


@app.get("/api/research/external_validation")
def get_research_external_validation() -> Dict[str, Any]:
    """Get Phase 5 external validation results and statistics on Elliptic++ Actors (56f)."""
    loader = get_data_loader()
    return loader.get_external_validation_summary()


@app.get("/api/research/temporal_stability")
def get_research_temporal_stability() -> Dict[str, Any]:
    """Get Phase 12 Experiment A candidate-edge shift census and multi-seed stability audit."""
    loader = get_data_loader()
    try:
        return loader.get_temporal_stability_summary()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.get("/api/research/error_mitigation")
def get_research_error_mitigation() -> Dict[str, Any]:
    """Get Phase 12 Experiment B prospective error mitigation auditing across 21 strategies."""
    loader = get_data_loader()
    try:
        return loader.get_error_mitigation_summary()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ============================================================================
# Phase 2: Agent Evidence and Hybrid Graph-RAG Endpoints
# ============================================================================

class AgentRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Research or forensic inquiry")
    top_k: int = Field(default=5, ge=1, le=20, description="Max chunks to retrieve")
    evidence_mode: str = Field(default="HISTORICAL_EVALUATION", description="HISTORICAL_EVALUATION or RETROSPECTIVE_INVESTIGATION")
    as_of_snapshot: Optional[int] = Field(default=None, ge=1, le=49, description="Simulation cutoff snapshot")


class AgentInvestigateContextRequest(BaseModel):
    transaction_id: int = Field(..., ge=0, le=2**63 - 1, description="Target transaction ID")
    question: str = Field(default="Forensic fraud risk and graph context investigation", max_length=500)
    evidence_mode: str = Field(default="HISTORICAL_EVALUATION")
    as_of_snapshot: Optional[int] = Field(default=None, ge=1, le=49)
    top_k_research: int = Field(default=4, ge=1, le=10)


class AgentInvestigateRequest(BaseModel):
    transaction_id: int = Field(..., ge=0, le=2**63 - 1, description="Target transaction ID")
    question: Optional[str] = Field(default=None, max_length=1000)
    evidence_mode: str = Field(default="RETROSPECTIVE_INVESTIGATION")
    as_of_snapshot: Optional[int] = Field(default=None, ge=1, le=49)
    llm_provider: Optional[str] = Field(default=None, description="LLM provider: mock, evidence_only, gemini, anthropic, ollama")
    allow_evidence_only_fallback: bool = Field(default=True, description="Fallback to evidence-only if live provider fails")


@app.get("/api/agent/evidence/transaction/{tx_id}")
def api_get_agent_transaction(
    tx_id: int = FastAPIPath(..., ge=0, le=2**63 - 1, description="Elliptic Bitcoin transaction ID"),
    evidence_mode: str = Query("HISTORICAL_EVALUATION"),
    as_of_snapshot: Optional[int] = Query(None, ge=1, le=49),
) -> Dict[str, Any]:
    """Retrieve structured transaction evidence with point-in-time protection."""
    from src.agent.tools.investigation_tools import get_transaction
    from src.agent.schemas import AgentToolError
    try:
        ev = get_transaction(tx_id, evidence_mode=evidence_mode, as_of_snapshot=as_of_snapshot)
        return ev.model_dump()
    except AgentToolError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err.message)
    except Exception as exc:
        from src.agent.llm.provider import (
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMProviderError,
        )
        if isinstance(exc, LLMAuthenticationError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "LLMAuthenticationError", "message": str(exc)},
            )
        elif isinstance(exc, LLMRateLimitError):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error_code": "LLMRateLimitError", "message": str(exc)},
            )
        elif isinstance(exc, LLMTimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"error_code": "LLMTimeoutError", "message": str(exc)},
            )
        elif isinstance(exc, LLMProviderError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error_code": "LLMProviderError", "message": str(exc)},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": type(exc).__name__, "message": str(exc)},
        )


@app.get("/api/agent/evidence/graph/{tx_id}")
def api_get_agent_graph(
    tx_id: int = FastAPIPath(..., ge=0, le=2**63 - 1, description="Elliptic Bitcoin transaction ID"),
    hops: int = Query(1, ge=1, le=2),
    max_nodes: int = Query(50, ge=1, le=100),
    evidence_mode: str = Query("HISTORICAL_EVALUATION"),
    as_of_snapshot: Optional[int] = Query(None, ge=1, le=49),
) -> Dict[str, Any]:
    """Retrieve directed neighborhood graph evidence with resource caps."""
    from src.agent.tools.investigation_tools import get_transaction_neighbors
    from src.agent.schemas import AgentToolError
    try:
        ev = get_transaction_neighbors(
            tx_id, hops=hops, max_nodes=max_nodes, evidence_mode=evidence_mode, as_of_snapshot=as_of_snapshot
        )
        return ev.model_dump()
    except AgentToolError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err.message)
    except Exception as exc:
        from src.agent.llm.provider import (
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMProviderError,
        )
        if isinstance(exc, LLMAuthenticationError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "LLMAuthenticationError", "message": str(exc)},
            )
        elif isinstance(exc, LLMRateLimitError):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error_code": "LLMRateLimitError", "message": str(exc)},
            )
        elif isinstance(exc, LLMTimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"error_code": "LLMTimeoutError", "message": str(exc)},
            )
        elif isinstance(exc, LLMProviderError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error_code": "LLMProviderError", "message": str(exc)},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": type(exc).__name__, "message": str(exc)},
        )


@app.post("/api/agent/retrieve")
def api_agent_retrieve(req: AgentRetrieveRequest) -> Dict[str, Any]:
    """Hybrid Graph-RAG retrieval over approved research documents."""
    from src.agent.rag.retriever import HybridRetriever
    from src.agent.schemas import EvidenceMode
    try:
        retriever = HybridRetriever()
        mode = EvidenceMode(req.evidence_mode)
        res = retriever.retrieve(
            query=req.query,
            top_k=req.top_k,
            evidence_mode=mode,
            as_of_snapshot=req.as_of_snapshot,
        )
        return res.model_dump()
    except Exception as exc:
        from src.agent.llm.provider import (
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMProviderError,
        )
        if isinstance(exc, LLMAuthenticationError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "LLMAuthenticationError", "message": str(exc)},
            )
        elif isinstance(exc, LLMRateLimitError):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error_code": "LLMRateLimitError", "message": str(exc)},
            )
        elif isinstance(exc, LLMTimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"error_code": "LLMTimeoutError", "message": str(exc)},
            )
        elif isinstance(exc, LLMProviderError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error_code": "LLMProviderError", "message": str(exc)},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": type(exc).__name__, "message": str(exc)},
        )


@app.post("/api/agent/investigate-context")
def api_agent_investigate_context(req: AgentInvestigateContextRequest) -> Dict[str, Any]:
    """Build unified investigation context combining graph, models, explanations, and RAG."""
    from src.agent.services.investigation_context_service import InvestigationContextService
    from src.agent.schemas import AgentToolError
    try:
        service = InvestigationContextService()
        ctx = service.build_investigation_context(
            tx_id=req.transaction_id,
            question=req.question,
            evidence_mode=req.evidence_mode,
            as_of_snapshot=req.as_of_snapshot,
            top_k_research=req.top_k_research,
        )
        return ctx.model_dump()
    except AgentToolError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err.message)
    except Exception as exc:
        from src.agent.llm.provider import (
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMProviderError,
        )
        if isinstance(exc, LLMAuthenticationError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "LLMAuthenticationError", "message": str(exc)},
            )
        elif isinstance(exc, LLMRateLimitError):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error_code": "LLMRateLimitError", "message": str(exc)},
            )
        elif isinstance(exc, LLMTimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"error_code": "LLMTimeoutError", "message": str(exc)},
            )
        elif isinstance(exc, LLMProviderError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error_code": "LLMProviderError", "message": str(exc)},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": type(exc).__name__, "message": str(exc)},
        )


# ============================================================================
# Phase 3: LangGraph Autonomous Investigation & Observable Telemetry Endpoints
# ============================================================================

@app.post("/api/agent/investigate")
def api_agent_investigate(req: AgentInvestigateRequest) -> Dict[str, Any]:
    """Execute LangGraph multi-step investigation workflow and return forensic report."""
    from src.agent.services.investigation_runner import get_investigation_runner
    try:
        runner = get_investigation_runner()
        result = runner.run_investigation(
            tx_id=req.transaction_id,
            question=req.question,
            evidence_mode=req.evidence_mode,
            as_of_snapshot=req.as_of_snapshot,
            llm_provider=req.llm_provider,
            allow_evidence_only_fallback=req.allow_evidence_only_fallback,
        )
        return {
            "investigation_id": result.get("investigation_id"),
            "status": result.get("status"),
            "report": result.get("final_report"),
            "warnings": result.get("warnings", []),
            "errors": result.get("errors", []),
        }
    except Exception as exc:
        from src.agent.llm.provider import (
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMProviderError,
        )
        if isinstance(exc, LLMAuthenticationError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "LLMAuthenticationError", "message": str(exc)},
            )
        elif isinstance(exc, LLMRateLimitError):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error_code": "LLMRateLimitError", "message": str(exc)},
            )
        elif isinstance(exc, LLMTimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"error_code": "LLMTimeoutError", "message": str(exc)},
            )
        elif isinstance(exc, LLMProviderError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error_code": "LLMProviderError", "message": str(exc)},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": type(exc).__name__, "message": str(exc)},
        )


@app.get("/api/agent/investigations/{investigation_id}")
def api_get_investigation(
    investigation_id: str = FastAPIPath(..., min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_\-]+$")
) -> Dict[str, Any]:
    """Retrieve full investigation report and execution status."""
    actual_id = str(_unwrap_param(investigation_id, investigation_id))
    from src.agent.services.investigation_runner import get_investigation_runner
    runner = get_investigation_runner()
    inv = runner.get_investigation(actual_id)
    if not inv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation ID '{actual_id}' not found.",
        )
    return {
        "investigation_id": inv.get("investigation_id"),
        "status": inv.get("status"),
        "report": inv.get("final_report"),
        "warnings": inv.get("warnings", []),
        "errors": inv.get("errors", []),
    }


@app.get("/api/agent/investigations/{investigation_id}/events")
def api_get_investigation_events(
    investigation_id: str = FastAPIPath(..., min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_\-]+$")
) -> List[Dict[str, Any]]:
    """Retrieve observable telemetry event stream for an investigation trace."""
    actual_id = str(_unwrap_param(investigation_id, investigation_id))
    from src.agent.telemetry.event_sink import get_event_sink
    sink = get_event_sink()
    events = sink.get_events(actual_id)
    return [e.model_dump() for e in events]


# ============================================================================
# Phase 4: Investigation History, Chat, Feedback & PDF Export
# ============================================================================

@app.get("/api/agent/investigations")
def api_list_investigations() -> List[Dict[str, Any]]:
    """List session-cached investigations with key analytical summaries."""
    from src.agent.services.investigation_runner import get_investigation_runner
    runner = get_investigation_runner()
    inv_ids = runner.list_investigations()
    summaries = []
    for iid in inv_ids:
        inv = runner.get_investigation(iid)
        if inv:
            report = inv.get("final_report") or {}
            summaries.append({
                "investigation_id": iid,
                "transaction_id": inv.get("tx_id"),
                "status": inv.get("status"),
                "consensus_verdict": report.get("consensus_verdict"),
                "risk_indicator": report.get("risk_indicator"),
                "model_disagreement": report.get("model_disagreement", False),
                "probability_spread": report.get("probability_spread", 0.0),
                "generation_mode": report.get("generation_mode", "SIMULATED"),
                "provider_type": report.get("provider_type", "mock"),
                "created_at": report.get("generated_at"),
                "session_note": "Session-limited in-memory cache (max 100 entries)",
            })
    return summaries


@app.get("/api/agent/investigations/{investigation_id}/export/pdf")
def api_export_investigation_pdf(
    investigation_id: str = FastAPIPath(..., min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_\-]+$")
) -> Response:
    """Export publication-quality forensic investigation PDF report."""
    if ".." in investigation_id or "/" in investigation_id or "\\" in investigation_id or "%" in investigation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid characters in investigation ID.",
        )
    from src.agent.services.investigation_runner import get_investigation_runner
    from src.agent.reports.pdf_exporter import generate_investigation_pdf
    runner = get_investigation_runner()
    inv = runner.get_investigation(investigation_id)
    if not inv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation '{investigation_id}' not found.",
        )
    report = inv.get("final_report")
    if not report:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Investigation has not generated a final report.",
        )

    pdf_bytes = generate_investigation_pdf(report)
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "", investigation_id)
    filename = f"ETGNN_Forensic_Report_{clean_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/agent/chat")
def api_agent_chat(req: Dict[str, Any]) -> Dict[str, Any]:
    """Interactive follow-up Q&A grounded strictly in investigation evidence bundle."""
    from src.agent.services.investigation_chat_service import get_chat_service, ChatRequest
    chat_service = get_chat_service()
    try:
        chat_req = ChatRequest(**req)
        res = chat_service.answer_question(chat_req)
        return res.model_dump()
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))
    except Exception as exc:
        from src.agent.llm.provider import (
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMProviderError,
        )
        if isinstance(exc, LLMAuthenticationError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "LLMAuthenticationError", "message": str(exc)},
            )
        elif isinstance(exc, LLMRateLimitError):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error_code": "LLMRateLimitError", "message": str(exc)},
            )
        elif isinstance(exc, LLMTimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"error_code": "LLMTimeoutError", "message": str(exc)},
            )
        elif isinstance(exc, LLMProviderError):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error_code": "LLMProviderError", "message": str(exc)},
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": type(exc).__name__, "message": str(exc)},
        )


@app.post("/api/agent/feedback")
def api_submit_feedback(req: Dict[str, Any]) -> Dict[str, Any]:
    """Submit human-in-the-loop analyst review feedback on investigation quality."""
    from src.agent.feedback.store import get_feedback_store, FeedbackSubmission
    store = get_feedback_store()
    try:
        sub = FeedbackSubmission(**req)
        record = store.submit_feedback(sub)
        return record.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.get("/api/agent/feedback")
def api_list_feedback(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    """Retrieve persisted analyst review feedback records and summary statistics."""
    from src.agent.feedback.store import get_feedback_store
    store = get_feedback_store()
    actual_limit = getattr(limit, "default", limit)
    actual_offset = getattr(offset, "default", offset)
    items = store.list_all_feedback(limit=int(actual_limit), offset=int(actual_offset))
    stats = store.get_feedback_stats()
    return {
        "stats": stats,
        "items": [i.model_dump() for i in items],
    }


@app.get("/api/agent/feedback/{investigation_id}")
def api_get_investigation_feedback(
    investigation_id: str = FastAPIPath(..., min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_\-]+$")
) -> List[Dict[str, Any]]:
    """Retrieve review feedback for a specific investigation."""
    from src.agent.feedback.store import get_feedback_store
    store = get_feedback_store()
    items = store.get_feedback_for_investigation(investigation_id)
    return [i.model_dump() for i in items]


# ============================================================================
# Phase 5 Research Monitoring & Epistemic Governance Endpoints
# ============================================================================

@app.get("/api/monitoring/drift")
def api_get_monitoring_drift() -> Dict[str, Any]:
    """Retrieve real computed temporal concept and topology drift report."""
    report_file = Path("results/drift/temporal_drift_report.json")
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    from src.ml.drift.drift_report import generate_and_save_drift_report
    return generate_and_save_drift_report()


@app.get("/api/monitoring/evaluation")
def api_get_monitoring_evaluation() -> Dict[str, Any]:
    """Retrieve Phase 5 agent ablation results and semantic citation benchmarks."""
    eval_file = Path("results/benchmarks/phase5_agent_ablation_results.json")
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            return json.load(f)
    from src.agent.evaluation.runner import run_phase5_ablation_benchmark
    return run_phase5_ablation_benchmark()


@app.get("/api/monitoring/candidates")
def api_get_monitoring_candidates() -> Dict[str, Any]:
    """List isolated continual learning candidate models and promotion gate status."""
    cand_dir = Path("artifacts/phase5_candidates")
    candidates = []
    if cand_dir.exists():
        for manifest_path in cand_dir.glob("*/candidate_manifest.json"):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    candidates.append(json.load(f))
            except Exception:
                pass

    return {
        "candidate_root": str(cand_dir),
        "total_candidates": len(candidates),
        "candidates": candidates,
        "promotion_gate_rules": {
            "primary_metric": "AUPRC",
            "minimum_improvement": 0.015,
            "max_fpr_increase": 0.020,
            "human_approval_required": True,
            "original_checkpoints_status": "IMMUTABLE_READ_ONLY",
        },
    }


@app.get("/api/monitoring/feedback-stats")
def api_get_monitoring_feedback_stats() -> Dict[str, Any]:
    """Retrieve analyst feedback governance metrics separating opinion from ground truth."""
    from src.agent.feedback.evaluator import AnalystFeedbackEvaluator
    evaluator = AnalystFeedbackEvaluator()
    report = evaluator.analyze_feedback()
    from dataclasses import asdict
    return asdict(report)


# Mount frontend SPA if built
if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        # Explicitly prevent path traversal attacks
        if ".." in full_path or "%2e%2e" in full_path.lower():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

        file_path = (DIST_DIR / full_path).resolve()
        try:
            file_path.relative_to(DIST_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
