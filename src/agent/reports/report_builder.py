from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ConsensusVerdict(str, Enum):
    HIGH_RISK_ILLICIT = "HIGH_RISK_ILLICIT"
    MODERATE_SUSPICION = "MODERATE_SUSPICION"
    LOW_RISK_LICIT = "LOW_RISK_LICIT"
    INCONCLUSIVE_DATA_GAP = "INCONCLUSIVE_DATA_GAP"
    ELEVATED_RISK_INDICATION = "ELEVATED_RISK_INDICATION"
    BASELINE_RISK_INDICATION = "BASELINE_RISK_INDICATION"
    MODEL_DIVERGENCE_DETECTED = "MODEL_DIVERGENCE_DETECTED"


class ModelPredictionsSummary(BaseModel):
    random_forest_prob: float = Field(0.0, ge=0.0, le=1.0)
    xgboost_prob: float = Field(0.0, ge=0.0, le=1.0)
    canonical_etgnn_prob: float = Field(0.0, ge=0.0, le=1.0)
    matched_gat_prob: float = Field(0.0, ge=0.0, le=1.0)
    model_divergence_detected: bool = False
    divergence_explanation: Optional[str] = None


class HypothesisEvaluation(BaseModel):
    hypothesis_id: str
    hypothesis: str
    classification: str = Field(
        default="SUPPORTED",
        description="SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or NOT_TESTABLE",
    )
    evidence_ids: List[str] = Field(default_factory=list)
    rationale: str = ""


class GraphEvidenceSummary(BaseModel):
    incoming_edges_count: int = 0
    outgoing_edges_count: int = 0
    subgraph_nodes_2hop: int = 0
    illicit_neighbors_count: int = 0
    is_isolated_incoming: bool = False
    is_fully_isolated: bool = False


class GNNExplainerEvidenceSummary(BaseModel):
    top_feature_indices: List[int] = Field(default_factory=list)
    fidelity_drop: float = 0.0
    interpretation_disclaimer: str = (
        "GNNExplainer attributions identify sensitivity in model representations. "
        "Controlled experiments show top-edge deletion does not statistically outperform random edge deletion "
        "(p=0.6827), and indicates model false-alarm sensitivity rather than deterministic proof of fraud."
    )


class SupportedFinding(BaseModel):
    finding: str
    evidence_id: str
    confidence: str = Field(default="HIGH", description="HIGH, MEDIUM, or LOW")


class ModelVersionsMetadata(BaseModel):
    etgnn_hash: str = "checkpoint_etgnn_reproduced_2026"
    rf_hash: str = "rf_baseline_split32_2026"
    agent_version: str = "ETGNN-Agent-v3.5.0"


class ForensicInvestigationReport(BaseModel):
    """
    Forensic investigation report produced by ETGNN-Agent.
    Conforms strictly to docs/phase1_5/AGENT_TRANSPARENCY_SPECIFICATION.md and Phase 3.5 guidelines.
    """
    model_config = ConfigDict(extra="ignore")

    investigation_id: str
    target_transaction_id: int
    snapshot_id: int
    status: str = Field(default="COMPLETED", description="COMPLETED, ABSTAINED, or FAILED")
    investigation_status: str = Field(default="COMPLETED")
    consensus_verdict: ConsensusVerdict = ConsensusVerdict.INCONCLUSIVE_DATA_GAP

    # Neutral calibrated model fields
    predicted_class: Optional[int] = None
    model_probability: Optional[float] = None
    model_disagreement: bool = False
    probability_spread: float = 0.0
    risk_indicator: str = "BASELINE_RISK"
    epistemic_notice: str = (
        "Model probability represents raw statistical classification output, NOT a calibrated Bayesian confidence. "
        "Predictions reflect distribution shifts and local graph topology, not judicial proof of criminal activity."
    )

    # Provider Transparency & Mode Fields (Phase 3.5 & Phase 5 requirement)
    provider_type: str = Field(default="mock", description="mock, gemini, anthropic, ollama, or evidence_only")
    provider_model_id: Optional[str] = Field(default="deterministic_mock", description="Specific model ID invoked")
    generation_mode: str = Field(default="SIMULATED", description="SIMULATED, LIVE, or DETERMINISTIC_EVIDENCE_ONLY")
    live_llm_verified: bool = Field(default=False, description="True ONLY when real provider API succeeded")
    token_usage: Optional[Dict[str, int]] = None
    llm_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    evidence_completeness: str = Field(default="COMPLETE", description="COMPLETE, PARTIAL_RESULT, or EVIDENCE_INSUFFICIENT")
    workflow_stages_completed: List[str] = Field(default_factory=list)

    narrative_summary: Optional[str] = None
    model_predictions: ModelPredictionsSummary = Field(default_factory=ModelPredictionsSummary)
    graph_evidence: GraphEvidenceSummary = Field(default_factory=GraphEvidenceSummary)
    gnnexplainer_evidence: Optional[GNNExplainerEvidenceSummary] = None
    supported_findings: List[SupportedFinding] = Field(default_factory=list)
    hypothesis_evaluations: List[HypothesisEvaluation] = Field(default_factory=list)
    uncertain_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    evidence_limitations: List[str] = Field(default_factory=list)
    model_versions: ModelVersionsMetadata = Field(default_factory=ModelVersionsMetadata)
    research_citations: List[str] = Field(default_factory=list)
    provenance_chain: List[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
