from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class EvidenceMode(str, Enum):
    HISTORICAL_EVALUATION = "HISTORICAL_EVALUATION"
    RETROSPECTIVE_INVESTIGATION = "RETROSPECTIVE_INVESTIGATION"

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED_HYPOTHESIS = "UNVERIFIED_HYPOTHESIS"
    PROPRIETARY_UNKNOWN = "PROPRIETARY_UNKNOWN"
    CONFLICTING = "CONFLICTING"

class LabelStatus(str, Enum):
    ILLICIT = "illicit"
    LICIT = "licit"
    UNKNOWN = "unknown"
    RESTRICTED = "restricted_by_policy"

class SourceAuthorityTier(int, Enum):
    TIER_1_REPRODUCED_METRICS = 1
    TIER_2_CHECKPOINT_METADATA = 2
    TIER_3_REVIEWED_METHODOLOGY = 3
    TIER_4_QUALIFIED_AUDIT = 4
    TIER_5_HISTORICAL_REPORT = 5

class EvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., description="Unique deterministic evidence ID, e.g. EV-TX-23042587")
    source_type: str = Field(..., description="Type of evidence: transaction, graph, prediction, explanation, document, metric")
    source_artifact: str = Field(..., description="File path, checkpoint, or dataset origin")
    source_checksum: Optional[str] = Field(None, description="SHA-256 or version hash of the source artifact")
    content_hash: Optional[str] = Field(None, description="Deterministic SHA-256 hash of the evidence content")
    tx_id: Optional[int] = Field(None, description="Bitcoin transaction ID if applicable")
    snapshot_id: Optional[int] = Field(None, description="Snapshot number (1-49)")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    available_as_of_step: Optional[int] = Field(None, description="Earliest snapshot step at which evidence was available")
    dataset_split: Optional[str] = Field(None, description="train, validation, or test")
    model_version: Optional[str] = Field(None, description="Model identifier or checkpoint name if applicable")
    verification_status: VerificationStatus = Field(default=VerificationStatus.VERIFIED)
    confidence_level: str = Field(default="HIGH", description="HIGH, MEDIUM, LOW, or UNRATED")
    limitations: List[str] = Field(default_factory=list, description="Explicit disclosures and boundaries")
    citation: str = Field(..., description="Academic or codebase citation reference")

class TransactionEvidence(EvidenceRecord):
    node_index: int = Field(..., description="0-indexed position in dataset")
    label_status: LabelStatus = Field(..., description="illicit (1), licit (0), unknown (-1), or restricted")
    raw_label_code: Optional[int] = Field(None, description="Raw label code if permitted by policy")
    in_degree: int = Field(..., description="Number of incoming transaction flows")
    out_degree: int = Field(..., description="Number of outgoing transaction flows")
    is_isolated_incoming: bool = Field(..., description="True if in-degree is 0")
    is_completely_isolated: bool = Field(..., description="True if both in-degree and out-degree are 0")
    feature_dimension: int = Field(165, description="165 total features (93 local, 72 aggregated)")
    local_features_count: int = Field(93, description="93 local point-in-time features")
    aggregated_features_count: int = Field(72, description="72 forward/backward neighbor aggregations")
    safe_feature_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Safe numerical feature statistics without invented semantic names"
    )

class GraphEdge(BaseModel):
    source_tx_id: int
    target_tx_id: int
    source_snapshot: int
    target_snapshot: int
    is_intra_snapshot: bool = True

class GraphEvidence(EvidenceRecord):
    hops: int = Field(..., description="Traversal depth (1 or 2)")
    max_nodes_limit: int = Field(..., description="Resource cap enforced during extraction")
    truncated: bool = Field(False, description="True if graph exceeded max_nodes_limit")
    total_subgraph_nodes: int = Field(...)
    total_subgraph_edges: int = Field(...)
    subgraph_nodes: List[int] = Field(..., description="Transaction IDs in extracted neighborhood")
    directed_edges: List[GraphEdge] = Field(..., description="Directed payment flows")
    illicit_neighbors_count: int = Field(0, description="Confirmed illicit neighbors visible under policy")
    licit_neighbors_count: int = Field(0, description="Confirmed licit neighbors visible under policy")
    unknown_neighbors_count: int = Field(0, description="Unlabelled background transactions in neighborhood")

class PredictionEvidence(EvidenceRecord):
    model_name: str = Field(..., description="Model identifier: etgnn, random_forest, xgboost, etc.")
    illicit_probability: float = Field(..., ge=0.0, le=1.0)
    decision_threshold: float = Field(0.50, description="Operational decision threshold")
    predicted_class: int = Field(..., description="1 for illicit, 0 for licit")
    is_calibrated_confidence: bool = Field(False, description="False: raw sigmoid/tree output, not Platt calibrated")
    inference_latency_ms: float = Field(...)
    feature_family_used: str = Field("all_165_features", description="all_165_features, local_93_only, or local_no_edges")
    training_cutoff_step: int = Field(32, description="Trained on snapshots <= 32")

class ExplanationEvidence(EvidenceRecord):
    explanation_method: str = Field("GNNExplainer", description="Attribution algorithm")
    target_model: str = Field("etgnn")
    top_feature_indices: List[int] = Field(default_factory=list, description="0-indexed influential feature numbers")
    top_feature_importances: List[float] = Field(default_factory=list)
    top_directed_edges: List[Dict[str, Any]] = Field(default_factory=list)
    has_meaningful_edge_attribution: bool = Field(...)
    explanation_fidelity_drop: float = Field(0.0, description="Signed probability drop upon edge removal")
    fidelity_disclaimer: str = Field(
        "GNNExplainer attributions identify sensitivity in model representations. "
        "Controlled experiments show top-edge deletion does not statistically outperform random edge deletion "
        "(p=0.6827), and indicates model false-alarm sensitivity rather than deterministic proof of fraud."
    )

class ModelMetadataEvidence(EvidenceRecord):
    model_name: str
    architecture_family: str
    input_dimension: int
    hidden_dimension: Optional[int] = None
    time_dimension: Optional[int] = None
    total_parameters: int
    training_snapshots: str = "Steps 1–32"
    loss_function: str
    reproducibility_status: str = "REPRODUCED"
    known_limitations: List[str] = Field(default_factory=list)

class ModelMetricEvidence(EvidenceRecord):
    model_name: str
    metric_name: str
    metric_value: float
    dataset_name: str = "Elliptic Bitcoin"
    evaluated_snapshots: str = "Steps 41–49 (Test Split)"
    sample_count: int = 9973
    decision_threshold: Optional[float] = 0.50
    reproduction_status: str = "INDEPENDENTLY_REPRODUCED"
    has_historical_conflict: bool = False
    conflict_notes: Optional[str] = None

class ResearchDocumentEvidence(EvidenceRecord):
    document_title: str
    section_heading: str
    source_path: str
    authority_tier: SourceAuthorityTier
    text_content: str
    extracted_claims: List[str] = Field(default_factory=list)
    has_unverified_hypothesis: bool = False

class RetrievalResult(BaseModel):
    query: str
    evidence_mode: EvidenceMode
    as_of_snapshot: Optional[int] = None
    structured_evidence: List[EvidenceRecord] = Field(default_factory=list)
    research_evidence: List[ResearchDocumentEvidence] = Field(default_factory=list)
    total_evidence_count: int
    retrieval_latency_ms: float
    max_relevance_score: float = 0.0
    reranker_applied: bool = True
    abstention_triggered: bool = False
    has_temporal_filter_applied: bool = False
    filtered_future_items_count: int = 0
    warnings: List[str] = Field(default_factory=list)

class ModelDisagreement(BaseModel):
    is_divergent: bool = False
    model_probabilities: Dict[str, float] = Field(default_factory=dict)
    probability_spread: float = 0.0
    divergence_reason: Optional[str] = None

class InvestigationContext(BaseModel):
    tx_id: int
    snapshot_id: int
    evidence_mode: EvidenceMode
    as_of_snapshot: Optional[int] = None
    transaction_evidence: TransactionEvidence
    graph_evidence: Optional[GraphEvidence] = None
    predictions: Dict[str, PredictionEvidence] = Field(default_factory=dict)
    disagreement: ModelDisagreement = Field(default_factory=ModelDisagreement)
    explanation: Optional[ExplanationEvidence] = None
    research_context: List[ResearchDocumentEvidence] = Field(default_factory=list)
    reproduced_metrics: List[ModelMetricEvidence] = Field(default_factory=list)
    provenance_chain: List[str] = Field(default_factory=list, description="Ordered list of verified evidence IDs")
    investigation_warnings: List[str] = Field(default_factory=list)
    context_generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

# Typed Exceptions
class AgentToolError(Exception):
    """Base exception for deterministic investigation tools."""
    def __init__(self, message: str, error_code: str = "TOOL_ERROR"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

class TransactionNotFoundError(AgentToolError):
    def __init__(self, tx_id: int):
        super().__init__(f"Transaction ID {tx_id} does not exist in the Elliptic dataset.", "TRANSACTION_NOT_FOUND")

class TemporalBoundaryViolationError(AgentToolError):
    def __init__(self, message: str):
        super().__init__(message, "TEMPORAL_LEAKAGE_BLOCKED")

class ModelNotSupportedError(AgentToolError):
    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' is not in the supported verified checkpoint registry.", "MODEL_NOT_SUPPORTED")

class ResourceLimitExceededError(AgentToolError):
    def __init__(self, message: str):
        super().__init__(message, "RESOURCE_LIMIT_EXCEEDED")

