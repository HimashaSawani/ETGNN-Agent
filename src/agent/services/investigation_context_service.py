from __future__ import annotations

from typing import List, Optional
from src.agent.rag.retriever import HybridRetriever
from src.agent.rag.evidence_validator import EvidenceValidator
from src.agent.schemas import (
    EvidenceMode,
    InvestigationContext,
    ModelDisagreement,
    PredictionEvidence,
)
from src.agent.tools.investigation_tools import (
    InvestigationToolRegistry,
    get_transaction,
    get_transaction_neighbors,
    get_prediction,
    get_explanation,
    get_evaluation_metrics,
)

class InvestigationContextService:
    """
    Core Phase 2 service layer.
    Assembles a deterministic, validated, point-in-time compliant investigation context
    combining structured graph data, multi-model predictions, explanations, and RAG knowledge.
    """
    def __init__(
        self,
        tool_registry: Optional[InvestigationToolRegistry] = None,
        retriever: Optional[HybridRetriever] = None,
        validator: Optional[EvidenceValidator] = None,
    ):
        self.tools = tool_registry or InvestigationToolRegistry.get_instance()
        self.retriever = retriever or HybridRetriever()
        self.validator = validator or EvidenceValidator()

    def build_investigation_context(
        self,
        tx_id: int,
        question: str = "Forensic fraud risk and graph context investigation",
        evidence_mode: str = "HISTORICAL_EVALUATION",
        as_of_snapshot: Optional[int] = None,
        top_k_research: int = 4,
    ) -> InvestigationContext:
        mode = EvidenceMode(evidence_mode)
        warnings: List[str] = []

        # 1. Parameter Validation
        val_tx = self.validator.validate_transaction_id(tx_id)
        if not val_tx.is_valid:
            for issue in val_tx.issues:
                warnings.append(f"Validation issue on {issue.field}: {issue.message}")

        if as_of_snapshot is not None:
            val_step = self.validator.validate_snapshot_step(as_of_snapshot)
            if not val_step.is_valid:
                for issue in val_step.issues:
                    warnings.append(f"Validation issue on {issue.field}: {issue.message}")

        # Scan question for potential prompt injection
        val_q = self.validator.validate_document_content_security(question)
        if not val_q.is_valid:
            warnings.append("Adversarial prompt injection pattern detected and sanitized in query question.")
            clean_question = val_q.sanitized_text or question
        else:
            clean_question = question

        # 2. Retrieve Transaction Evidence
        tx_evidence = self.tools.get_transaction(
            tx_id=tx_id,
            evidence_mode=mode,
            as_of_snapshot=as_of_snapshot,
        )
        tx_snapshot = tx_evidence.snapshot_id or 1

        # 3. Retrieve Graph Evidence (1-hop)
        graph_evidence = self.tools.get_transaction_neighbors(
            tx_id=tx_id,
            hops=1,
            max_nodes=50,
            evidence_mode=mode,
            as_of_snapshot=as_of_snapshot,
        )

        # 4. Multi-Model Predictions (ETGNN, Random Forest, XGBoost)
        predictions = {}
        for m_name in ("etgnn", "random_forest", "xgboost"):
            try:
                pred = self.tools.get_prediction(
                    tx_id=tx_id,
                    model_name=m_name,
                    evidence_mode=mode,
                    as_of_snapshot=as_of_snapshot,
                )
                predictions[m_name] = pred
            except Exception as e:
                warnings.append(f"Model prediction failed for {m_name}: {e}")

        # 5. Compute Model Disagreement
        rf_prob = predictions.get("random_forest", PredictionEvidence(
            evidence_id="", source_type="", source_artifact="", tx_id=tx_id, model_name="",
            illicit_probability=0.0, predicted_class=0, inference_latency_ms=0.0, citation=""
        )).illicit_probability

        et_prob = predictions.get("etgnn", PredictionEvidence(
            evidence_id="", source_type="", source_artifact="", tx_id=tx_id, model_name="",
            illicit_probability=0.0, predicted_class=0, inference_latency_ms=0.0, citation=""
        )).illicit_probability

        spread = abs(rf_prob - et_prob)
        is_divergent = spread >= 0.30

        disagreement = ModelDisagreement(
            is_divergent=is_divergent,
            model_probabilities={
                "random_forest": round(rf_prob, 4),
                "etgnn": round(et_prob, 4),
                "xgboost": round(predictions.get("xgboost", PredictionEvidence(
                    evidence_id="", source_type="", source_artifact="", tx_id=tx_id, model_name="",
                    illicit_probability=0.0, predicted_class=0, inference_latency_ms=0.0, citation=""
                )).illicit_probability, 4),
            },
            probability_spread=round(spread, 4),
            divergence_reason=(
                f"Model divergence detected (spread={spread:.3f}). "
                "In heterophilic financial transaction networks (>95% licit), GNN neighborhood aggregation can smear "
                "isolated fraud indicators across edges, while tabular Random Forest preserves local feature risk."
                if is_divergent else None
            ),
        )

        # 6. Retrieve Explanation Evidence
        try:
            explanation = self.tools.get_explanation(
                tx_id=tx_id,
                top_k_features=10,
                evidence_mode=mode,
            )
        except Exception as e:
            explanation = None
            warnings.append(f"Explanation extraction failed: {e}")

        # 7. Hybrid Research Knowledge Retrieval
        rag_res = self.retriever.retrieve(
            query=clean_question,
            top_k=top_k_research,
            evidence_mode=mode,
            as_of_snapshot=as_of_snapshot or tx_snapshot,
        )
        research_chunks = rag_res.research_evidence

        # 8. Retrieve Authoritative Benchmark Metrics
        reproduced_metrics = self.tools.get_evaluation_metrics()

        # 9. Build Ordered Provenance Chain
        provenance = [tx_evidence.evidence_id, graph_evidence.evidence_id]
        for p in predictions.values():
            provenance.append(p.evidence_id)
        if explanation:
            provenance.append(explanation.evidence_id)
        for r in research_chunks:
            provenance.append(r.evidence_id)
        for m in reproduced_metrics:
            provenance.append(m.evidence_id)

        return InvestigationContext(
            tx_id=tx_id,
            snapshot_id=tx_snapshot,
            evidence_mode=mode,
            as_of_snapshot=as_of_snapshot,
            transaction_evidence=tx_evidence,
            graph_evidence=graph_evidence,
            predictions=predictions,
            disagreement=disagreement,
            explanation=explanation,
            research_context=research_chunks,
            reproduced_metrics=reproduced_metrics,
            provenance_chain=provenance,
            investigation_warnings=warnings,
        )

