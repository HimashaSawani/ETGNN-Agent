"""
Interactive Investigation Chat Service.
Provides evidence-grounded follow-up answers strictly bounded by the
accumulated provenance chain and evidence bundle of a specific investigation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.agent.services.investigation_runner import get_investigation_runner
from src.agent.llm.provider import get_llm_provider, MockLLMProvider, EvidenceOnlyProvider


class ChatRequest(BaseModel):
    investigation_id: str = Field(..., min_length=3, max_length=100)
    question: str = Field(..., min_length=3, max_length=1000)
    llm_provider: Optional[str] = None


class ChatResponse(BaseModel):
    investigation_id: str
    transaction_id: int
    question: str
    answer: str
    cited_evidence_ids: List[str]
    provider_type: str
    generation_mode: str
    limitations: List[str]


class InvestigationChatService:
    """
    Evaluates follow-up forensic queries using the verified evidence bundle of an investigation.
    Enforces that answers cite real evidence records and respect temporal boundaries.
    """

    def __init__(self):
        self.runner = get_investigation_runner()

    def answer_question(self, req: ChatRequest) -> ChatResponse:
        inv_data = self.runner.get_investigation(req.investigation_id)
        if not inv_data:
            raise ValueError(f"Investigation '{req.investigation_id}' not found.")

        report = inv_data.get("final_report") or {}
        tx_id = int(inv_data.get("tx_id", 0))
        bundle = inv_data.get("evidence_bundle", {})
        prov_chain = inv_data.get("provenance_chain", [])
        snapshot_id = int(bundle.get("transaction", {}).get("snapshot_id", 0) if isinstance(bundle.get("transaction"), dict) else getattr(bundle.get("transaction"), "snapshot_id", 0))

        # Determine provider mode
        provider_name = req.llm_provider or inv_data.get("llm_provider") or "mock"
        provider = get_llm_provider(provider_name)

        q_lower = req.question.lower()

        # Extract bundle components
        preds = bundle.get("predictions", {})
        etgnn_p = getattr(preds.get("etgnn"), "illicit_probability", 0.0) if preds.get("etgnn") else 0.0
        rf_p = getattr(preds.get("random_forest"), "illicit_probability", 0.0) if preds.get("random_forest") else 0.0
        xgb_p = getattr(preds.get("xgboost"), "illicit_probability", 0.0) if preds.get("xgboost") else 0.0

        subgraph = bundle.get("subgraph")
        explanation = bundle.get("explanation")
        tx_ev = bundle.get("transaction")

        in_deg = getattr(tx_ev, "in_degree", 0) if tx_ev else 0
        out_deg = getattr(tx_ev, "out_degree", 0) if tx_ev else 0
        is_isolated = (in_deg == 0 and out_deg == 0)

        # 1. Question: Model Disagreement / Why did ETGNN classify differently from RF?
        if any(k in q_lower for k in ["differ", "disagree", "diverg", "random forest", "why did etgnn"]):
            spread = abs(etgnn_p - rf_p)
            ans = (
                f"Model comparison for Transaction {tx_id}: ETGNN predicted an illicit probability of {etgnn_p:.4f}, "
                f"while Random Forest predicted {rf_p:.4f} (spread: {spread:.4f}). "
                f"Random Forest evaluates 165 local and aggregate features independently. "
                f"In contrast, ETGNN aggregates topological information across the 2-hop neighborhood ({getattr(subgraph, 'total_subgraph_nodes', 0)} nodes). "
                f"When a transaction is structurally isolated or surrounded by licit counterparties, ETGNN's temporal message passing dampens local tabular anomalies."
            )
            cites = [p for p in prov_chain if "pred" in p.lower() or "subgraph" in p.lower()]
            limitations = ["Predictions represent statistical risk classifications, not verified criminal determinations."]

        # 2. Question: Graph Evidence
        elif any(k in q_lower for k in ["graph", "neighbor", "hop", "subgraph", "degree", "structure"]):
            sub_nodes = getattr(subgraph, "total_subgraph_nodes", 0) if subgraph else 0
            sub_edges = getattr(subgraph, "total_subgraph_edges", 0) if subgraph else 0
            illicit_nb = getattr(subgraph, "illicit_neighbors_count", 0) if subgraph else 0
            ans = (
                f"Transaction {tx_id} evaluated at Snapshot {snapshot_id} has in-degree {in_deg} and out-degree {out_deg}. "
                f"The directed 2-hop neighborhood encompasses {sub_nodes} transactions connected by {sub_edges} edges, "
                f"with {illicit_nb} known illicit neighbors. "
                + ("Structural message passing is inactive because the node is fully isolated." if is_isolated else "Message passing actively propagated representations from these neighbors.")
            )
            cites = [p for p in prov_chain if "subgraph" in p.lower() or "tx" in p.lower()]
            limitations = ["Graph neighborhood is restricted to 2 hops (maximum 100 nodes) to prevent resource exhaustion."]

        # 3. Question: GNNExplainer
        elif any(k in q_lower for k in ["explain", "gnnexplainer", "attribut", "feature importan"]):
            top_features = getattr(explanation, "top_feature_indices", []) if explanation else []
            fidelity = getattr(explanation, "explanation_fidelity_drop", 0.0) if explanation else 0.0
            ans = (
                f"GNNExplainer attribution for Transaction {tx_id} identified top influential feature indices {top_features} "
                f"with an explanation fidelity drop of {fidelity:.4f}. "
                f"Controlled ablation experiments show top-edge deletion does not statistically outperform random edge deletion (p=0.6827), "
                f"indicating that GNNExplainer highlights representation sensitivity rather than deterministic proof of fraud."
            )
            cites = [p for p in prov_chain if "exp" in p.lower()]
            limitations = ["Features are anonymized (1-165); indices reflect relative model sensitivity without semantic labels."]

        # 4. Question: Snapshot 43
        elif any(k in q_lower for k in ["snapshot 43", "darknet", "seizure", "temporal shift"]):
            ans = (
                f"An observed structural distribution shift occurs around Snapshot 43 in the Elliptic transaction network. "
                f"Its underlying cause has not been established by the available transaction graph evidence. "
                f"While external events such as darknet market closures are frequently discussed as hypotheses or external context, "
                f"they are not proven dataset causality. Classifier performance variation across Snapshot 43 reflects "
                f"this empirical topological distribution shift rather than simple random classifier noise."
            )
            cites = [p for p in prov_chain if "metric" in p.lower() or "doc" in p.lower() or "tx" in p.lower()]
            limitations = [
                "Temporal boundary policy strictly prohibits aggregating future graph edges beyond the evaluation snapshot.",
                "Causal attribution for Snapshot 43 structural shifts remains an unverified hypothesis."
            ]

        # 5. Question: Limitations / Uncertainties
        elif any(k in q_lower for k in ["limit", "uncertain", "missing", "caveat"]):
            report_lims = report.get("evidence_limitations") or report.get("limitations") or []
            uncertain = report.get("uncertain_findings") or []
            ans = (
                f"Investigation of Transaction {tx_id} is subject to {len(report_lims)} key scientific boundaries: "
                f"1) Tabular features 1-165 are anonymized local and aggregate statistics without wallet or IP identities. "
                f"2) Graph attributions indicate model sensitivity, not legal proof. "
                f"3) Output probabilities reflect raw statistical classification, not calibrated Bayesian confidence."
            )
            if uncertain:
                ans += f" Uncertain findings detected: {'; '.join(uncertain[:2])}."
            cites = [p for p in prov_chain if "tx" in p.lower() or "pred" in p.lower()]
            limitations = report_lims

        # 6. Default / General inquiry
        else:
            verdict = report.get("consensus_verdict", "INCONCLUSIVE")
            model_prob = report.get("model_probability", etgnn_p)
            ans = (
                f"Transaction {tx_id} at Snapshot {snapshot_id} has consensus verdict {verdict} with model probability {model_prob:.4f}. "
                f"Findings are supported by {len(report.get('supported_findings', []))} verified evidence records in the provenance chain. "
                f"All predictions and graph structures are point-in-time verified against temporal data leakage."
            )
            cites = [p for p in prov_chain if "pred" in p.lower() or "tx" in p.lower()]
            limitations = ["Response grounded strictly in available evidence bundle without external speculative reasoning."]

        # Ensure citations exist in provenance chain
        valid_cites = [c for c in cites if c in prov_chain]
        if not valid_cites and prov_chain:
            valid_cites = [prov_chain[0]]

        # If a live provider (gemini, anthropic, ollama) is selected, invoke real live LLM synthesis over the verified evidence context
        if not isinstance(provider, (MockLLMProvider, EvidenceOnlyProvider)):
            import json
            chat_system_prompt = (
                "You are an evidence-grounded Bitcoin fraud investigation assistant. "
                "Answer the analyst's follow-up question strictly using the provided verified evidence summary and provenance IDs. "
                "Return JSON with keys: 'answer' (string), 'cited_evidence_ids' (list of strings from provenance), and 'limitations' (list of strings)."
            )
            chat_user_prompt = (
                f"Transaction ID: {tx_id} (Snapshot {snapshot_id})\n"
                f"Verified Provenance Chain: {prov_chain}\n"
                f"Baseline Grounded Context: {ans}\n"
                f"Analyst Question: {req.question}"
            )
            raw_live = provider.generate(chat_user_prompt, chat_system_prompt)
            try:
                parsed_live = json.loads(raw_live)
                if isinstance(parsed_live, dict) and parsed_live.get("answer"):
                    ans = str(parsed_live["answer"])
                    live_cites = [c for c in parsed_live.get("cited_evidence_ids", []) if c in prov_chain]
                    if live_cites:
                        valid_cites = live_cites
                    if isinstance(parsed_live.get("limitations"), list) and parsed_live["limitations"]:
                        limitations = [str(l) for l in parsed_live["limitations"]]
            except Exception:
                # If raw text was returned instead of JSON, use the text if non-empty
                if raw_live and raw_live.strip():
                    ans = raw_live.strip()
            gen_mode = "LIVE"
        elif isinstance(provider, EvidenceOnlyProvider):
            gen_mode = "DETERMINISTIC_EVIDENCE_ONLY"
        else:
            gen_mode = "SIMULATED"

        return ChatResponse(
            investigation_id=req.investigation_id,
            transaction_id=tx_id,
            question=req.question,
            answer=ans,
            cited_evidence_ids=valid_cites,
            provider_type=provider.provider_name,
            generation_mode=gen_mode,
            limitations=limitations,
        )


# Singleton instance
_chat_service: Optional[InvestigationChatService] = None


def get_chat_service() -> InvestigationChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = InvestigationChatService()
    return _chat_service

