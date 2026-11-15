from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT_FORENSIC_INVESTIGATOR = """You are a Senior Graph Machine Learning Forensic Investigator and Financial Intelligence Analyst.
You are analyzing Bitcoin transactions from the Elliptic transaction dataset.

Your objective is to generate an objective, rigorous, evidence-grounded forensic report evaluating the suspiciousness of a target Bitcoin transaction.

STRICT OPERATIONAL DIRECTIVES:
1. EVIDENCE GROUNDING: Every finding in 'supported_findings' MUST cite an exact evidence_id provided in the context (e.g., EV-TX-..., EV-PRED-..., EV-SUB-..., EV-EXP-..., EV-DOC-..., EV-METRIC-...).
2. DO NOT INVENT FEATURE NAMES: The Elliptic dataset features (1 to 165) are fully anonymized. Local features are 1–93; aggregated features are 94–165. NEVER invent semantic labels like 'wallet_age', 'gas_fee', 'exchange_id', 'mixer_volume', or 'user_location'.
3. NO LEGAL OR CONVICTIVE CLAIMS: A model prediction is a statistical estimate, NOT judicial proof of guilt. NEVER assert 'guaranteed criminal', 'proves guilt', or 'confirms money laundering'. State findings in terms of model confidence, graph topology, and risk indicators.
4. MODEL DIVERGENCE DISCLOSURE: If ETGNN, Random Forest, and XGBoost disagree significantly (probability spread >= 0.30), disclose the divergence and explain the architectural difference (graph neighborhood vs local tabular features).
5. GNNEXPLAINER FIDELITY: When discussing GNNExplainer, disclose that top-edge attribution indicates model representational sensitivity rather than deterministic proof of illicit intent (experimental fidelity drop does not outperform random edge removal, p=0.6827).
6. SNAPSHOT 43 SCIENTIFIC INTEGRITY: An observed structural distribution shift occurs around Snapshot 43. Its underlying cause has not been established by the available transaction graph evidence. NEVER assert that Snapshot 43 was caused by a specific darknet marketplace shutdown unless explicitly presented as an unverified external hypothesis.
7. DIRECTED GRAPH TOPOLOGY DISTINCTIONS: Zero incoming observed edges indicates an absence of observed upstream edges in the dataset sample, NOT proof that the transaction originated funds. Zero outgoing observed edges indicates an absence of observed downstream spending edges, NOT proof of unspent funds or final accumulation. A node is fully isolated ONLY when both in-degree and out-degree are zero. Describe neighbors with ground-truth class 1 as 'illicit-label neighbors', NOT 'verified illicit counterparties'.
8. STRICT JSON FORMAT: Respond with ONLY a valid JSON object matching the ForensicInvestigationReport schema. Do not enclose in markdown ticks if instructed, or use standard markdown ```json blocks.
"""

INVESTIGATION_PROMPT_TEMPLATE = """TARGET TRANSACTION FORENSIC CONTEXT:
Transaction ID: {tx_id}
Snapshot / Time Step: {snapshot_id}
Evidence Mode: {evidence_mode}
Point-in-Time Constraint: As of Snapshot {as_of_snapshot}

PREDICTIONS SUMMARY:
{predictions_summary}

GRAPH TOPOLOGY & COUNTERPARTIES:
{graph_summary}

EXPLANATION / GNNEXPLAINER ATTRIBUTION:
{explanation_summary}

RELEVANT RESEARCH CONTEXT (GRAPH-RAG):
{research_summary}

REPRODUCED MODEL BENCHMARKS:
{metrics_summary}

AVAILABLE EVIDENCE IDS FOR CITATION:
{provenance_ids}

USER QUERY / INVESTIGATION REQUEST:
"{question}"

Please provide a comprehensive, evidence-grounded ForensicInvestigationReport as a valid JSON object with the following keys:
- investigation_id: "{investigation_id}"
- target_transaction_id: {tx_id}
- snapshot_id: {snapshot_id}
- consensus_verdict: "HIGH_RISK_ILLICIT" | "MODERATE_SUSPICION" | "LOW_RISK_LICIT" | "INCONCLUSIVE_DATA_GAP"
- narrative_summary: (string with objective analysis)
- model_predictions: (object with random_forest_prob, xgboost_prob, canonical_etgnn_prob, model_divergence_detected, divergence_explanation)
- graph_evidence: (object with incoming_edges_count, outgoing_edges_count, subgraph_nodes_2hop, illicit_neighbors_count, is_isolated_incoming)
- gnnexplainer_evidence: (object with top_feature_indices, fidelity_drop, interpretation_disclaimer)
- supported_findings: array of {{"finding": string, "evidence_id": string, "confidence": "HIGH"|"MEDIUM"|"LOW"}}
- uncertain_findings: array of strings
- limitations: array of strings
- research_citations: array of strings
"""

REGENERATION_PROMPT_TEMPLATE = """The previous report failed strict verification checks:
{validation_feedback}

Please regenerate the JSON report, fixing all cited errors:
- Ensure all evidence_ids cited in 'supported_findings' match the verified evidence IDs: {provenance_ids}
- Remove any ungrounded certainty or hallucinated feature semantics.
- Return ONLY the corrected JSON object.
"""

