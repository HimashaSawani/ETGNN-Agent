from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import httpx


# ============================================================================
# Typed Error Hierarchy
# ============================================================================

class LLMProviderError(Exception):
    """Base exception for all LLM inference errors."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when an external LLM provider call exceeds deadline."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when an external provider returns HTTP 429 Too Many Requests."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when API credentials are missing, rejected, or expired."""
    pass


class LLMOutputFormatError(LLMProviderError):
    """Raised when provider returns output that cannot be parsed into target schema."""
    pass


# ============================================================================
# Abstract Base Provider
# ============================================================================

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM inference providers.
    Declares provider identity, model specification, timeout, and execution modes.
    """
    provider_name: str = "base"
    model_id: str = "unknown"
    timeout_sec: float = 30.0
    max_retries: int = 2
    last_token_usage: Optional[Dict[str, int]] = None
    last_estimated_cost_usd: float = 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generates raw text response given prompt and system instructions."""
        pass


# ============================================================================
# Mode 1: Deterministic Mock Provider (Simulated / Testing Mode)
# ============================================================================

class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic rule-based mock provider for offline tests and development.
    Explicitly marked as simulated generation (generation_mode='SIMULATED').
    Strictly uses facts from context_data and cites existing provenance records.
    """
    provider_name: str = "mock"
    model_id: str = "deterministic_mock_v3"
    timeout_sec: float = 0.0
    max_retries: int = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        data = context_data or {}
        tx_id = data.get("tx_id", 0)
        snapshot_id = data.get("snapshot_id", 0)
        inv_id = data.get("investigation_id", "inv_mock_001")
        evidence_bundle = data.get("evidence_bundle", {})

        # Extract predictions
        preds = evidence_bundle.get("predictions", {})
        etgnn_pred = preds.get("etgnn")
        gat_pred = preds.get("matched_gat")
        rf_pred = preds.get("random_forest")
        xgb_pred = preds.get("xgboost")

        etgnn_p = (
            etgnn_pred.get("illicit_probability", 0.0)
            if isinstance(etgnn_pred, dict)
            else getattr(etgnn_pred, "illicit_probability", 0.0)
        )
        gat_p = (
            gat_pred.get("illicit_probability", 0.0)
            if isinstance(gat_pred, dict)
            else getattr(gat_pred, "illicit_probability", 0.0)
        )
        rf_p = (
            rf_pred.get("illicit_probability", 0.0)
            if isinstance(rf_pred, dict)
            else getattr(rf_pred, "illicit_probability", 0.0)
        )
        xgb_p = (
            xgb_pred.get("illicit_probability", 0.0)
            if isinstance(xgb_pred, dict)
            else getattr(xgb_pred, "illicit_probability", 0.0)
        )

        probs = [p for p in (etgnn_p, gat_p, rf_p, xgb_p) if p is not None and p > 0.0] or [etgnn_p, rf_p, xgb_p]
        mean_prob = sum(probs) / len(probs) if probs else 0.0
        prob_spread = (max(probs) - min(probs)) if probs else 0.0
        divergence = prob_spread >= 0.30

        # Calibrated neutral risk indicator
        if divergence:
            risk_indicator = "DIVERGENT_MODELS"
            verdict = "HIGH_RISK_ILLICIT" if mean_prob >= 0.50 else "LOW_RISK_LICIT"
        elif mean_prob >= 0.65:
            risk_indicator = "ELEVATED_RISK"
            verdict = "HIGH_RISK_ILLICIT"
        elif mean_prob >= 0.35:
            risk_indicator = "MODERATE_SUSPICION"
            verdict = "MODERATE_SUSPICION"
        elif mean_prob >= 0.0:
            risk_indicator = "BASELINE_RISK"
            verdict = "LOW_RISK_LICIT"
        else:
            risk_indicator = "DATA_GAP"
            verdict = "INCONCLUSIVE_DATA_GAP"

        # Graph evidence
        subgraph = evidence_bundle.get("subgraph")
        tx_ev = evidence_bundle.get("transaction")
        in_deg = (
            tx_ev.get("in_degree", 0)
            if isinstance(tx_ev, dict)
            else getattr(tx_ev, "in_degree", 0)
        )
        out_deg = (
            tx_ev.get("out_degree", 0)
            if isinstance(tx_ev, dict)
            else getattr(tx_ev, "out_degree", 0)
        )
        is_isolated_in = in_deg == 0
        is_fully_isolated = (in_deg == 0 and out_deg == 0)

        subgraph_nodes = 0
        illicit_nb = 0
        if subgraph:
            subgraph_nodes = (
                subgraph.get("total_subgraph_nodes", 0)
                if isinstance(subgraph, dict)
                else getattr(subgraph, "total_subgraph_nodes", 0)
            )
            illicit_nb = (
                subgraph.get("illicit_neighbors_count", 0)
                if isinstance(subgraph, dict)
                else getattr(subgraph, "illicit_neighbors_count", 0)
            )

        # GNNExplainer evidence
        explanation = evidence_bundle.get("explanation")
        top_features = []
        fidelity_drop = 0.0
        if explanation:
            top_features = (
                explanation.get("top_feature_indices", [])
                if isinstance(explanation, dict)
                else getattr(explanation, "top_feature_indices", [])
            )
            fidelity_drop = (
                explanation.get("explanation_fidelity_drop", 0.0)
                if isinstance(explanation, dict)
                else getattr(explanation, "explanation_fidelity_drop", 0.0)
            )

        # Provenance IDs to cite strictly from verified provenance
        provenance = evidence_bundle.get("provenance_chain", []) or data.get("provenance_chain", [])
        tx_eid = next((e for e in provenance if e.startswith("EV-TX-")), None)
        pred_eid = next((e for e in provenance if e.endswith("-etgnn") or "etgnn" in e.lower()), None)
        gat_eid = next((e for e in provenance if "matched_gat" in e.lower()), None)
        rf_eid = next((e for e in provenance if "random_forest" in e.lower() or "-rf" in e.lower()), None)
        sub_eid = next((e for e in provenance if e.startswith("EV-GRAPH-") or e.startswith("EV-SUB-")), None)
        exp_eid = next((e for e in provenance if e.startswith("EV-EXP-") or e.startswith("EV-EXPL-")), None)

        supported_findings = []
        if tx_eid:
            supported_findings.append({
                "finding": f"Transaction {tx_id} evaluated with in-degree of {in_deg} and out-degree of {out_deg}.",
                "evidence_id": tx_eid,
                "confidence": "HIGH"
            })
        if pred_eid:
            supported_findings.append({
                "finding": f"ETGNN model predicted illicit risk probability of {etgnn_p:.2f}.",
                "evidence_id": pred_eid,
                "confidence": "HIGH"
            })
        if gat_eid:
            supported_findings.append({
                "finding": f"Matched GAT baseline predicted illicit probability of {gat_p:.2f}.",
                "evidence_id": gat_eid,
                "confidence": "HIGH"
            })
        if rf_eid:
            supported_findings.append({
                "finding": f"Random Forest baseline predicted illicit probability of {rf_p:.2f}.",
                "evidence_id": rf_eid,
                "confidence": "HIGH"
            })
        if sub_eid:
            supported_findings.append({
                "finding": f"2-hop graph neighborhood contains {subgraph_nodes} reachable transactions.",
                "evidence_id": sub_eid,
                "confidence": "HIGH"
            })
        if exp_eid:
            supported_findings.append({
                "finding": f"GNNExplainer identified {len(top_features)} influential feature representations with fidelity drop of {fidelity_drop:.3f}.",
                "evidence_id": exp_eid,
                "confidence": "MEDIUM"
            })

        # Phase 5D: Structured Hypothesis Reasoning Classifications
        # Classify hypotheses as SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or NOT_TESTABLE
        model_eids = [e for e in (pred_eid, gat_eid, rf_eid) if e]
        hypothesis_evaluations = [
            {
                "hypothesis_id": "HYP-1-MULTI-MODEL-CONSENSUS",
                "hypothesis": "GNN and tree-based models converge on a consistent directional risk classification.",
                "classification": "PARTIALLY_SUPPORTED" if divergence else "SUPPORTED",
                "evidence_ids": model_eids,
                "rationale": (
                    f"Models diverge across architectures (spread={prob_spread:.2f}); tree models and graph attention weight local vs topological signals differently."
                    if divergence
                    else f"ETGNN ({etgnn_p:.2f}), Matched GAT ({gat_p:.2f}), Random Forest ({rf_p:.2f}), and XGBoost ({xgb_p:.2f}) exhibit bounded spread ({prob_spread:.2f} < 0.30)."
                ),
            },
            {
                "hypothesis_id": "HYP-2-NEIGHBORHOOD-CONTAGION",
                "hypothesis": "Target transaction is directly connected to confirmed illicit counterparties in its observed 2-hop snapshot subgraph.",
                "classification": (
                    "NOT_TESTABLE"
                    if is_fully_isolated or snapshot_id >= 41
                    else ("SUPPORTED" if illicit_nb > 0 else "UNSUPPORTED")
                ),
                "evidence_ids": [e for e in (tx_eid, sub_eid) if e],
                "rationale": (
                    "Node is completely isolated or resides in the post-cutoff test split where neighbor ground-truth labels are restricted under historical evaluation."
                    if is_fully_isolated or snapshot_id >= 41
                    else (
                        f"Observed 2-hop subgraph contains {illicit_nb} confirmed illicit neighbors."
                        if illicit_nb > 0
                        else "Observed 2-hop subgraph contains 0 confirmed illicit neighbors."
                    )
                ),
            },
            {
                "hypothesis_id": "HYP-3-EXOGENOUS-EVENT-ATTRIBUTION",
                "hypothesis": "Structural edge sparsity or risk score is causally attributable to an external real-world darknet marketplace takedown.",
                "classification": "NOT_TESTABLE",
                "evidence_ids": [tx_eid] if tx_eid else [],
                "rationale": "Elliptic features (1-165) and anonymized graph topology cannot establish exogenous real-world causal attribution.",
            },
        ]

        if not supported_findings and provenance:
            supported_findings.append({
                "finding": f"Transaction {tx_id} evaluated under historical snapshot context.",
                "evidence_id": provenance[0],
                "confidence": "HIGH"
            })

        limitations = [
            "Features are anonymized local (1-93) and aggregated (94-165) statistics without semantic identity.",
            "Predictions represent statistical classification risk, not verified judicial criminal evidence.",
        ]
        if snapshot_id == 43:
            limitations.append(
                "Snapshot 43 Observation: An observed structural distribution shift occurs around Snapshot 43. "
                "Its underlying cause has not been established by the available transaction graph evidence."
            )
        if is_fully_isolated:
            limitations.append(
                "Topological Qualification: Node is fully isolated in snapshot graph (both in-degree and out-degree are 0); "
                "structural message passing is inactive and prediction relies strictly on local tabular attributes."
            )
        elif is_isolated_in:
            limitations.append(
                "Topological Qualification: Node has zero incoming edges represented in the directed graph; "
                "this indicates an absence of observed upstream edges in the dataset sample, not proof that the transaction originated funds."
            )
        elif out_deg == 0 and in_deg > 0:
            limitations.append(
                "Topological Qualification: Node has zero outgoing edges represented in the directed graph; "
                "this indicates an absence of observed downstream spending edges, not proof of unspent funds or final accumulation."
            )

        narrative = (
            f"Transaction {tx_id} evaluated at snapshot {snapshot_id}. "
            f"ETGNN illicit probability is {etgnn_p:.2f}, Matched GAT is {gat_p:.2f}, Random Forest is {rf_p:.2f}, XGBoost is {xgb_p:.2f}. "
            f"Risk indicator is {risk_indicator}."
        )

        prompt_toks = max(1, len((system_prompt + " " + prompt).split()))
        comp_toks = 240
        report = {
            "investigation_id": inv_id,
            "target_transaction_id": tx_id,
            "snapshot_id": snapshot_id,
            "status": "COMPLETED",
            "investigation_status": "COMPLETED",
            "consensus_verdict": verdict,
            "predicted_class": int(mean_prob >= 0.50),
            "model_probability": round(etgnn_p, 4),
            "model_disagreement": divergence,
            "probability_spread": round(prob_spread, 4),
            "risk_indicator": risk_indicator,
            "epistemic_notice": (
                "Model probability represents raw statistical output, NOT a calibrated Bayesian confidence. "
                "Statistical risk predictions do not constitute judicial proof or legal evidence of financial crime."
            ),
            "provider_type": "mock",
            "provider_model_id": self.model_id,
            "generation_mode": "SIMULATED",
            "live_llm_verified": False,
            "token_usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": prompt_toks + comp_toks,
            },
            "llm_latency_ms": 1.2,
            "estimated_cost_usd": 0.0,
            "narrative_summary": narrative,
            "model_predictions": {
                "random_forest_prob": round(rf_p, 4),
                "xgboost_prob": round(xgb_p, 4),
                "canonical_etgnn_prob": round(etgnn_p, 4),
                "matched_gat_prob": round(gat_p, 4),
                "model_divergence_detected": divergence,
                "divergence_explanation": (
                    f"Model divergence detected (spread {prob_spread:.2f} >= 0.30): ETGNN and Matched GAT incorporate graph structure "
                    f"while tree baselines evaluate tabular features independently."
                    if divergence else None
                )
            },
            "graph_evidence": {
                "incoming_edges_count": in_deg,
                "outgoing_edges_count": out_deg,
                "subgraph_nodes_2hop": subgraph_nodes,
                "illicit_neighbors_count": illicit_nb,
                "is_isolated_incoming": is_isolated_in,
                "is_fully_isolated": is_fully_isolated,
            },
            "gnnexplainer_evidence": {
                "top_feature_indices": top_features,
                "fidelity_drop": round(fidelity_drop, 4),
                "interpretation_disclaimer": (
                    "GNNExplainer attributions identify sensitivity in model representations. "
                    "Controlled experiments show top-edge deletion does not statistically outperform random edge deletion "
                    "(p=0.6827), and indicates model false-alarm sensitivity rather than deterministic proof of fraud."
                )
            } if explanation else None,
            "supported_findings": supported_findings,
            "hypothesis_evaluations": hypothesis_evaluations,
            "uncertain_findings": [],
            "limitations": limitations,
            "evidence_limitations": limitations,
            "model_versions": {
                "etgnn_hash": "checkpoint_etgnn_reproduced_2026",
                "rf_hash": "rf_baseline_split32_2026",
                "agent_version": "ETGNN-Agent-v3.5.0"
            },
            "research_citations": [
                "Weber et al. (2019) Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks",
                "Elmougy & Liu (2023) Demystifying Fraudulent Transactions on the Bitcoin Network (Elliptic++)"
            ],
            "provenance_chain": provenance,
        }

        return json.dumps(report, indent=2)


# ============================================================================
# Mode 2: Deterministic Evidence-Only Provider (Non-Generative Factual Mode)
# ============================================================================

class EvidenceOnlyProvider(BaseLLMProvider):
    """
    Deterministic evidence-only provider.
    Produces structured factual reports without LLM generation or reasoning speculation.
    Explicitly marked as DETERMINISTIC_EVIDENCE_ONLY.
    """
    provider_name: str = "evidence_only"
    model_id: str = "deterministic_rules_v1"
    timeout_sec: float = 0.0
    max_retries: int = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        # Re-uses the deterministic factual synthesis from MockLLMProvider
        # but marks explicit non-generative generation mode
        mock_raw = MockLLMProvider().generate(prompt, system_prompt, context_data=context_data)
        report = json.loads(mock_raw)
        report["provider_type"] = "evidence_only"
        report["provider_model_id"] = self.model_id
        report["generation_mode"] = "DETERMINISTIC_EVIDENCE_ONLY"
        report["live_llm_verified"] = False
        report["narrative_summary"] = (
            "Deterministic non-generative factual summary derived directly from verified evidence records "
            "without LLM reasoning or synthetic text generation."
        )
        return json.dumps(report, indent=2)


# ============================================================================
# Mode 3: Live LLM Providers (Cloud / Local Inference)
# ============================================================================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini live provider via REST API."""
    provider_name: str = "gemini"
    model_id: str = "gemini-1.5-pro"
    timeout_sec: float = 30.0
    max_retries: int = 2

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMAuthenticationError("Gemini API key not found in GEMINI_API_KEY or GOOGLE_API_KEY environment variables.")
        self.model = model
        self.model_id = model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                res = client.post(self.endpoint, json=payload)
                if res.status_code == 429:
                    raise LLMRateLimitError(f"Gemini API rate limit exceeded: {res.text}")
                elif res.status_code in (401, 403):
                    raise LLMAuthenticationError(f"Gemini API authentication failed: {res.text}")
                res.raise_for_status()
                data = res.json()
                usage = data.get("usageMetadata", {})
                p_toks = int(usage.get("promptTokenCount", max(1, len((system_prompt + " " + prompt).split()))))
                c_toks = int(usage.get("candidatesTokenCount", 250))
                self.last_token_usage = {
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "total_tokens": int(usage.get("totalTokenCount", p_toks + c_toks)),
                }
                # Gemini 1.5 Pro standard pricing estimate ($1.25 / 1M input, $5.00 / 1M output)
                self.last_estimated_cost_usd = round((p_toks * 1.25 + c_toks * 5.00) / 1_000_000.0, 6)
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Gemini API timed out after {self.timeout_sec}s: {e}") from e
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Gemini API request failed: {e}") from e


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude live provider via REST API."""
    provider_name: str = "anthropic"
    model_id: str = "claude-3-5-sonnet-20241022"
    timeout_sec: float = 30.0
    max_retries: int = 2

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMAuthenticationError("Anthropic API key not found in ANTHROPIC_API_KEY environment variable.")
        self.model = model
        self.model_id = model
        self.endpoint = "https://api.anthropic.com/v1/messages"

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                res = client.post(self.endpoint, headers=headers, json=payload)
                if res.status_code == 429:
                    raise LLMRateLimitError(f"Anthropic API rate limit exceeded: {res.text}")
                elif res.status_code in (401, 403):
                    raise LLMAuthenticationError(f"Anthropic API authentication failed: {res.text}")
                res.raise_for_status()
                data = res.json()
                usage = data.get("usage", {})
                p_toks = int(usage.get("input_tokens", max(1, len((system_prompt + " " + prompt).split()))))
                c_toks = int(usage.get("output_tokens", 250))
                self.last_token_usage = {
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "total_tokens": p_toks + c_toks,
                }
                # Claude 3.5 Sonnet standard pricing estimate ($3.00 / 1M input, $15.00 / 1M output)
                self.last_estimated_cost_usd = round((p_toks * 3.00 + c_toks * 15.00) / 1_000_000.0, 6)
                return data["content"][0]["text"]
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Anthropic API timed out after {self.timeout_sec}s: {e}") from e
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Anthropic API request failed: {e}") from e


class OllamaProvider(BaseLLMProvider):
    """Local Ollama live provider via REST API."""
    provider_name: str = "ollama"
    model_id: str = "llama3:latest"
    timeout_sec: float = 45.0
    max_retries: int = 1

    def __init__(self, base_url: Optional[str] = None, model: str = "llama3:latest"):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model
        self.model_id = model
        self.endpoint = f"{self.base_url}/api/generate"

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                res = client.post(self.endpoint, json=payload)
                res.raise_for_status()
                data = res.json()
                p_toks = int(data.get("prompt_eval_count", max(1, len((system_prompt + " " + prompt).split()))))
                c_toks = int(data.get("eval_count", 250))
                self.last_token_usage = {
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "total_tokens": p_toks + c_toks,
                }
                self.last_estimated_cost_usd = 0.0
                return data.get("response", "")
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Ollama local API timed out after {self.timeout_sec}s: {e}") from e
        except Exception as e:
            raise LLMProviderError(f"Ollama local API request failed: {e}") from e


# ============================================================================
# Factory Method
# ============================================================================

def get_llm_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory function to instantiate the active LLM provider.
    Strictly instantiates the requested provider.
    Never silently falls back to MockLLMProvider when another provider is specified.
    """
    if provider_name is None:
        name = os.getenv("ETGNN_LLM_PROVIDER", "mock").lower().strip()
    else:
        name = str(provider_name).lower().strip()

    if name in ("mock", "offline", "test"):
        return MockLLMProvider()
    elif name in ("evidence_only", "deterministic"):
        return EvidenceOnlyProvider()
    elif name in ("gemini", "google"):
        return GeminiProvider()
    elif name in ("anthropic", "claude"):
        return AnthropicProvider()
    elif name in ("ollama", "local"):
        return OllamaProvider()
    else:
        raise LLMProviderError(
            f"Unsupported or unconfigured LLM provider: '{provider_name}'. "
            f"Supported providers are: 'mock', 'evidence_only', 'gemini', 'anthropic', 'ollama'."
        )
