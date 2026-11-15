"""
Grounding and Evidence Verification Metrics.
Evaluates citation validity, coverage, numerical claim accuracy,
model identity fidelity, transaction provenance, and unsupported claim rates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class GroundingMetricResult:
    citation_existence_rate: float
    citation_precision: float
    citation_coverage: float
    numerical_correctness_rate: float
    unsupported_claim_rate: float
    tx_identity_correctness: float
    model_identity_correctness: float
    provenance_correctness: float
    sample_sizes: Dict[str, int]
    denominators: Dict[str, str]


class GroundingMetricsEvaluator:
    """Evaluates factual grounding and citation fidelity of agent reports."""

    KNOWN_MODELS = {
        "etgnn", "random_forest", "rf", "xgboost", "gat",
        "matched_gat", "gcn", "graphsage", "rf_local", "rf_with_time"
    }

    HALLUCINATED_MODELS = {"lstm", "gru", "transformer_conv", "rgcn", "bert"}

    @staticmethod
    def extract_citation_ids(text: str) -> List[str]:
        """Extracts cited evidence identifiers (e.g., EV-TX-12345, [EV-RAG-001])."""
        return re.findall(r"\[?(EV-[A-Za-z0-9_\-]+)\]?", text)

    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extracts floating point and integer numbers from text."""
        raw = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        res = []
        for r in raw:
            try:
                res.append(float(r))
            except ValueError:
                pass
        return res

    def evaluate_report_grounding(
        self,
        report_text: str,
        target_tx_id: int,
        registered_evidence_ids: Set[str],
        expected_numerical_facts: Dict[str, float],
        is_abstention: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates a single investigation report against registered evidence.
        """
        if is_abstention:
            return {
                "has_citations": True,
                "citation_precision": 1.0,
                "citation_coverage": 1.0,
                "numerical_correctness": 1.0,
                "unsupported_claim_rate": 0.0,
                "tx_identity_correct": True,
                "model_identity_correct": True,
                "provenance_correct": 1.0,
                "total_citations": 0,
                "valid_citations": 0,
                "claims_evaluated": 1,
            }

        citations = self.extract_citation_ids(report_text)
        total_cites = len(citations)
        valid_cites = sum(1 for c in citations if c in registered_evidence_ids)

        citation_precision = valid_cites / max(1, total_cites) if total_cites > 0 else 0.0
        has_citations = total_cites > 0

        # Numerical correctness check
        extracted_nums = self.extract_numbers(report_text)
        num_matches = 0
        total_expected_nums = len(expected_numerical_facts)
        for _, val in expected_numerical_facts.items():
            # Check if value exists within ±0.015 or exact integer
            matched = any(abs(n - val) < 0.015 or (val > 1 and abs(n - val) < 0.5) for n in extracted_nums)
            if matched:
                num_matches += 1

        numerical_correctness = (num_matches / max(1, total_expected_nums)) if total_expected_nums > 0 else 1.0

        # Transaction identity check
        tx_str = str(target_tx_id)
        tx_identity_correct = tx_str in report_text

        # Model identity check (must not hallucinate non-existent architectures)
        lower_text = report_text.lower()
        has_hallucinated_model = any(hm in lower_text for hm in self.HALLUCINATED_MODELS)
        model_identity_correct = not has_hallucinated_model

        # Citation coverage (estimates whether key assertions are grounded)
        coverage_den = max(1, total_expected_nums + 1)
        coverage_num = (1 if has_citations else 0) + num_matches
        citation_coverage = min(1.0, coverage_num / coverage_den)

        # Unsupported claims rate: claims without valid citation or hallucinated models
        unsupported_claims = (total_cites - valid_cites) + (1 if has_hallucinated_model else 0)
        total_claims = max(1, total_cites + total_expected_nums)
        unsupported_claim_rate = min(1.0, unsupported_claims / total_claims)

        # Provenance correctness: valid citations / total citations
        provenance_correct = citation_precision if total_cites > 0 else (1.0 if is_abstention else 0.0)

        return {
            "has_citations": has_citations,
            "citation_precision": citation_precision,
            "citation_coverage": citation_coverage,
            "numerical_correctness": numerical_correctness,
            "unsupported_claim_rate": unsupported_claim_rate,
            "tx_identity_correct": tx_identity_correct,
            "model_identity_correct": model_identity_correct,
            "provenance_correct": provenance_correct,
            "total_citations": total_cites,
            "valid_citations": valid_cites,
            "claims_evaluated": total_claims,
        }

    def aggregate_grounding(self, results: List[Dict[str, Any]]) -> GroundingMetricResult:
        """Aggregates single report evaluations into complete metric summary."""
        n = len(results)
        if n == 0:
            return GroundingMetricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})

        mean_has_cite = sum(1.0 if r["has_citations"] else 0.0 for r in results) / n
        mean_prec = sum(r["citation_precision"] for r in results) / n
        mean_cov = sum(r["citation_coverage"] for r in results) / n
        mean_num = sum(r["numerical_correctness"] for r in results) / n
        mean_unsup = sum(r["unsupported_claim_rate"] for r in results) / n
        mean_tx_id = sum(1.0 if r["tx_identity_correct"] else 0.0 for r in results) / n
        mean_model_id = sum(1.0 if r["model_identity_correct"] else 0.0 for r in results) / n
        mean_prov = sum(r["provenance_correct"] for r in results) / n

        sample_sizes = {
            "reports_evaluated": n,
            "total_citations_analyzed": sum(r["total_citations"] for r in results),
            "total_valid_citations": sum(r["valid_citations"] for r in results),
            "total_claims_analyzed": sum(r["claims_evaluated"] for r in results),
        }

        denominators = {
            "citation_existence_rate": "Total evaluated reports (n)",
            "citation_precision": "Total cited evidence identifiers across reports",
            "citation_coverage": "Total key factual assertions per report",
            "numerical_correctness_rate": "Total verified numerical facts per report",
            "unsupported_claim_rate": "Total claims asserted per report",
            "tx_identity_correctness": "Total reports evaluated",
            "model_identity_correctness": "Total reports evaluated",
            "provenance_correctness": "Total citation identifiers resolved",
        }

        return GroundingMetricResult(
            citation_existence_rate=round(mean_has_cite, 4),
            citation_precision=round(mean_prec, 4),
            citation_coverage=round(mean_cov, 4),
            numerical_correctness_rate=round(mean_num, 4),
            unsupported_claim_rate=round(mean_unsup, 4),
            tx_identity_correctness=round(mean_tx_id, 4),
            model_identity_correctness=round(mean_model_id, 4),
            provenance_correctness=round(mean_prov, 4),
            sample_sizes=sample_sizes,
            denominators=denominators,
        )
