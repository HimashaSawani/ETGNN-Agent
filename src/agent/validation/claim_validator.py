from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClaimClassification(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    DERIVED = "DERIVED"
    HYPOTHESIS = "HYPOTHESIS"
    UNVERIFIED = "UNVERIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_TESTABLE = "NOT_TESTABLE"


class ClaimVerificationRecord(BaseModel):
    claim_text: str
    cited_evidence_id: Optional[str] = None
    classification: ClaimClassification
    reasons: List[str] = Field(default_factory=list)
    confidence: str = "HIGH"


class ClaimValidationSummary(BaseModel):
    is_valid: bool
    total_claims: int
    supported_count: int
    unsupported_count: int
    unverified_count: int
    claims: List[ClaimVerificationRecord] = Field(default_factory=list)
    critical_errors: List[str] = Field(default_factory=list)


class ClaimValidator:
    """Validates factual assertions and numerical claims against retrieved EvidenceRecords.
    Enforces metric-specific tolerances, prevents hallucinated semantics, prohibits uncalibrated
    certainty claims, and validates topological assertions.
    """
    PROHIBITED_CERTAINTY_PATTERNS = [
        re.compile(r"\b(guaranteed|undeniably|proves guilt|convicts|certainly a criminal)\b", re.IGNORECASE),
        re.compile(r"\b(confirmed money launderer|darknet operator confirmed|conclusive evidence of fraud)\b", re.IGNORECASE),
        re.compile(r"\b(calibrated confidence|bayesian confidence)\b", re.IGNORECASE),  # unverified calibration claim
    ]

    HALLUCINATED_FEATURE_PATTERNS = [
        re.compile(r"\b(wallet_age|wallet_balance|gas_fee|exchange_fee|mixer_fee|ip_address)\b", re.IGNORECASE),
        re.compile(r"\b(user_location|sender_identity|recipient_identity|owner_identity)\b", re.IGNORECASE),
    ]

    CAUSAL_DARKNET_FALLACY_PATTERNS = [
        re.compile(r"\b(darknet seizure caused this transaction|market takedown proved criminal)\b", re.IGNORECASE),
        re.compile(r"\b(snapshot 43 was caused by silk road|darknet shutdown caused|confirmed darknet structural shift)\b", re.IGNORECASE),
    ]

    # Metric-Specific Tolerances (Phase 3.5 Requirement)
    PROBABILITY_TOLERANCE: float = 0.01      # Precision-aware probability tolerance
    BENCHMARK_METRIC_TOLERANCE: float = 0.001 # AUPRC / F1 precision tolerance

    def __init__(self, probability_tolerance: float = 0.01, numerical_tolerance: Optional[float] = None):
        self.probability_tolerance = numerical_tolerance if numerical_tolerance is not None else probability_tolerance

    def validate_claim(
        self,
        finding_text: str,
        evidence_id: Optional[str],
        evidence_bundle: Dict[str, Any],
        target_tx_id: Optional[int] = None,
    ) -> ClaimVerificationRecord:
        reasons: List[str] = []
        classification = ClaimClassification.SUPPORTED

        # 1. Check prohibited certainty claims
        for pat in self.PROHIBITED_CERTAINTY_PATTERNS:
            if pat.search(finding_text):
                classification = ClaimClassification.UNSUPPORTED
                reasons.append(f"Prohibited definitive or uncalibrated certainty claim detected: matches '{pat.pattern}'")

        # 2. Check hallucinated feature names
        for pat in self.HALLUCINATED_FEATURE_PATTERNS:
            if pat.search(finding_text):
                classification = ClaimClassification.UNSUPPORTED
                reasons.append(f"Hallucinated feature semantics detected: matches '{pat.pattern}' (Elliptic features are anonymized 1-165)")

        # 3. Check causal darknet fallacy
        for pat in self.CAUSAL_DARKNET_FALLACY_PATTERNS:
            if pat.search(finding_text):
                classification = ClaimClassification.UNSUPPORTED
                reasons.append("Unwarranted causal assertion linking market takedown to individual guilt.")

        # 4. Check evidence ID linkage
        provenance = evidence_bundle.get("provenance_chain", [])
        if not evidence_id or evidence_id not in provenance:
            if classification == ClaimClassification.SUPPORTED:
                classification = ClaimClassification.UNVERIFIED
            reasons.append(f"Evidence ID '{evidence_id}' is missing or not in verified provenance chain")

        # 5. Exact match on target transaction ID
        if target_tx_id is not None:
            tx_id_matches = re.findall(r"\b\d{7,12}\b", finding_text)
            for raw_tid in tx_id_matches:
                if int(raw_tid) != target_tx_id:
                    classification = ClaimClassification.UNSUPPORTED
                    reasons.append(f"Mismatched transaction ID: claim asserts {raw_tid}, but target is {target_tx_id}")

        # 6. Metric-specific numerical alignment for model predictions
        #    a) Probability predictions
        prob_matches = re.findall(r"(\\d+(?:\\.\\d+)?)\\s*%", finding_text)
        if not prob_matches:
            prob_matches = re.findall(r"\\b0\\.\\d+\\b", finding_text)

        predictions = evidence_bundle.get("predictions", {})
        matched_prob = None
        for model_key, pred in predictions.items():
            if isinstance(pred, dict) and pred.get("evidence_id") == evidence_id:
                matched_prob = pred.get("illicit_probability")
                break
            elif hasattr(pred, "evidence_id") and pred.evidence_id == evidence_id:
                matched_prob = getattr(pred, "illicit_probability", None)
                break

        if matched_prob is not None and prob_matches:
            try:
                val = float(prob_matches[0])
                if "%" in finding_text or val > 1.0:
                    val = val / 100.0
                if abs(val - matched_prob) > self.PROBABILITY_TOLERANCE:
                    classification = ClaimClassification.UNSUPPORTED
                    reasons.append(
                        f"Numeric discrepancy: probability {val:.4f} differs from source evidence probability {matched_prob:.4f} (tolerance ±{self.PROBABILITY_TOLERANCE})"
                    )
            except ValueError:
                pass

        #    b) Benchmark metric validation
        if evidence_id:
            metrics = evidence_bundle.get("metrics", [])
            for metric in metrics:
                # support dict or simple object
                if isinstance(metric, dict):
                    evid_id = metric.get("evidence_id")
                    metric_name = metric.get("metric_name")
                    metric_val = metric.get("metric_value")
                else:
                    evid_id = getattr(metric, "evidence_id", None)
                    metric_name = getattr(metric, "metric_name", None)
                    metric_val = getattr(metric, "metric_value", None)
                if evid_id == evidence_id and metric_name:
                    # try to extract numeric claim for this metric
                    pattern = rf"{re.escape(metric_name)}.*?([0-9]+(?:\.[0-9]+)?)"
                    m = re.search(pattern, finding_text, re.IGNORECASE)
                    if m:
                        claimed_metric = float(m.group(1))
                        if abs(claimed_metric - metric_val) > self.BENCHMARK_METRIC_TOLERANCE:
                            classification = ClaimClassification.UNSUPPORTED
                            reasons.append(
                                f"Metric discrepancy: {metric_name} {claimed_metric:.4f} differs from source {metric_val:.4f} (tolerance ±{self.BENCHMARK_METRIC_TOLERANCE})"
                            )
                    break

        if evidence_id:
            prob_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", finding_text)
            if not prob_matches:
                prob_matches = re.findall(r"\b0\.\d+\b", finding_text)

            predictions = evidence_bundle.get("predictions", {})
            matched_prob = None
            for model_key, pred in predictions.items():
                if isinstance(pred, dict) and pred.get("evidence_id") == evidence_id:
                    matched_prob = pred.get("illicit_probability")
                    break
                elif hasattr(pred, "evidence_id") and pred.evidence_id == evidence_id:
                    matched_prob = getattr(pred, "illicit_probability", None)
                    break

            if matched_prob is not None and prob_matches:
                try:
                    val = float(prob_matches[0])
                    if "%" in finding_text or val > 1.0:
                        val = val / 100.0
                    if abs(val - matched_prob) > self.PROBABILITY_TOLERANCE:
                        classification = ClaimClassification.UNSUPPORTED
                        reasons.append(
                            f"Numeric discrepancy: probability {val:.4f} differs from source evidence probability {matched_prob:.4f} (tolerance ±{self.PROBABILITY_TOLERANCE})"
                        )
                except ValueError:
                    pass

        # 7. Exact match on graph degrees
        if evidence_id and evidence_id.startswith("EV-TX-"):
            tx_ev = evidence_bundle.get("transaction")
            if tx_ev:
                true_in = tx_ev.get("in_degree") if isinstance(tx_ev, dict) else getattr(tx_ev, "in_degree", None)
                true_out = tx_ev.get("out_degree") if isinstance(tx_ev, dict) else getattr(tx_ev, "out_degree", None)

                in_match = re.search(r"in-degree\s*(?:of)?\s*(\d+)", finding_text, re.IGNORECASE)
                if in_match and true_in is not None:
                    asserted_in = int(in_match.group(1))
                    if asserted_in != true_in:
                        reasons.append(f"In-degree mismatch: claim asserts {asserted_in}, exact source is {true_in}")

                out_match = re.search(r"out-degree\s*(?:of)?\s*(\d+)", finding_text, re.IGNORECASE)
                if out_match and true_out is not None:
                    asserted_out = int(out_match.group(1))
                    if asserted_out != true_out:
                        reasons.append(f"Out-degree mismatch: claim asserts {asserted_out}, exact source is {true_out}")

        return ClaimVerificationRecord(
            claim_text=finding_text,
            cited_evidence_id=evidence_id,
            classification=classification,
            reasons=reasons,
            confidence="HIGH" if classification == ClaimClassification.SUPPORTED else "LOW",
        )

    def validate_findings(
        self,
        findings: List[Dict[str, Any]],
        evidence_bundle: Dict[str, Any],
        target_tx_id: Optional[int] = None,
    ) -> ClaimValidationSummary:
        """Validate a list of claim dictionaries.

        Supports both ``claim_text`` (preferred) and legacy ``finding`` keys.
        Numerical probability tolerance is applied to any prediction evidence
        (not limited to IDs starting with ``EV-PRED-``).
        """
        records: List[ClaimVerificationRecord] = []
        critical_errors: List[str] = []
        supported_count = 0
        unsupported_count = 0
        unverified_count = 0

        for f in findings:
            text = f.get("claim_text") or f.get("finding") or ""
            eid = f.get("evidence_id")
            if not eid:
                citations = f.get("evidence_citations")
                if isinstance(citations, list) and len(citations) > 0:
                    eid = citations[0]
            rec = self.validate_claim(text, eid, evidence_bundle, target_tx_id=target_tx_id)
            records.append(rec)
            if rec.classification == ClaimClassification.SUPPORTED:
                supported_count += 1
            elif rec.classification == ClaimClassification.UNSUPPORTED:
                unsupported_count += 1
                critical_errors.extend(rec.reasons)
            elif rec.classification in (ClaimClassification.UNVERIFIED, ClaimClassification.HYPOTHESIS):
                unverified_count += 1

        is_valid = (unsupported_count == 0)
        return ClaimValidationSummary(
            is_valid=is_valid,
            total_claims=len(findings),
            supported_count=supported_count,
            unsupported_count=unsupported_count,
            unverified_count=unverified_count,
            claims=records,
            critical_errors=critical_errors,
        )
