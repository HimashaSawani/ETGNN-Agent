from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.agent.validation.claim_validator import ClaimValidator, ClaimValidationSummary
from src.agent.validation.citation_validator import CitationValidator, CitationValidationResult


class ReportValidationResult(BaseModel):
    is_valid: bool
    can_proceed: bool
    should_regenerate: bool
    reasons: List[str] = Field(default_factory=list)
    claim_summary: ClaimValidationSummary
    citation_summary: CitationValidationResult
    sanitized_report: Optional[Dict[str, Any]] = None


class ReportValidator:
    """
    Validates end-to-end report integrity against research ground truth,
    epistemic boundaries, topological distinctions, and temporal leakage policies.
    """
    def __init__(self):
        self.claim_validator = ClaimValidator()
        self.citation_validator = CitationValidator()

    def validate_report(
        self,
        report_dict: Dict[str, Any],
        evidence_bundle: Dict[str, Any],
        snapshot_id: int,
        target_tx_id: Optional[int] = None,
    ) -> ReportValidationResult:
        reasons: List[str] = []
        should_regenerate = False

        # 1. Validate citations of supported findings
        findings = report_dict.get("supported_findings", [])
        cited_ids = [
            f.get("evidence_id") if isinstance(f, dict) else getattr(f, "evidence_id", "")
            for f in findings
        ]
        provenance = evidence_bundle.get("provenance_chain", [])
        cit_result = self.citation_validator.validate_citations(
            cited_ids,
            provenance,
            target_tx_id=target_tx_id,
            evidence_bundle=evidence_bundle,
        )

        if not cit_result.is_valid:
            reasons.extend(cit_result.warnings)
            should_regenerate = True

        # 2. Validate factual and numerical claims with exact/precision tolerances
        claim_summary = self.claim_validator.validate_findings(
            findings,
            evidence_bundle,
            target_tx_id=target_tx_id,
        )
        if not claim_summary.is_valid:
            reasons.extend(claim_summary.critical_errors)
            should_regenerate = True

        # 3. Snapshot 43 qualification & topological distinction policy
        limitations = report_dict.get("limitations", [])
        if not isinstance(limitations, list):
            limitations = []

        tx_ev = evidence_bundle.get("transaction")
        # Missing transaction/topology evidence is NOT zero-degree topology.
        # Only explicitly verified degree values may drive isolation classification.
        in_deg = tx_ev.get("in_degree") if isinstance(tx_ev, dict) else getattr(tx_ev, "in_degree", None)
        out_deg = tx_ev.get("out_degree") if isinstance(tx_ev, dict) else getattr(tx_ev, "out_degree", None)
        topology_known = isinstance(in_deg, (int, float)) and isinstance(out_deg, (int, float))
        if not topology_known:
            in_deg = out_deg = None
        is_fully_isolated = topology_known and in_deg == 0 and out_deg == 0
        is_isolated_in = topology_known and in_deg == 0 and out_deg > 0

        if snapshot_id == 43:
            # Combine both 'limitations' and 'evidence_limitations' to detect qualification
            ev_lims = report_dict.get("evidence_limitations", [])
            combined_limitations = limitations + (ev_lims if isinstance(ev_lims, list) else [])
            limitations_text = " ".join([str(l) for l in combined_limitations]).lower()
            # If Snapshot 43 qualification is missing, record a reason to invalidate the report
            if "snapshot 43" not in limitations_text:
                reasons.append("Missing required Snapshot 43 qualification")
                limitations.append(
                    "Snapshot 43 Observation: An observed structural distribution shift occurs around Snapshot 43. "
                    "Its underlying cause has not been established by the available transaction graph evidence."
                )

            # already added above if necessary


        if is_fully_isolated:
            if not any(
                "fully isolated" in str(l).lower()
                for l in list(limitations) + list(report_dict.get("evidence_limitations", []) or [])
            ):
                # Record failure reason for missing qualification
                reasons.append("Missing required topology qualification for fully isolated node")
                limitations.append(
                    "Topological Qualification: Node is fully isolated in snapshot graph (both in-degree and out-degree are 0); "
                    "structural message passing is inactive and prediction relies strictly on local tabular attributes."
                )
        elif is_isolated_in:
            if not any("zero incoming" in str(l).lower() or "0 incoming" in str(l).lower() for l in limitations):
                limitations.append(
                    "Topological Qualification: Node has zero incoming observed edges represented in the directed graph; "
                    "this indicates an absence of observed upstream edges in the dataset sample, not proof that the transaction originated funds."
                )
        elif topology_known and out_deg == 0 and in_deg > 0:
            if not any("zero outgoing" in str(l).lower() or "0 outgoing" in str(l).lower() for l in limitations):
                limitations.append(
                    "Topological Qualification: Node has zero outgoing observed edges represented in the directed graph; "
                    "this indicates an absence of observed downstream spending edges, not proof of unspent funds or final accumulation."
                )

        report_dict["limitations"] = limitations

        # 4. GNNExplainer interpretation disclaimer check
        explanation = evidence_bundle.get("explanation")
        if explanation:
            gnnexplainer_evidence = report_dict.get("gnnexplainer_evidence", {})
            if isinstance(gnnexplainer_evidence, dict) and "interpretation_disclaimer" not in gnnexplainer_evidence:
                gnnexplainer_evidence["interpretation_disclaimer"] = (
                    "GNNExplainer attributions identify sensitivity in model representations. "
                    "Controlled experiments show top-edge deletion does not statistically outperform random edge deletion "
                    "(p=0.6827), and indicates model false-alarm sensitivity rather than deterministic proof of fraud."
                )
                report_dict["gnnexplainer_evidence"] = gnnexplainer_evidence

        # 5. Fallback degradation: Move unsupported/unverified claims to uncertain_findings
        if claim_summary.unsupported_count > 0 or claim_summary.unverified_count > 0:
            cleaned_findings = []
            uncertain_findings = report_dict.get("uncertain_findings", [])
            for f, rec in zip(findings, claim_summary.claims):
                if rec.classification.value in ("SUPPORTED", "DERIVED"):
                    cleaned_findings.append(f)
                else:
                    claim_str = f.get("finding") if isinstance(f, dict) else str(f)
                    uncertain_findings.append(f"Degraded claim: {claim_str} (Reason: {'; '.join(rec.reasons)})")
            report_dict["supported_findings"] = cleaned_findings
            report_dict["uncertain_findings"] = uncertain_findings

        is_valid = len(reasons) == 0
        can_proceed = True  # Can proceed if sanitized/degraded

        return ReportValidationResult(
            is_valid=is_valid,
            can_proceed=can_proceed,
            should_regenerate=should_regenerate,
            reasons=reasons,
            claim_summary=claim_summary,
            citation_summary=cit_result,
            sanitized_report=report_dict
        )
