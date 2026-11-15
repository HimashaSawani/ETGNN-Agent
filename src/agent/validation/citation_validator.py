from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class CitationValidationResult(BaseModel):
    is_valid: bool
    total_citations: int
    valid_citations: int
    invalid_citations: List[str] = Field(default_factory=list)
    missing_evidence_ids: List[str] = Field(default_factory=list)
    mismatched_target_ids: List[int] = Field(default_factory=list)
    unsupported_claim_citations: List[str] = Field(default_factory=list)
    citation_coverage: float = 1.0
    validation_errors: List[str] = Field(default_factory=list)



class CitationValidator:
    """
    Validates evidence IDs and citations in forensic investigation reports.
    Ensures every analytical finding maps to an existing, immutable EvidenceRecord,
    verifies transaction identity consistency, and checks claim-to-evidence support.
    """
    EVIDENCE_ID_PATTERN = re.compile(r"^EV-(TX|PRED|SUB|GRAPH|EXP|EXPL|DOC|METRIC)-[A-Za-z0-9_\-\.]+$")

    def __init__(self, min_coverage: float = 0.80):
        self.min_coverage = min_coverage

    # Compatibility wrapper used by older test code
    def validate_claim_citations(
        self,
        claim_text: str,
        evidence_citations: List[str],
        evidence_bundle: Dict[str, Any],
        target_tx_id: Optional[int] = None,
    ) -> CitationValidationResult:
        """Forward to the primary ``validate_citations`` method.
        ``claim_text`` is currently not needed for validation logic.
        """
        provenance = evidence_bundle.get("provenance_chain", [])
        result = self.validate_citations(
            cited_evidence_ids=evidence_citations,
            valid_provenance_chain=provenance,
            target_tx_id=target_tx_id,
            evidence_bundle=evidence_bundle,
        )
        # Additional validation: ensure claim text aligns with evidence type.
        validation_errors = []
        for eid in evidence_citations:
            if isinstance(eid, str) and self.verify_claim_evidence_type_match(claim_text, eid):
                continue
            validation_errors.append("Evidence type mismatch")
        # Attach errors to result (if field exists).
        result.validation_errors = validation_errors
        return result

    def validate_citations(
        self,
        cited_evidence_ids: List[str],
        valid_provenance_chain: List[str],
        target_tx_id: Optional[int] = None,
        evidence_bundle: Optional[Dict[str, Any]] = None,
    ) -> CitationValidationResult:
        """Validate citation IDs against provenance and target transaction.
        Returns a ``CitationValidationResult`` describing validity, coverage and warnings.
        """
        provenance_set: Set[str] = set(valid_provenance_chain)
        invalid_format: List[str] = []
        missing_ids: List[str] = []
        mismatched_targets: List[str] = []
        valid_count = 0

        target_tx_str = str(target_tx_id) if target_tx_id is not None else None

        for raw_eid in cited_evidence_ids:
            # Accept both the standard EV- pattern and legacy "tx_<id>" style IDs used in tests.
            raw_str = str(raw_eid) if raw_eid is not None else ""
            stripped = raw_str.strip()
            # Check standard pattern first.
            if self.EVIDENCE_ID_PATTERN.match(stripped):
                eid = stripped
            elif stripped.lower().startswith("tx_") and stripped[3:].isdigit():
                # Legacy transaction ID format – treat as valid evidence ID.
                eid = stripped
            else:
                invalid_format.append(stripped)
                continue


            if eid not in provenance_set:
                missing_ids.append(eid)
                continue

            # Verify target transaction identity inside evidence ID where applicable
            if target_tx_str and (
                eid.startswith("EV-TX-")
                or eid.startswith("EV-PRED-")
                or eid.startswith("EV-GRAPH-")
                or eid.startswith("EV-EXPL-")
            ):
                parts = eid.split("-")
                if len(parts) >= 3:
                    extracted_id = parts[2]
                    if extracted_id.isdigit() and extracted_id != target_tx_str:
                        mismatched_targets.append(extracted_id)
                        continue
            # Legacy tx_ format mismatch check
            if target_tx_str and stripped.lower().startswith("tx_"):
                extracted_id = stripped.split("_", 1)[1]
                if extracted_id.isdigit() and extracted_id != target_tx_str:
                    mismatched_targets.append(int(extracted_id))
                    continue

            valid_count += 1

        total = len(cited_evidence_ids)
        coverage = (valid_count / total) if total > 0 else 1.0
        warnings = []

        if invalid_format:
            warnings.append(f"Found {len(invalid_format)} citations with invalid format: {invalid_format}")
        if missing_ids:
            warnings.append(f"Found {len(missing_ids)} citations not in verified provenance chain: {missing_ids}")
        if mismatched_targets:
            warnings.append(f"Found {len(mismatched_targets)} citations referencing mismatched transactions: {mismatched_targets}")

        is_valid = (
            len(invalid_format) == 0
            and len(missing_ids) == 0
            and len(mismatched_targets) == 0
            and coverage >= self.min_coverage
        )

        return CitationValidationResult(
            is_valid=is_valid,
            total_citations=total,
            valid_citations=valid_count,
            invalid_citations=invalid_format,
            missing_evidence_ids=missing_ids,
            mismatched_target_ids=mismatched_targets,
            citation_coverage=coverage,
            warnings=warnings,
        )

    def verify_claim_evidence_type_match(self, claim_text: str, evidence_id: str) -> bool:
        """Ensures the referenced evidence type semantically aligns with the assertion."""
        claim_lower = claim_text.lower()
        if evidence_id.startswith("EV-PRED-"):
            return any(k in claim_lower for k in ["predict", "probab", "model", "risk", "forest", "etgnn", "xgboost", "class"])
        elif evidence_id.startswith("EV-TX-"):
            return any(k in claim_lower for k in ["transaction", "degree", "snapshot", "feature", "in-degree", "out-degree", "node"])
        elif evidence_id.startswith("EV-GRAPH-") or evidence_id.startswith("EV-SUB-"):
            return any(k in claim_lower for k in ["graph", "neighbor", "hop", "subgraph", "reach", "connect", "path"])
        elif evidence_id.startswith("EV-EXP-") or evidence_id.startswith("EV-EXPL-"):
            return any(k in claim_lower for k in ["explain", "attribut", "mask", "fidelity", "influential", "drop"])
        elif evidence_id.startswith("EV-METRIC-"):
            return any(k in claim_lower for k in ["auprc", "f1", "precision", "recall", "metric", "benchmark", "accuracy"])
        elif evidence_id.startswith("EV-DOC-"):
            return False
        return False
