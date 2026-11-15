from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ValidationIssue(BaseModel):
    severity: str = Field(..., description="ERROR, WARNING, or INFO")
    field: str
    message: str

class ValidationResult(BaseModel):
    is_valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    sanitized_text: Optional[str] = None

class EvidenceValidator:
    """
    Deterministic validator enforcing research integrity, prompt-injection defense,
    and parameter boundary checks for RAG evidence items.
    """
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
        re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"exec\s*\(|eval\s*\(|os\.system", re.IGNORECASE),
        re.compile(r"drop\s+table|delete\s+from", re.IGNORECASE),
    ]

    def validate_transaction_id(self, tx_id: int, known_tx_ids: Optional[set] = None) -> ValidationResult:
        issues: List[ValidationIssue] = []
        if not isinstance(tx_id, int):
            issues.append(ValidationIssue(severity="ERROR", field="tx_id", message=f"Transaction ID must be integer, got {type(tx_id)}"))
        elif tx_id <= 0:
            issues.append(ValidationIssue(severity="ERROR", field="tx_id", message=f"Transaction ID must be positive, got {tx_id}"))
        elif known_tx_ids and tx_id not in known_tx_ids:
            issues.append(ValidationIssue(severity="ERROR", field="tx_id", message=f"Transaction ID {tx_id} not found in dataset."))

        return ValidationResult(is_valid=(len(issues) == 0), issues=issues)

    def validate_snapshot_step(self, step: int) -> ValidationResult:
        issues: List[ValidationIssue] = []
        if not (1 <= step <= 49):
            issues.append(ValidationIssue(
                severity="ERROR",
                field="snapshot_id",
                message=f"Snapshot step must be between 1 and 49, got {step}."
            ))
        return ValidationResult(is_valid=(len(issues) == 0), issues=issues)

    def validate_document_content_security(self, content: str) -> ValidationResult:
        """Scan retrieved document text for adversarial prompt injection vectors."""
        issues: List[ValidationIssue] = []
        sanitized = content

        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(content):
                issues.append(ValidationIssue(
                    severity="WARNING",
                    field="text_content",
                    message=f"Potentially adversarial prompt injection pattern detected: '{pattern.pattern}'"
                ))
                sanitized = pattern.sub("[FILTERED_POTENTIAL_INJECTION]", sanitized)

        return ValidationResult(
            is_valid=(len(issues) == 0),
            issues=issues,
            sanitized_text=sanitized,
        )

    def validate_metric_claim(
        self,
        model_name: str,
        metric_name: str,
        claimed_value: float,
        reproduced_metrics: Dict[str, Dict[str, float]],
        tolerance: float = 1e-4,
    ) -> ValidationResult:
        """Verify that a cited metric matches authoritative benchmark tables."""
        issues: List[ValidationIssue] = []
        norm_model = model_name.lower().replace("-", "_").replace(" ", "_")
        
        model_table = reproduced_metrics.get(norm_model)
        if not model_table:
            issues.append(ValidationIssue(
                severity="ERROR",
                field="model_name",
                message=f"Model '{model_name}' has no verified benchmark entry."
            ))
            return ValidationResult(is_valid=False, issues=issues)

        norm_metric = metric_name.lower().replace("-", "_")
        expected = model_table.get(norm_metric)
        if expected is None:
            issues.append(ValidationIssue(
                severity="ERROR",
                field="metric_name",
                message=f"Metric '{metric_name}' not recorded for model '{model_name}'."
            ))
            return ValidationResult(is_valid=False, issues=issues)

        if abs(claimed_value - expected) > tolerance:
            issues.append(ValidationIssue(
                severity="ERROR",
                field="metric_value",
                message=f"Claimed {metric_name}={claimed_value} diverges from verified benchmark {expected} (diff > {tolerance})."
            ))

        return ValidationResult(is_valid=(len(issues) == 0), issues=issues)

