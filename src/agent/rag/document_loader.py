from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.agent.schemas import SourceAuthorityTier, VerificationStatus

class ApprovedDocument(BaseModel):
    path: str
    title: str
    authority_tier: SourceAuthorityTier
    checksum: str
    raw_content: str
    is_authoritative_metric_source: bool = False

# Strict Approved Source Allowlist
APPROVED_SOURCE_ALLOWLIST = {
    # Tier 1: Machine-generated benchmark tables & reproduced results
    "results/tables/final_test_results.csv": (
        "Authoritative Test Benchmark Table",
        SourceAuthorityTier.TIER_1_REPRODUCED_METRICS,
        True,
    ),
    "docs/phase1_5/CHECKPOINT_REPRODUCIBILITY_REPORT.md": (
        "Phase 1.5 Checkpoint Reproducibility Report",
        SourceAuthorityTier.TIER_1_REPRODUCED_METRICS,
        True,
    ),
    # Tier 2: Verified Checkpoint and Dataset Integrity
    "docs/phase1_5/TEMPORAL_LEAKAGE_AUDIT.md": (
        "Temporal, Graph and Feature Leakage Audit",
        SourceAuthorityTier.TIER_2_CHECKPOINT_METADATA,
        False,
    ),
    "docs/phase1_audit/ETGNN_DATASET_INTEGRITY_REPORT.md": (
        "Dataset Integrity and Leakage Verification Report",
        SourceAuthorityTier.TIER_2_CHECKPOINT_METADATA,
        False,
    ),
    "docs/phase1_audit/ETGNN_MODEL_BASELINE_REPORT.md": (
        "Model Baseline and Checkpoint Verification Report",
        SourceAuthorityTier.TIER_2_CHECKPOINT_METADATA,
        False,
    ),
    # Tier 3: Reviewed Methodology & Transparency Specifications
    "docs/phase1_5/AGENT_TRANSPARENCY_SPECIFICATION.md": (
        "Agent Transparency and Evidence-Trace Specification",
        SourceAuthorityTier.TIER_3_REVIEWED_METHODOLOGY,
        False,
    ),
    "docs/phase1_5/CONTINUAL_LEARNING_EVALUATION_PROTOCOL.md": (
        "Continual-Learning Evaluation Protocol Specification",
        SourceAuthorityTier.TIER_3_REVIEWED_METHODOLOGY,
        False,
    ),
    "docs/phase1_audit/ETGNN_AGENT_ARCHITECTURE_PLAN.md": (
        "ETGNN-Agent System Architecture & Integration Plan",
        SourceAuthorityTier.TIER_3_REVIEWED_METHODOLOGY,
        False,
    ),
    # Tier 4: Qualified Audit Findings & Reviews
    "docs/phase1_5/SNAPSHOT43_CAUSALITY_REVIEW.md": (
        "Snapshot 43 Historical Causality & Structural Analysis Review",
        SourceAuthorityTier.TIER_4_QUALIFIED_AUDIT,
        False,
    ),
    "docs/phase1_audit/ETGNN_PHASE1_AUDIT_REPORT.md": (
        "Repository & Codebase Audit Report",
        SourceAuthorityTier.TIER_4_QUALIFIED_AUDIT,
        False,
    ),
}

class ApprovedDocumentLoader:
    """
    Loads documents exclusively from an approved research allowlist.
    Prevents indexing of raw model weights, environment variables, secrets, or unreviewed notes.
    """
    def __init__(self, root_dir: Optional[Path] = None):
        self.root = root_dir or Path(__file__).resolve().parent.parent.parent.parent

    def load_approved_documents(self) -> List[ApprovedDocument]:
        docs: List[ApprovedDocument] = []
        for rel_path, (title, tier, is_metric) in APPROVED_SOURCE_ALLOWLIST.items():
            full_path = self.root / rel_path
            if not full_path.exists():
                continue

            content = full_path.read_text(encoding="utf-8")
            sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

            docs.append(
                ApprovedDocument(
                    path=rel_path.replace("\\", "/"),
                    title=title,
                    authority_tier=tier,
                    checksum=sha256,
                    raw_content=content,
                    is_authoritative_metric_source=is_metric,
                )
            )
        return docs

    def get_source_checksums(self) -> Dict[str, str]:
        """Returns map of relative path -> SHA-256 for change detection."""
        checksums = {}
        for rel_path in APPROVED_SOURCE_ALLOWLIST.keys():
            full_path = self.root / rel_path
            if full_path.exists():
                checksums[rel_path.replace("\\", "/")] = hashlib.sha256(
                    full_path.read_bytes()
                ).hexdigest()
        return checksums

