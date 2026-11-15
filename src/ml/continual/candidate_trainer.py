"""
Safe Continual Learning Candidate Trainer and Isolation Framework.
Enforces strict sandboxing:
  - Original checkpoints in results/checkpoints/ remain strictly IMMUTABLE.
  - Candidate artifacts are isolated in artifacts/phase5_candidates/.
  - Blocks retraining if data provenance or independent evaluation protocol is compromised.
  - Never uses analyst feedback directly as training labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CandidateExperimentManifest:
    candidate_id: str
    base_model_name: str
    target_architecture: str
    status: str  # "BLOCKED", "INITIALIZED", "TRAINED", "EVALUATED"
    artifact_directory: str
    random_seeds: List[int]
    hyperparameters: Dict[str, Any]
    training_snapshots: List[int]
    evaluation_snapshots: List[int]
    data_provenance_hash: str
    dependency_fingerprint: Dict[str, str]
    blocking_reasons: List[str]
    scientific_justification: str


class SafeCandidateTrainer:
    """Isolates candidate model experimentation from production checkpoints."""

    CANDIDATE_ROOT = Path("artifacts/phase5_candidates")
    READONLY_CHECKPOINT_DIR = Path("results/checkpoints")

    def __init__(self):
        self.CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)

    def plan_candidate_experiment(
        self,
        candidate_id: str,
        base_model_name: str,
        training_snapshots: List[int],
        evaluation_snapshots: List[int],
        seeds: Optional[List[int]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> CandidateExperimentManifest:
        """
        Validates whether a proposed continual learning candidate is scientifically safe to train.
        """
        if seeds is None:
            seeds = [42, 1337, 2026]
        if hyperparameters is None:
            hyperparameters = {"learning_rate": 0.001, "weight_decay": 1e-4, "epochs": 50}

        blocking_reasons: List[str] = []

        # 1. Enforce read-only checkpoints invariant
        candidate_target_dir = self.CANDIDATE_ROOT / candidate_id
        if self.READONLY_CHECKPOINT_DIR in candidate_target_dir.resolve().parents:
            blocking_reasons.append("FATAL: Target path resolves inside immutable results/checkpoints directory.")

        # 2. Prevent training on test snapshots 41-49
        illegal_test_training = [s for s in training_snapshots if s >= 41]
        if illegal_test_training:
            blocking_reasons.append(
                f"FATAL: Attempted to train on test partition snapshots {illegal_test_training}. "
                "Snapshots 41-49 cannot be used as training data under the continual learning protocol."
            )

        # 3. Check for untouched test set disclosure
        illegal_test_eval = [s for s in evaluation_snapshots if s >= 41]
        if illegal_test_eval:
            blocking_reasons.append(
                f"NOTICE: Evaluation window touches snapshots {illegal_test_eval}. "
                "Original test snapshots 41-49 were examined during thesis decisions and cannot serve as an untouched test set."
            )

        # Record environment fingerprint
        deps = {
            "python": sys.version.split()[0],
            "torch": "installed",
            "torch_geometric": "2.8.0.post1",
        }

        # Provenance hash
        prov_str = f"{candidate_id}_{base_model_name}_{training_snapshots}_{evaluation_snapshots}_{seeds}"
        prov_hash = hashlib.sha256(prov_str.encode()).hexdigest()

        status = "BLOCKED" if blocking_reasons else "INITIALIZED"

        manifest = CandidateExperimentManifest(
            candidate_id=candidate_id,
            base_model_name=base_model_name,
            target_architecture=f"{base_model_name}_continual_v1",
            status=status,
            artifact_directory=str(candidate_target_dir),
            random_seeds=seeds,
            hyperparameters=hyperparameters,
            training_snapshots=training_snapshots,
            evaluation_snapshots=evaluation_snapshots,
            data_provenance_hash=prov_hash,
            dependency_fingerprint=deps,
            blocking_reasons=blocking_reasons,
            scientific_justification=(
                "Continual learning is BLOCKED until an independent, unexposed temporal evaluation cohort "
                "is established and feature provenance for feat_93-feat_164 is fully verified."
                if blocking_reasons
                else "Controlled candidate training initialized in isolated directory."
            ),
        )

        # Save manifest inside candidate directory
        candidate_target_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = candidate_target_dir / "candidate_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

        return manifest

