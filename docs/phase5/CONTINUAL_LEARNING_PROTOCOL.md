# Safe Continual Learning Protocol
## Experimental Framework, Candidate Isolation, and Anti-Contamination Rules

**Project:** ETGNN-Agent  
**Document:** `docs/phase5/CONTINUAL_LEARNING_PROTOCOL.md`  
**Target Artifact Directory:** `artifacts/phase5_candidates/`  
**Checkpoint Protection Status:** READ-ONLY IMMUTABLE  

---

### 1. Continual Learning Governance Principles

Continual learning is not treated as an automatic benefit in ETGNN-Agent. Without strict safeguards, continual updating risks catastrophic forgetting, false positive explosion, label contamination, and self-reinforcing bias.

```
+-----------------------------------------------------------------------------+
|                     CONTINUAL LEARNING PROTOCOL                             |
+-----------------------------------------------------------------------------+
|  [Step 1: Drift Justification]                                              |
|   - Require statistically significant performance drop or topological drift |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|  [Step 2: Candidate Experiment Planning]                                    |
|   - SafeCandidateTrainer creates unique versioned candidate ID              |
|   - Enforce data restrictions: Snapshots 41-49 strictly PROHIBITED as train |
|   - Save candidate artifacts ONLY to artifacts/phase5_candidates/           |
|   - results/checkpoints/ remains strictly IMMUTABLE                         |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|  [Step 3: Multi-Seed Controlled Training]                                   |
|   - Multi-seed runs (min 3 seeds: 42, 1337, 2026)                           |
|   - Full hyperparameter and dependency logging                              |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|  [Step 4: Deterministic Model Promotion Gate]                              |
|   - Must beat frozen baseline on AUPRC (primary metric) by >= +1.5%         |
|   - FPR increase <= +2.0%                                                   |
|   - Human approval MANDATORY before any promotion                           |
+-----------------------------------------------------------------------------+
```

---

### 2. Candidate Model Rules

1. **Immutable Production Baseline:**  
   The baseline checkpoints in `results/checkpoints/` (`etgnn.pt`, `gat.pt`, `random_forest.joblib`, etc.) have verified SHA-256 hashes recorded in [baseline_manifest.json](file:///D:/ETGNN-Fraud-Detection/artifacts/baseline_manifest.json). They cannot be overwritten, modified, or deleted by candidate training runs.
2. **Candidate Artifact Sandboxing:**  
   Candidate models must write solely to isolated subdirectories:
   `artifacts/phase5_candidates/<candidate_id>/`
3. **Data Eligibility Rules:**  
   - Only legitimately labelled transactions available prior to the candidate evaluation window are eligible.
   - Original test snapshots (41–49) cannot be treated as training data.
   - Analyst feedback is never used as training ground truth.
4. **Current Candidate Track Status: BLOCKED:**  
   Because Snapshots 41–49 were previously consulted during exploratory thesis analysis, and no untouched out-of-sample temporal partition currently exists in the public Elliptic release, candidate training runs targeting snapshots 41–49 are categorized as `BLOCKED`. Infrastructure is fully implemented and tested.

