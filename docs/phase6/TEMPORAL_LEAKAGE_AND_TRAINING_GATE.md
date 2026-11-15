# ETGNN-Agent — Temporal Leakage Prevention & Continual Learning Safety Gate Audit

**Repository:** `D:\ETGNN-Fraud-Detection`  
**Auditor:** Production ML Reliability Engineer & MLOps Architect  
**Date:** 2026-11-11  

---

### 1. Invariant: Checkpoints and Raw Data Are Immutable
A foundational ethical and scientific invariant of this research platform is that baseline models and raw benchmarks remain strictly read-only:
- **Baseline Checkpoints:** All 11 files in `results/checkpoints/` match their registered SHA-256 hashes in `artifacts/baseline_manifest.json` 100%.
- **Raw Datasets:** All 4 CSV files in `data/raw/` match their registered SHA-256 hashes 100%.
- **No In-Place Overwrites:** No execution script or API endpoint is permitted to overwrite baseline checkpoints.

---

### 2. Continual Learning Gate: Status `BLOCKED`

The continual learning candidate promotion pipeline is locked at:
```json
{
  "status": "BLOCKED",
  "reason": "Continual learning candidate promotion gate locked. Retraining blocked pending prospective validation and human authorization.",
  "original_checkpoints_status": "IMMUTABLE_READ_ONLY"
}
```

#### Why Candidate Promotion is Blocked:
1. **Temporal Label Delay:** In real-world financial intelligence, ground-truth labels for illicit transactions are delayed by weeks or months. Training on unlabelled transactions or recent snapshots risks massive label noise.
2. **Extreme Class Imbalance:** Snapshot 43 contains only 24 illicit transactions out of 5,063 (0.47%). Fine-tuning a neural network on 24 positive instances induces catastrophic forgetting and severe probability distortion.
3. **Analyst Feedback Separation:** Subjective user feedback (ratings 1–5, helpful/unhelpful) collected via the SQLite database is strictly segregated from model training sets. Analyst opinions are **never** treated as ground-truth fraud labels.

---

### 3. Promotion Gate Invariants
If an experimental candidate model is evaluated in the future, it must satisfy all 5 hard gates before consideration:
1. **AUPRC Improvement:** Primary metric must improve by $\ge +0.015$ over the canonical ETGNN baseline.
2. **False Positive Rate Ceiling:** FPR increase must not exceed $\le 0.020$ on held-out test data.
3. **Zero Historical Leakage:** Training window must strictly precede test window ($t_{\text{train}} < t_{\text{eval}}$).
4. **Isolated Storage:** Candidate checkpoints must be stored in `artifacts/phase5_candidates/<candidate_id>/`, never in `results/checkpoints/`.
5. **Human Approval:** Automatic model promotion is architecturally disabled. Explicit human authorization is required.

