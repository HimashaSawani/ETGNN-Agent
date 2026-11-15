# Temporal Data Leakage Audit
## Comprehensive Provenance & Point-in-Time Verification Gate

**Project:** ETGNN-Agent  
**Document:** `docs/phase5/TEMPORAL_LEAKAGE_AUDIT.md`  
**Mandatory Gate Status:** VERIFIED WITH DOCUMENTED UNKNOWN PROVENANCE  
**Date:** November 11, 2026  

---

### 1. Audit Framework & Classification Standard

Every pipeline stage, preprocessing operation, graph edge, and feature subset is classified into one of four formal categories:
- **`VERIFIED_SAFE`**: Mathematically and chronologically verified to use only information strictly available prior to or at time $t$.
- **`VERIFIED_LEAKAGE`**: Confirmed leakage of future information into the model input or training pipeline.
- **`UNKNOWN_PROVENANCE`**: Data provenance cannot be verified from published documentation or source code.
- **`NOT_APPLICABLE`**: Components operating purely at test time on authorized point-in-time fixtures.

---

### 2. Systematic Pipeline Audit

| Pipeline Component | Description / Operation | Finding Classification | Evidence & Provenance Details |
| :--- | :--- | :---: | :--- |
| **Chronological Split** | Snapshots 1–32 (Train), 33–40 (Val), 41–49 (Test) | **`VERIFIED_SAFE`** | Strictly partition-based. No future nodes or edges exist in train split. |
| **Feature Scaling** | `train_scaler.joblib` fitted on training split only | **`VERIFIED_SAFE`** | Fit strictly on `train_mask`. Zero test-set mean/variance exposure. |
| **Local Features (`feat_0`–`feat_92`)** | Raw transaction fees, inputs, outputs, lock time | **`VERIFIED_SAFE`** | Derived strictly from individual Bitcoin transaction payloads. |
| **Aggregated Features (`feat_93`–`feat_164`)** | 1-hop backward/forward neighbor aggregation | **`UNKNOWN_PROVENANCE`** | Weber et al. (2019) release does not document whether aggregation timestamps within the 2-week window were strictly point-in-time. |
| **Graph Edges** | Directed Bitcoin transaction payments ($u \to v$) | **`VERIFIED_SAFE`** | Directed edges connect inputs to outputs. No future edges are visible to historical queries. |
| **Neighbor Retrieval** | Graph subgraph retrieval tool | **`VERIFIED_SAFE`** | In `HISTORICAL_EVALUATION` mode, filtered strictly by `as_of_snapshot`. |
| **Model Checkpoints** | Checkpoints in `results/checkpoints/` | **`VERIFIED_SAFE`** | Checkpoints frozen; hashes match baseline manifest. |
| **RAG Retrieval** | Literature chunks & synthetic evidence | **`VERIFIED_SAFE`** | Retrieved evidence records carry explicit snapshot metadata. |
| **Analyst Feedback** | Persisted ratings in `analyst_feedback.db` | **`VERIFIED_SAFE`** | Strictly isolated from training labels. Classified as opinion only. |
| **Test Set Re-Use** | Original Test Snapshots 41–49 | **`UNKNOWN_PROVENANCE` / NOTICE** | Prior thesis milestones examined Snapshots 41–49 for error analysis. Disclosed as prior-exposed partition. |

---

### 3. Scientific Impact & Claim Restrictions

1. **Restriction on Complete Leakage-Freedom Claim:**  
   Because features `feat_93` through `feat_164` carry `UNKNOWN_PROVENANCE` in the original Elliptic data release (Weber et al. 2019), ETGNN-Agent **does NOT claim absolute temporal leakage freedom**. Publications and thesis statements must state that local features (`0`–`92`) are verified point-in-time, while aggregated features (`93`–`164`) carry an inherent unverified upstream provenance.
2. **Prior Test Set Disclosure:**  
   Continual learning candidate experiments cannot treat Snapshots 41–49 as an untouched virgin test set. Any candidate model claiming superior performance on 41–49 must disclose prior exploratory exposure.

