# Model Promotion Policy
## Deterministic Multi-Metric Gate Specification and Human Approval Governance

**Project:** ETGNN-Agent  
**Document:** `docs/phase5/MODEL_PROMOTION_POLICY.md`  
**Implementation:** [`src/ml/continual/promotion_gate.py`](file:///D:/ETGNN-Fraud-Detection/src/ml/continual/promotion_gate.py)  
**Primary Metric:** AUPRC (Area Under the Precision-Recall Curve)  

---

### 1. Decision States

The `ModelPromotionGate` returns one of four deterministic outcomes:
1. **`ELIGIBLE_FOR_REVIEW`**: Candidate outperforms the frozen baseline by at least $\Delta \ge +0.015$ on AUPRC without exceeding false positive rate or latency tolerances, with multi-seed stability confirmed. **Mandates human investigator review.**
2. **`REJECTED`**: Candidate fails to meet minimum AUPRC improvement, exhibits FPR regression, excessive latency degradation, or high multi-seed variance.
3. **`INSUFFICIENT_EVIDENCE`**: Evaluation sample size is $< 100$, or fewer than 3 random seeds were evaluated.
4. **`BLOCKED_BY_LEAKAGE_RISK`**: Candidate provenance touches prior-exposed test partitions (Snapshots 41–49) or unverified upstream feature leakage.

---

### 2. Multi-Metric Gate Thresholds

| Metric | Evaluation Threshold | Action if Violated | Rationale |
| :--- | :---: | :---: | :--- |
| **Primary: AUPRC** | $\Delta \text{AUPRC} \ge +0.015$ (+1.5%) | `REJECTED` | Demonstrates genuine discriminatory uplift under extreme class imbalance. |
| **False Positive Rate (FPR)** | $\Delta \text{FPR} \le +0.020$ (+2.0%) | `REJECTED` | Prevents operational analyst alert fatigue. |
| **Inference Latency** | $\Delta \text{Latency} \le +15.0\text{ ms}$ | `REJECTED` | Preserves interactive copilot responsiveness. |
| **Multi-Seed Stability** | $\sigma_{\text{AUPRC}} \le 0.025$ | `REJECTED` | Discards models that succeed on only a single favorable random seed. |
| **Evaluation Sample Size** | $N \ge 100$ labelled cases | `INSUFFICIENT_EVIDENCE` | Prevents promotion on small unrepresentative cohorts. |
| **Seeds Evaluated** | $K \ge 3$ distinct seeds | `INSUFFICIENT_EVIDENCE` | Ensures reproducible generalization. |
| **Data Contamination** | Exposes Snapshots 41–49 | `BLOCKED_BY_LEAKAGE_RISK` | Upholds scientific test partition integrity. |

---

### 3. Human Approval Mandate

**No automated model replacement:**  
Even when a candidate achieves `ELIGIBLE_FOR_REVIEW`, the system **never automatically replaces** the active checkpoint in production. Model promotion mandates:
1. Written sign-off from the Lead AI Forensic Investigator.
2. Independent viva examination or audit verification.
3. Creation of an updated cryptographically signed baseline manifest.

