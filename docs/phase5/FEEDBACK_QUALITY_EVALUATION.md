# Analyst Feedback Quality Evaluation
## Epistemic Governance and Provenance Safeguards for Qualitative Analyst Reviews

**Project:** ETGNN-Agent  
**Document:** `docs/phase5/FEEDBACK_QUALITY_EVALUATION.md`  
**Database:** `data/analyst_feedback.db` (6 review records)  
**Implementation:** [`src/agent/feedback/evaluator.py`](file:///D:/ETGNN-Fraud-Detection/src/agent/feedback/evaluator.py)  

---

### 1. Empirical Feedback Metrics

The feedback database contains review records submitted via the Phase 4 investigation dashboard:

```
Feedback Quality Metrics Summary:
  Total Feedback Records:          6 records
  Unique Investigations:           5 distinct investigations
  Unique Transactions:             2 unique transactions (230425980, 999999999)
  Unique Reviewers:                2 distinct reviewers (analyst_alice, auditor_e2e)
  Mean Usability Rating:           5.0 / 5.0
  Rating Distribution:             5-Star: 6 records (100%)
  Category Distribution:           'helpful': 6 records (100%)
  Disagreement Incidents:          0 transactions with conflicting ratings
  Citation Issue Frequency:        0.0%
  Missing Evidence Frequency:      0.0%
  Numerical Error Frequency:       0.0%
```

---

### 2. Epistemic Governance Rules

1. **Analyst Opinion != Fraud Ground Truth:**  
   Analyst ratings evaluate the qualitative readability, clarity, and perceived usefulness of the investigation report. A 5-star rating indicates that the analyst found the explanation clear; **it does NOT establish that the transaction was legitimately illicit or licit**.
2. **Never Derive Training Labels from Feedback:**  
   Analyst reviews are strictly isolated from the training dataset. Continual learning candidate trainers are forbidden from consuming feedback comments or ratings as supervised classification targets.
3. **Adjudication Hierarchy:**  
   - Level 1: Official Ground Truth (Raw labels verified by Elliptic)
   - Level 2: Epistemic Validation (Mathematical proof of graph degree and point-in-time features)
   - Level 3: Qualitative Feedback (Subjective analyst perceptions in `analyst_feedback.db`)

