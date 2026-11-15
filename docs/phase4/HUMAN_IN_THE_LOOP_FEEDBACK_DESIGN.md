# ETGNN-Agent — Phase 4: Human-in-the-Loop Feedback Design
## Feedback Store Architecture, Safety Isolation & Phase 5 Continual Learning Interface

**Module Path:** `src/agent/feedback/store.py`  
**Database Path:** `data/analyst_feedback.db`  
**Design Pattern:** Isolated Thread-Safe WAL SQLite Repository  

---

## 1. Safety & Scientific Invariants

A paramount requirement of the ETGNN-Agent system is **strict isolation between analyst feedback and existing machine learning checkpoints**:
1. **Zero Checkpoint Mutation**: Feedback records collected during analyst investigations are stored solely in an isolated SQLite database (`data/analyst_feedback.db`).
2. **Zero Label Pollution**: Analyst assessments **never** modify the original Elliptic ground truth labels (`data/raw/elliptic_txs_classes.csv`).
3. **No Uncontrolled Retraining**: Submitting feedback never triggers automated model fine-tuning or checkpoint modification. Feedback is accumulated exclusively for offline review, quality auditing, and future Phase 5 continual learning governance.

---

## 2. Schema Specification

The `analyst_feedback` table in `data/analyst_feedback.db` is initialized with WAL journaling:

```sql
CREATE TABLE IF NOT EXISTS analyst_feedback (
    feedback_id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    transaction_id INTEGER NOT NULL,
    feedback_category TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    reviewer_id TEXT DEFAULT 'anonymous_analyst',
    report_version TEXT DEFAULT 'ETGNN-Agent-v3.5.0',
    evidence_references TEXT, -- JSON array of cited EvidenceRecord IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_inv ON analyst_feedback (investigation_id);
CREATE INDEX IF NOT EXISTS idx_feedback_tx ON analyst_feedback (transaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_cat ON analyst_feedback (feedback_category);
```

---

## 3. Feedback Categories & Evaluation Taxonomy

Analyst feedback is classified into 6 standardized categories:

1. `helpful`: The investigation produced accurate predictions, topological reasoning, and faithful explanations.
2. `unhelpful`: The explanation was repetitive, uninformative, or vague.
3. `citation_issue`: A factual assertion cited the wrong evidence record or lacked evidentiary support.
4. `missing_evidence`: Critical topological or tabular features were omitted from the synthesis.
5. `incorrect_numerical`: The probability, degree count, or feature value differed from ground truth.
6. `needs_further_review`: High epistemic uncertainty or significant model divergence ($\ge 0.30$) requires senior committee review.

---

## 4. Phase 5 Retraining Handoff Contract

In Phase 5, feedback records with category `incorrect_numerical` or `citation_issue` will enter an offline vetting pipeline:
- Minimum quorum: $\ge 3$ senior analyst confirmations before an exemplar can be queued for retraining.
- Drift detection: Feedback distributions will be correlated against temporal snapshot drift metrics.
- Candidate buffer: Retraining sets will be generated as immutable versioned datasets, completely separate from the Elliptic benchmark dataset.

