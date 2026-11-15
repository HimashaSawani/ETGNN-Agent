# Human Analyst Feedback Security, Provenance & Isolation Audit (Phase 4.5)

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 4.5 Stabilization & Scientific Integrity Audit  
**Audited Module:** `src/agent/feedback/store.py` & `/api/agent/feedback` Endpoints  
**Status:** COMPLETE & VERIFIED  

---

## 1. Scientific & Checkpoint Isolation Guarantee

A core scientific integrity requirement of the ETGNN-Agent platform is that **human-in-the-loop analyst feedback must never automatically mutate original Elliptic ground-truth labels, trained model checkpoints, or historical evaluation benchmarks**.

### Verified Isolation Invariants
1. **Dedicated SQLite Storage:** All feedback records are persisted exclusively in `data/analyst_feedback.db` (or test-scoped temporary SQLite files).
2. **Read-Only Model & Dataset Separation:** `AnalystFeedbackStore` imports zero PyTorch or dataset mutation routines. Checkpoints (`models/checkpoints/*.pt`, `*.joblib`, `*.json`) and raw Elliptic CSV files (`data/raw/elliptic_bitcoin_dataset/*.csv`) remain strictly read-only.
3. **Human Feedback != Ground Truth:** Every feedback record is stored as an auditable review annotation (`FeedbackRecord`) with `safety_invariant = "read_only_isolation_no_automatic_retraining"`.

---

## 2. Input Validation & Data Sanitization Hardening

In Phase 4.5, `FeedbackSubmission` (`src/agent/feedback/store.py`) was hardened with strict Pydantic v2 `@field_validator` guards:

| Field | Validation & Sanitization Policy |
| :--- | :--- |
| `investigation_id` | `min_length=3, max_length=100`, non-empty string. |
| `transaction_id` | Integer `ge=0`. |
| `feedback_category` | Strict whitelist validation (`ALLOWED_CATEGORIES`): `{"helpful", "unhelpful", "citation_issue", "missing_evidence", "incorrect_numerical", "needs_further_review"}`. Rejects arbitrary or injection strings with `ValueError`. |
| `rating` | Integer bounded strictly to `1 <= rating <= 5`. |
| `comment` | Optional string `max_length=2000`; control characters and null bytes (`\x00`–`\x08`, `\x0b`, `\x0c`, `\x0e`–`\x1f`) are stripped via regex before storage. |
| `reviewer_id` | Regex validated `^[a-zA-Z0-9_\-\.@]+$` (`max_length=50`) to prevent log spoofing or script injection. |
| `evidence_references` | Capped at maximum 50 items; each reference truncated to 100 characters. |
| `created_at` | Server-generated UTC ISO-8601 timestamp (`datetime.now(timezone.utc).isoformat()`), preventing client timestamp forgery. |

---

## 3. SQLite Security & Concurrency Architecture

1. **100% Parameterized SQL Queries:** Every `INSERT` and `SELECT` statement in `AnalystFeedbackStore` uses `?` bound parameters, eliminating SQL injection vectors.
2. **Write-Ahead Logging (`WAL` Mode):** Enabled `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` inside `_init_db()` alongside `threading.RLock()` synchronization to ensure safe concurrent reads and writes without database locking errors.
3. **Explicit Single-User Local Deployment Boundary:**
   - Because the research platform operates as a local single-user forensic workstation without multi-tenant OAuth/JWT authentication, `get_feedback_stats()` and the UI explicitly declare `deployment_mode = "single_user_local_only"`.
   - **Security Disclosure:** This system is **not** designed to be exposed to untrusted multi-user public networks without placing an authenticated reverse proxy / RBAC gateway in front of FastAPI.

---

## 4. Verification Evidence

- `test_analyst_feedback_store_wal_mode_enabled` (`tests/test_phase4_5_reliability_and_integrity.py`): Verifies `PRAGMA journal_mode` returns `"wal"`. — **PASSED**
- `test_analyst_feedback_store_input_validation`: Verifies out-of-range ratings (`0`, `6`) and unwhitelisted categories (`"malicious_category_injection"`) raise `ValueError`, and null bytes (`\x00`) are stripped from comments. — **PASSED**
- `test_analyst_feedback_checkpoint_isolation`: Confirms database path isolation from `models/checkpoints` and `data/raw`. — **PASSED**
- `Scenario J` (Playwright E2E): Verifies end-to-end browser form submission and confirmation banner. — **PASSED (2.9s)**

