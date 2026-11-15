# PDF & JSON Report Export Integrity Report (Phase 4.5)

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 4.5 Stabilization & Scientific Integrity Audit  
**Audited Module:** `src/agent/reports/pdf_exporter.py`  
**Status:** COMPLETE & VERIFIED  

---

## 1. Audit Findings & Checksum Determinism Fix

During the Phase 4.5 audit of `src/agent/reports/pdf_exporter.py`, two issues were identified and resolved:

1. **Non-Deterministic Python `hash()` Replaced with Canonical SHA-256:**
   - Previously, the footer digest used Python's built-in `hash()`, which is randomized across Python invocations via `PYTHONHASHSEED` and therefore could not be independently reproduced across sessions.
   - **Correction:** Replaced with a deterministic **SHA-256** digest (`hashlib.sha256`) computed over the UTF-8 bytes of a canonical key-sorted JSON object (`json.dumps(canonical_summary, sort_keys=True)`).
2. **Terminology Alignment:**
   - Updated the neighborhood summary in Section 3 of the PDF dossier from `illicit neighbors` to `illicit-label neighbors` to match the topological provenance policy.

---

## 2. Canonical SHA-256 Checksum Specification

The footer of every exported PDF dossier prints a 64-character hexadecimal SHA-256 digest computed over the following exact canonical JSON structure:

```python
canonical_summary = {
    "investigation_id": str(report_dict.get("investigation_id", "N/A")),
    "target_transaction_id": str(report_dict.get("target_transaction_id", "N/A")),
    "snapshot_id": str(report_dict.get("snapshot_id", "N/A")),
    "verdict": str(report_dict.get("consensus_verdict", "N/A")),
    "model_predictions": report_dict.get("model_predictions") or {},
    "provenance_chain": report_dict.get("provenance_chain", []),
}
canonical_bytes = json.dumps(canonical_summary, sort_keys=True).encode("utf-8")
canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
```

This allows any auditor holding the exported JSON report to independently recompute and verify the exact SHA-256 digest printed on the PDF dossier.

---

## 3. Preserved Epistemic & Provenance Elements in PDF & JSON Exports

Every generated PDF (`GET /api/agent/investigations/{id}/export/pdf`) and JSON download (`ETGNN_Investigation_{id}_Tx{tx_id}.json`) preserves:
- **Investigation Identity:** `investigation_id`, `target_transaction_id`, `snapshot_id`, `status`, and UTC timestamp.
- **Provider & Generation Mode Provenance:** Explicitly displays `generation_mode` (`SIMULATED`, `DETERMINISTIC_EVIDENCE_ONLY`, or `LIVE`) and `provider_type` (`mock`, `evidence_only`, `gemini`, `anthropic`, `ollama`).
- **Multi-Model Probability Table:** Displays `ETGNN (Canonical)`, `Random Forest`, and `XGBoost` probabilities alongside probability spread and divergence analysis.
- **Supported Findings & Verified Citations:** Lists each finding with its `[EV-...]` provenance citations.
- **Methodological Boundaries & Limitations:** Preserves all Snapshot 43 and topological degree qualifications.
- **Epistemic Boundary Notice:** Explicitly states that model probabilities represent raw statistical outputs, not calibrated Bayesian confidence or legal proof of financial crime.

---

## 4. Verification Evidence

- `test_pdf_export_deterministic_sha256_checksum` (`tests/test_phase4_5_reliability_and_integrity.py`): Verifies `generate_investigation_pdf()` returns valid PDF bytes (`%PDF...`, `> 500` bytes) and produces a deterministic 64-char SHA-256 checksum. — **PASSED**
- `Scenario K` (`web/frontend/e2e/investigation_copilot.spec.ts`): Verifies browser JSON download event and PDF export link attribute in Chromium. — **PASSED (2.2s)**

