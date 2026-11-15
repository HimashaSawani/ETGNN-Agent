# Phase 3 API Integration Guide

## 1. Overview & Service Base URL

The Phase 3 API extends the FastAPI backend in `src/web/app.py` with three core endpoints for autonomous multi-step Bitcoin fraud investigation:

- `POST /api/agent/investigate`
- `GET /api/agent/investigations/{investigation_id}`
- `GET /api/agent/investigations/{investigation_id}/events`

Base URL: `http://localhost:8000` (or `http://127.0.0.1:8000`)

---

## 2. API Endpoints

### 2.1 POST `/api/agent/investigate`
Executes an autonomous forensic investigation for a specified transaction.

#### Request Body (`application/json`):
```json
{
  "transaction_id": 232629023,
  "question": "Investigate model consensus and counterparty risk.",
  "evidence_mode": "RETROSPECTIVE_INVESTIGATION",
  "as_of_snapshot": null
}
```

#### Fields:
- `transaction_id` (int, required): Elliptic Bitcoin transaction ID.
- `question` (string, optional): Specific analytical hypothesis or focus. Default: `"Perform comprehensive forensic investigation of this Bitcoin transaction."`
- `evidence_mode` (string, optional): `"RETROSPECTIVE_INVESTIGATION"` (full historical context) or `"HISTORICAL_EVALUATION"` (point-in-time constrained).
- `as_of_snapshot` (int, optional): Required if `evidence_mode` is `"HISTORICAL_EVALUATION"` (valid range: 1–49).

#### Response (`200 OK`):
```json
{
  "investigation_id": "inv_9b83a0f12c",
  "status": "COMPLETED",
  "report": {
    "investigation_id": "inv_9b83a0f12c",
    "target_transaction_id": 232629023,
    "snapshot_id": 1,
    "status": "COMPLETED",
    "consensus_verdict": "HIGH_RISK_ILLICIT",
    "narrative_summary": "Transaction 232629023 exhibits high model-predicted illicit risk probability...",
    "model_predictions": { ... },
    "supported_findings": [
      {
        "finding": "ETGNN model predicted illicit risk probability of 0.94.",
        "evidence_id": "EV-PRED-232629023-etgnn",
        "confidence": "HIGH"
      }
    ],
    "uncertain_findings": [],
    "limitations": [ ... ]
  },
  "warnings": [],
  "errors": []
}
```

---

### 2.2 GET `/api/agent/investigations/{investigation_id}`
Retrieves a previously computed forensic report by its investigation ID.

#### Response (`200 OK`):
Same payload schema as `POST /api/agent/investigate`.

#### Errors:
- `404 Not Found`: If `investigation_id` is unknown or evicted from the LRU cache.

---

### 2.3 GET `/api/agent/investigations/{investigation_id}/events`
Retrieves the complete observable telemetry event stream for audit tracing.

#### Response (`200 OK`):
```json
[
  {
    "event_id": "evt_0a1b2c3d4e",
    "investigation_id": "inv_9b83a0f12c",
    "timestamp": "2026-11-09T09:40:00.123456Z",
    "event_type": "INVESTIGATION_STARTED",
    "status": "SUCCESS",
    "message": "Starting investigation",
    "duration_ms": 0,
    "safe_input_summary": null,
    "safe_output_summary": null,
    "evidence_ref_ids": []
  },
  {
    "event_id": "evt_5f6g7h8i9j",
    "investigation_id": "inv_9b83a0f12c",
    "timestamp": "2026-11-09T09:40:00.450123Z",
    "event_type": "INVESTIGATION_COMPLETED",
    "status": "SUCCESS",
    "message": "Investigation successfully completed with verified evidence.",
    "duration_ms": 450,
    "safe_input_summary": null,
    "safe_output_summary": {
      "verdict": "HIGH_RISK_ILLICIT",
      "findings_count": 4
    },
    "evidence_ref_ids": []
  }
]
```

