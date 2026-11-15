# ETGNN-Agent — Phase 4: API Integration Report
## Backend RESTful Contracts, OpenAPI Documentation & Serialization

**FastAPI Application:** `src/web/app.py`  
**Frontend Client:** `web/frontend/src/services/agentApi.ts`  
**API Prefix:** `/api/agent`  

---

## 1. Phase 4 Endpoints Specification

### 1. Investigation Execution
- **Endpoint**: `POST /api/agent/investigate`
- **Request**:
  ```json
  {
    "transaction_id": 288717023,
    "question": "Assess counterparty illicit risk and topological fan-out",
    "evidence_mode": "RETROSPECTIVE_INVESTIGATION",
    "llm_provider": "mock",
    "allow_evidence_only_fallback": true
  }
  ```
- **Response**: `200 OK` (`AgentInvestigateResponse` containing `investigation_id`, `status`, and `report`).

### 2. Observable Telemetry Events
- **Endpoint**: `GET /api/agent/investigations/{investigation_id}/events`
- **Response**: Array of `AgentInvestigationEvent` with stage timestamps, durations, and safe summaries.

### 3. Investigation List
- **Endpoint**: `GET /api/agent/investigations`
- **Response**: List of recent cached investigation summary items for session history.

### 4. Publication-Quality PDF Export
- **Endpoint**: `GET /api/agent/investigations/{investigation_id}/export/pdf`
- **Response**: `200 OK`, `Content-Type: application/pdf`, `Content-Disposition: attachment; filename="ETGNN_Forensic_Report_{investigation_id}.pdf"`.

### 5. Interactive Follow-Up Chat
- **Endpoint**: `POST /api/agent/chat`
- **Request**:
  ```json
  {
    "investigation_id": "inv_288717023_abc",
    "question": "What is the probability spread between ETGNN and Random Forest?",
    "llm_provider": "mock"
  }
  ```
- **Response**: `200 OK` (`ChatResponse` with grounded answer, cited evidence IDs, and limitations).

### 6. Analyst Feedback Submission & Query
- **Endpoint**: `POST /api/agent/feedback`
- **Request**:
  ```json
  {
    "investigation_id": "inv_288717023_abc",
    "transaction_id": 288717023,
    "feedback_category": "helpful",
    "rating": 5,
    "comment": "Accurately noted multi-hop neighborhood and feature limitations.",
    "reviewer_id": "analyst_alice",
    "evidence_references": ["EV-001", "EV-002"]
  }
  ```
- **Response**: `200 OK` (`FeedbackRecord`).
- **Endpoint**: `GET /api/agent/feedback`
- **Response**: `200 OK` (`FeedbackListResponse` with aggregate statistics and feedback records).
- **Endpoint**: `GET /api/agent/feedback/{investigation_id}`
- **Response**: `200 OK` (List of feedback records for the specified investigation).

