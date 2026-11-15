# Phase 4 Integration Contract: Frontend Investigation Drawer & Live Telemetry

## 1. Overview & Scope

Phase 4 will build the user-facing investigation interface inside the React/Vite dashboard. This contract establishes the exact data models, API endpoints, WebSocket/SSE interfaces, and UI expectations required between the Phase 3 backend and Phase 4 frontend.

**Constraint Reminder**: Do NOT begin Phase 4 frontend development until Phase 3 readiness is signed off.

---

## 2. Component Architecture Contract

```
+-------------------------------------------------------------+
|                  React Dashboard (App.tsx)                  |
|                                                             |
|   +-----------------------+     +-----------------------+   |
|   | 2D Ego-Net Graph View |     | Tabular Baseline View |   |
|   +-----------------------+     +-----------------------+   |
|                                                             |
|   +-----------------------------------------------------+   |
|   |  Phase 4 Component: <InvestigationDrawer />         |   |
|   |                                                     |   |
|   |  1. Investigation Query Input & Mode Selector       |   |
|   |  2. Live Telemetry Event Stepper                    |   |
|   |  3. Model Consensus Pill (GNN vs Tabular Spread)    |   |
|   |  4. Forensic Findings List with Citation Badges     |   |
|   |  5. Audit Limitations & GNNExplainer Disclaimer     |   |
|   +-----------------------------------------------------+   |
+-------------------------------------------------------------+
```

---

## 3. UI State Model & Data Types

The Phase 4 TypeScript frontend must declare the following types matching Phase 3 schemas:

```typescript
export interface SupportedFinding {
  finding: string;
  evidence_id: string;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface ConsensusVerdict {
  verdict: 'HIGH_RISK_ILLICIT' | 'LOW_RISK_LICIT' | 'SUSPICIOUS_HIGH_PROB' | 'INCONCLUSIVE_DATA_GAP';
  risk_score: number;
  model_divergence_detected: boolean;
  probability_spread: number;
}

export interface ForensicReport {
  investigation_id: string;
  target_transaction_id: number;
  snapshot_id: number;
  status: 'COMPLETED' | 'ABSTAINED' | 'FAILED';
  consensus_verdict: string;
  narrative_summary: string;
  supported_findings: SupportedFinding[];
  uncertain_findings: string[];
  limitations: string[];
  audit_hash: string;
  generated_at: string;
}

export interface TelemetryEvent {
  event_id: string;
  investigation_id: string;
  timestamp: string;
  event_type: string;
  status: 'PENDING' | 'RUNNING' | 'SUCCESS' | 'FAILED' | 'SKIPPED';
  tool_name?: string;
  duration_ms: number;
  safe_output_summary?: Record<string, any>;
  message?: string;
}
```

---

## 4. Interaction Workflow

1. **Triggering an Investigation**:
   - The user selects a transaction in the 2D network graph or enters a `txId` manually in the search bar.
   - The user clicks `"Investigate with Agent"`.
   - The frontend issues `POST /api/agent/investigate`.
2. **Displaying Live Telemetry**:
   - The drawer opens immediately in a loading state.
   - The frontend polls `GET /api/agent/investigations/{id}/events` every 500ms (or streams events) to render step-by-step progress (`Validating Request` $\to$ `Planning Investigation` $\to$ `Executing GNN Models` $\to$ `Checking Citations`).
3. **Rendering the Forensic Report**:
   - Upon completion, the drawer renders the `ForensicReport`.
   - Every `supported_finding` displays an interactive clickable citation pill (`[EV-PRED-...]`). Clicking the pill highlights the corresponding node or edge in the graph view.
   - If `model_divergence_detected == true`, an amber warning alert highlights the probability spread between GNN and Random Forest.

