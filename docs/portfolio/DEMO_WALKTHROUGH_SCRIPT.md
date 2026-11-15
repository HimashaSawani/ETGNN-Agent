# ETGNN-Agent Portfolio Demonstration & Viva Defense Walkthrough Script

**Duration:** ~2.5 to 3.0 Minutes  
**Target Audience:** AI Engineering Hiring Managers, Technical Evaluators, University Viva Defense Committee  
**Target Transaction:** Held-Out Chronological Test Split — `Tx 94340906` (Snapshot 43, Ground Truth: Illicit)

---

## 1. Video & Live Demo Flow

### 0:00 - 0:30 | Introduction & Scientific Research Problem
- **Visual**: Open `http://127.0.0.1:8000/` (Overview Tab).
- **Spoken Script**:
  > *"Welcome. This is the ETGNN Research Demonstration Platform, an end-to-end AI system exploring Explainable Temporal Graph Neural Networks and Autonomous Agentic AI for cryptocurrency fraud detection on the Elliptic Bitcoin dataset ($203,769$ transactions).*
  >
  > *A core contribution of our empirical research was discovering that under standard benchmark evaluations, traditional GNNs actually underperform tree-based ensembles (Random Forest achieves 0.6323 AUPRC versus 0.2250 for standard ETGNN) due to heterophilic neighborhood mixing and directed edge starvation. To bridge this gap, we engineered ego-residual GNN architectures and built an autonomous LangGraph agent that cross-validates GNN predictions against tabular baselines and graph topology."*

---

### 0:30 - 1:15 | Model Comparison Arena & Real Inference
- **Visual**: Click on the **Model Predictions** tab. Enter `94340906` in the inspect input and click **Refresh**.
- **Key Callouts on Screen**:
  - Point to **Ground Truth Label**: `Illicit (1)`.
  - Point to **Prediction Cards**:
    - `Canonical ETGNN`: ~0.42 illicit probability.
    - `Residual ETGNN`: ~0.55 illicit probability.
    - `Random Forest`: ~0.64 illicit probability.
- **Spoken Script**:
  > *"Here we see live forward-pass inference on held-out test transaction 94340906 from Snapshot 43. Notice the divergence: standard ETGNN dilutes illicit signal because in this snapshot, 58.33% of illicit nodes have zero incoming directed edges. Our Residual ETGNN preserves local ego-features through residual skip projections, aligning closely with Random Forest."*

---

### 1:15 - 2:00 | AI Investigation Copilot (Autonomous LangGraph Workflow)
- **Visual**: Click **Investigate with AI** button. The slide-over drawer opens.
- **Actions in Drawer**:
  - Show Tx ID is synced to `94340906`.
  - Select Provider: `EVIDENCE_ONLY (Factual Non-Generative)`.
  - Click **Launch Investigation**.
  - Show the live workflow stepper transitions: `RETRIEVING_CONTEXT` -> `MODEL_INFERENCE` -> `HYPOTHESIS_EVALUATION` -> `REPORT_SYNTHESIS` -> `VALIDATING` -> `COMPLETED`.
- **Spoken Script**:
  > *"Now let's launch the AI Investigation Copilot. The orchestrator runs an official LangGraph state machine. It retrieves temporal subgraph context, evaluates multi-model consensus, and runs hypothesis testing. Notice our fail-closed design: when no cloud LLM API keys are provided, it operates deterministically in EVIDENCE_ONLY mode without inventing hallucinated explanations."*

---

### 2:00 - 2:30 | Evidence Validation & Scientific Guardrails
- **Visual**: Scroll down the generated **Forensic Investigation Dossier**.
- **Key Callouts on Screen**:
  - **Supported Findings & Citations**: Click an evidence citation anchor (e.g. `[EV-PRED-ETGNN-1]`), showing smooth highlight of the source record in the Evidence tab.
  - **Snapshot 43 Observation**: Highlight the limitation text stating that structural distribution shifts around Snapshot 43 cannot be causally proven from anonymized benchmark data alone.
  - **Epistemic Notice**: Point out the statutory academic research disclaimer.
- **Spoken Script**:
  > *"Every claim in this dossier is audited by a three-tier validation engine: ClaimValidator checks numerical tolerances, CitationValidator ensures zero hallucinated citations, and ReportValidator enforces topological qualifications. For instance, it automatically mandates a non-causal disclaimer on Snapshot 43 rather than allowing speculative darknet takedown claims."*

---

### 2:30 - 3:00 | Analyst Review Loop & Full End-to-End Testing
- **Visual**: Click **Analyst Review** tab, select a rating, enter brief notes, and click **Submit Feedback**. Then switch to terminal showing `184 backend tests passed` and `14 Playwright E2E tests passed`.
- **Spoken Script**:
  > *"Human analysts can review findings, assign confidence ratings, and persist feedback to SQLite for continual auditability. The entire platform is supported by 184 automated backend regression tests and 14 Playwright browser E2E tests, verifying real browser-to-backend model inference.*
  >
  > *This demonstrates a complete, scientifically grounded AI engineering system combining modern Deep Learning, Explainable AI, LangGraph multi-agent orchestration, and production-grade full-stack engineering. Thank you."*

---

## 2. Accompanying Artifacts

- **Full-Page Live Investigation Screenshot**:  
  [`web/frontend/playwright-report/live_real_backend_tx94340906.png`](file:///d:/ETGNN-Fraud-Detection/web/frontend/playwright-report/live_real_backend_tx94340906.png)
- **Live Playwright Execution Trace**:  
  `web/frontend/test-results/live_real_backend-Real-Fas-433a0-itcoin-transaction-94340906-chromium/trace.zip`
- **Audit Verification Report**:  
  [`docs/phase6_8/PHASE_6_8_VERIFICATION_REPORT.md`](file:///d:/ETGNN-Fraud-Detection/docs/phase6_8/PHASE_6_8_VERIFICATION_REPORT.md)

