# ETGNN-Agent System Architecture Specification

## 1. High-Level Architectural Topology

```mermaid
flowchart TD
    subgraph Frontend["React 19 Presentation Layer (SPA)"]
        UI_Main["App.tsx Shell"]
        UI_Arena["Prediction Arena"]
        UI_Graph["Interactive Graph Viewer"]
        UI_Drawer["AI Investigation Copilot Drawer"]
        UI_Stepper["Workflow Stepper"]
        UI_Report["Forensic Report Viewer"]
        UI_Feedback["Analyst Feedback Form"]
    end

    subgraph API["FastAPI Transport & Gateway (Port 8000)"]
        EP_Health["GET /api/health"]
        EP_Tx["GET /api/transactions/{id}"]
        EP_Subgraph["GET /api/transactions/{id}/subgraph"]
        EP_Predict["POST /api/predict"]
        EP_Investigate["POST /api/agent/investigate"]
        EP_Events["GET /api/agent/investigations/{id}/events"]
        EP_Feedback["POST /api/agent/feedback"]
    end

    subgraph LangGraph["LangGraph Multi-Agent Orchestrator"]
        StateGraph["StateGraph: InvestigationState"]
        Node_Retrieve["Node 1: Graph & Context Retrieval"]
        Node_Infer["Node 2: Multi-Model Inference Consensus"]
        Node_Eval["Node 3: Hypothesis & Attribution Evaluation"]
        Node_Report["Node 4: Report Synthesis & Epistemic Guardrails"]
        Node_Validate["Node 5: Strict Evidence Validation Gate"]
        Node_Fallback{"Validation Passed?"}
        Node_Degrade["Sanitize / Degrade to Uncertain"]
    end

    subgraph RAG["Temporal Graph-RAG Engine"]
        DocLoader["Document Loader (Allowlist Enforced)"]
        VectorIndex["TF-IDF / Cosine Vector Index"]
        TemporalFilter["Temporal Containment (As-Of Snapshot Filter)"]
        SubgraphSampler["2-Hop Directed Subgraph Extractor"]
    end

    subgraph ML["Machine Learning Inference Engine"]
        M_ETGNN["Canonical ETGNN (PyG)"]
        M_Residual["Residual ETGNN (Ego-Skip GNN)"]
        M_GAT["Matched GAT (Ablation)"]
        M_RF["Random Forest (165f Top Performer)"]
        M_XGB["XGBoost (165f)"]
        XAI_Explainer["GNNExplainer (Sensitivity Attribution)"]
        XAI_LOO["Exact LOO Occlusion (C1/C2 Validation)"]
    end

    subgraph Validation["Evidence Grounding & Validation Gate"]
        Val_Claim["ClaimValidator (Numerical & Factual Drift)"]
        Val_Cit["CitationValidator (Provenance Chain Matching)"]
        Val_Rep["ReportValidator (Topological & S43 Disclaimers)"]
    end

    subgraph Storage["Persistence & Data Layer"]
        DB_Feedback[("SQLite: data/analyst_feedback.db")]
        Data_Elliptic[("Elliptic Raw & Processed CSV/PT")]
        Docs_Research[("Curated Research Markdown Index")]
        Results_Store[("results/benchmarks/*.json")]
    end

    %% Wiring
    UI_Main --> UI_Arena & UI_Graph & UI_Drawer
    UI_Drawer --> UI_Stepper & UI_Report & UI_Feedback
    
    UI_Arena --> EP_Tx & EP_Predict
    UI_Graph --> EP_Subgraph
    UI_Drawer --> EP_Investigate & EP_Events & EP_Feedback

    EP_Investigate --> LangGraph
    EP_Tx & EP_Predict --> ML
    EP_Subgraph --> SubgraphSampler

    LangGraph --> Node_Retrieve --> Node_Infer --> Node_Eval --> Node_Report --> Node_Validate
    Node_Validate --> Node_Fallback
    Node_Fallback -- No --> Node_Degrade --> Storage
    Node_Fallback -- Yes --> Storage

    Node_Retrieve --> RAG
    RAG --> DocLoader & VectorIndex & TemporalFilter & SubgraphSampler
    DocLoader --> Docs_Research
    SubgraphSampler --> Data_Elliptic

    Node_Infer --> ML
    M_ETGNN & M_Residual & M_GAT & M_RF & M_XGB --> Data_Elliptic

    Node_Validate --> Validation
    Validation --> Val_Claim & Val_Cit & Val_Rep

    EP_Feedback --> DB_Feedback
```

---

## 2. Component Design & Inter-Module Communication

### 2.1 React 19 Frontend Layer (`web/frontend/`)
- **Technology Stack**: React 19, TypeScript ~6.0, Vite 8, Lucide-React.
- **Key Modules**:
  - `PredictionArena`: Provides side-by-side forward-pass prediction comparisons across GNNs and tree models with interactive feature override capabilities.
  - `InvestigationCopilotDrawer`: Slide-over drawer orchestrating autonomous investigation lifecycles, streaming workflow stepper status, model agreement panels, and evidence citation explorer.
  - `InteractiveGraphEvidenceViewer`: Renders candidate transaction subgraphs, highlighting directed edge flow, degree characteristics, and topological qualifications.

### 2.2 FastAPI Gateway (`src/web/app.py`)
- **Architecture**: Asynchronous ASGI application with singleton model and dataset caching (`DataLoader`).
- **Safety Boundaries**:
  - Enforces CORS policies and input validation via Pydantic schemas.
  - Serves static assets for the React SPA at `/` with clean client-side routing fallback.
  - Integrates SSE streaming telemetry for agent execution stages.

### 2.3 LangGraph Autonomous Agent (`src/agent/`)
- **Workflow State Machine**: `StateGraph` cycling through deterministic stages:
  1. `retrieve_context`: Fetches transaction metadata, directed subgraph, and research documents up to the evaluation snapshot.
  2. `model_inference`: Executes forward passes across PyG and scikit-learn models; calculates consensus and probability spread.
  3. `evaluate_hypotheses`: Compares model agreement and evaluates structured fraud hypotheses against topological evidence.
  4. `generate_report`: Compiles narrative summary, supported findings, and limitation disclosures.
  5. `validate_report`: Passes all synthesized claims and citations through the three-tier validation gate. If invalid, sanitizes and degrades claims to uncertain findings.

### 2.4 Evidence Grounding Gate (`src/agent/validation/`)
- **`ClaimValidator`**: Flags numerical drift between report claims and underlying prediction probabilities (strict tolerance $\pm 0.01$ for probabilities, $\pm 0.001$ for benchmark metrics).
- **`CitationValidator`**: Enforces strict provenance; every cited evidence ID must exist in the validated `provenance_chain`.
- **`ReportValidator`**: Enforces scientific integrity policies:
  - Distinguishes missing topology from verified zero-degree topology.
  - Enforces mandatory non-causal historical observation for Snapshot 43 transactions.
  - Requires sensitivity disclaimers on GNNExplainer attributions.

---

## 3. Data Flow & Security Containment

1. **Temporal Isolation**: No document, transaction, or edge timestamped greater than the target transaction's snapshot $t$ is accessible during inference or RAG retrieval.
2. **Fail-Closed Fallback**: In the absence of external LLM credentials (Google Gemini / Anthropic Claude), the platform automatically falls back to `EVIDENCE_ONLY` mode, generating factual reports directly from verified evidence records without hallucinated narratives.
3. **Auditability**: Every generated investigation dossier is signed with a deterministic SHA256 provenance hash linking findings to exact underlying `EvidenceRecord` IDs.

