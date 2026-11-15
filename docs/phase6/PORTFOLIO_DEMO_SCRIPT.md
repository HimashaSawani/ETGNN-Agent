# ETGNN-Agent — Portfolio & Thesis Defense Walkthrough Script

**Project:** Evidence-Grounded Agentic AI for Explainable Bitcoin Fraud Detection  
**Target Audience:** ML Hiring Managers, AI Technical Leads, Academic Examiners  
**Estimated Walkthrough Time:** 5–7 Minutes  

---

### Step 1: System Purpose & Scientific Framing (0:00 – 1:00)
- **What to say:**
  > "ETGNN-Agent is a research-grounded AI engineering platform combining Temporal Graph Neural Networks (ETGNN) with an autonomous LangGraph agent for explainable Bitcoin transaction fraud detection on the Elliptic dataset. Rather than relying on ungrounded LLM generation, the agent enforces mathematical provenance: every factual finding must be backed by an immutable `EvidenceRecord`."
- **What to show:**
  - Open `http://127.0.0.1:8000/`.
  - Highlight the epistemic disclaimer on the header: the system explicitly notes that model outputs represent raw statistical scores, not legal proof or calibrated Bayesian certainty.

---

### Step 2: Multi-Model Forward Pass & Model Disagreement (1:00 – 2:30)
- **What to say:**
  > "A core strength of this system is that it does not treat a single model as ground truth. In Bitcoin transaction analysis, tree-based models like Random Forest and XGBoost evaluate 165 tabular features in isolation, whereas ETGNN evaluates the 2-hop topological structure."
- **What to show:**
  - Select transaction `288717023` (Snapshot 43).
  - Click **Predict**.
  - Show the model consensus panel:
    - ETGNN predicts $0.8850$ (Illicit).
    - XGBoost predicts $0.8200$ (Illicit).
    - Random Forest predicts $0.5000$ (Uncertain).
  - Show the **Model Divergence Detected** banner ($0.385 \ge 0.30$ spread), explaining why graph structure explains the divergence.

---

### Step 3: Interactive Graph Topology & In-Degree Census (2:30 – 3:30)
- **What to say:**
  > "Notice the directed graph topology. Transaction `288717023` has zero incoming edges and two outgoing edges. A common misconception in prior work was describing this as an 'isolated node'. In graph theory, an isolated node has degree zero. Here, the node has outgoing transactions, meaning it is an origin in the observed sample."
- **What to show:**
  - Highlight the directed edges pointing from the target node to its successors.
  - Show the topological qualification banner.

---

### Step 4: Autonomous LangGraph Investigation & Claim Provenance (3:30 – 5:00)
- **What to say:**
  > "Now let's launch the autonomous AI Copilot. The LangGraph workflow executes 8 state nodes: planning, tool execution, evidence consolidation, narrative drafting, claim validation, citation validation, and finalization."
- **What to show:**
  - Click **Investigate Transaction**.
  - Inspect the generated report:
    - Every claim is tagged with an evidence badge like `EV-GNN-PRED-288717023` or `EV-GRAPH-NEIGHBOR-288717023`.
    - Clicking a badge reveals the underlying immutable evidence payload.
    - Show the limitations section: notice the non-causal qualification on Snapshot 43 (AlphaBay closure is observational, not causal).

---

### Step 5: Research Monitoring & Continual Learning Governance (5:00 – 6:30)
- **What to say:**
  > "Finally, let's look at the Research Monitoring Dashboard. In production financial crime detection, concept drift is a critical risk. The dashboard monitors real Kolmogorov-Smirnov feature drift and topological degree shift."
- **What to show:**
  - Click the **Research Monitoring** tab.
  - Review the Snapshot 43 Ground Truth Census: $N=5,063$ nodes, 24 illicit, 1,346 licit, 3,693 unlabelled ($72.94\%$).
  - Point out the **Continual Learning Safety Status: BLOCKED**.
  - Explain why: retraining is blocked to prevent catastrophic forgetting and overfitting to delayed, small-sample labels ($N=24$), and analyst feedback is kept strictly isolated from model weights.

