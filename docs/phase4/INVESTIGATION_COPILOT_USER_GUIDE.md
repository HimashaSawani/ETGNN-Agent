# ETGNN-Agent — Phase 4: Investigation Copilot User Guide
## Forensic Analyst Handbook & Operational Runbook

**Audience:** Financial Crime Investigators, AML Compliance Officers, Research Analysts  
**Platform Version:** ETGNN-Agent v3.5.0  

---

## 1. Quick Start Guide

### Launching an AI Investigation
You can launch an AI Forensic Investigation from anywhere in the application:
1. **Via Header**: Click the **"Investigate with AI"** button in the top navigation bar.
2. **Via Transaction Explorer**: Locate any transaction row in the table and click the blue **"AI"** pill button on the right edge.
3. **Via Prediction Arena**: After inspecting comparative model probabilities, click **"Investigate with Copilot"**.

---

## 2. Drawer Controls & Parameters

| Control | Options | Description |
| :--- | :--- | :--- |
| **Transaction ID** | Numeric (e.g., `230425980`, `288717023`) | The target Bitcoin transaction to audit. |
| **Investigation Mode** | `Retrospective (All)` / `Point-in-Time (Strict)` | Controls temporal leakage boundaries. Strict mode prevents retrieving transactions or topological paths that occurred temporally after the target snapshot. |
| **Provider** | `MOCK` / `EVIDENCE_ONLY` / `Gemini` / `Claude` / `Ollama` | Selects LLM synthesis engine. MOCK provides fast, deterministic, reproducible offline responses with zero external API dependencies. |
| **Focus Query** | Text input | Custom investigative hypothesis (e.g., *"Assess counterparty illicit taint and fan-out risk"*). |

---

## 3. Interpreting the Forensic Report

### Multi-Model Comparison Card
- **Canonical ETGNN**: Temporal Graph Neural Network accounting for transaction graphs and dynamic node features.
- **Random Forest**: Tabular baseline utilizing local transaction features.
- **XGBoost**: Gradient-boosted decision tree baseline.
- **Probability Spread**: $\max(p) - \min(p)$.
- **Model Divergence Alert**: Triggered when spread $\ge 0.30$. Advises analysts to scrutinize feature distributions vs topological graph structures.

### Graph Evidence & Snapshot 43 Qualifications
- **Snapshot 43 Alert**: Snapshot 43 in the Elliptic Bitcoin dataset represents an abrupt structural regime change due to the darknet market shutdown. When inspecting transactions from snapshot 43, the Copilot surfaces a warning explaining potential topological disruption.
- **Isolated Node Disclaimers**: If in-degree $= 0$ and out-degree $= 0$, the system highlights that GNN predictions rely purely on local features without graph message passing.

### Citation Pills & Evidence Trace
- Click any yellow citation pill (e.g., `[EV-001]`, `[Weber et al., 2019]`) to inspect the verified underlying cryptographic evidence record.

---

## 4. Submitting Human Feedback (HITL)

1. Open the **"Analyst Feedback"** tab in the drawer.
2. Select an assessment category (`Helpful Explanation`, `Unhelpful`, `Citation Issue`, `Missing Evidence`, `Incorrect Numerical Statement`, `Needs Further Review`).
3. Provide a star rating (1 to 5).
4. Enter qualitative notes in the review comment box.
5. Click **"Submit Audit Feedback"**.
6. The record is permanently logged in `data/analyst_feedback.db` for continual learning queueing in Phase 5.

