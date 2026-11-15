# ETGNN Web Research Platform — Methodological & Operational Limitations

## 1. Research & Methodological Limitations

### 1.1. Static Snapshot Architecture (Non-Dynamic GNN)
- **Model Nature**: Despite the project designation *Explainable Temporal Graph Neural Networks*, ETGNN is implemented as a static 2-layer Graph Attention Network (GAT) with sinusoidal time-step encodings ($D_t=16$) concatenated to input features.
- **Absence of Temporal Memory**: The model does not maintain an evolving node state across discrete time steps (unlike continuous dynamic frameworks such as Temporal Graph Networks (TGN), DyRep, or JODIE).
- **Time Encoding Neutrality**: In controlled multi-seed repeat ablations (Seeds 7, 42, 123), ETGNN and Matched GAT (zeroed time vectors) performed identically (~0.551 vs ~0.550 validation AUPRC), demonstrating that sinusoidal time encodings provided no dynamic temporal adaptation.

### 1.2. The Elliptic Bitcoin Benchmark Constraints
- **Discrete Snapshot Disconnection**: In the Elliptic dataset, all graph edges exist strictly within the same 2-week time step. There are zero inter-step transaction edges, preventing temporal message passing across time boundaries.
- **Obfuscated Features**: All 165 features (93 local, 72 aggregate) are anonymized and standardized. Transaction amounts, wallet addresses, and gas/fee metrics are withheld, limiting contextual interpretability.
- **Extreme Class Imbalance & Prevalence Volatility**: Labelled illicit transactions constitute only ~10% of labelled nodes overall, and prevalence swings violently between 0.7% and 25.1% across test steps 41–49.

### 1.3. Tabular Baseline Dominance & Topological Noise
- **Empirical Superiority of Decision Trees**: On held-out chronological test steps 41–49, Random Forest (0.6323 AUPRC) and XGBoost (0.5472 AUPRC) dramatically outperformed ETGNN (0.2250 AUPRC) and all standard GNN baselines.
- **Topological Noise**: In transaction graphs without edge type filtering, raw payment edges mix illicit flows with licit exchange wallets. Ablation experiments showed that deleting all graph edges increased validation AUPRC by **+0.2468** (from 0.5401 to 0.7869).

### 1.4. Explainability & Causal Boundaries
- **Sensitivity vs Causality**: GNNExplainer masks measure model prediction drop under edge deletion. They quantify mathematical gradient sensitivity under intervention, not empirical evidence of criminal behavior.
- **C1 Inconclusive**: Hypothesis C1 (temporal explanation sensitivity) was statistically inconclusive as 95% confidence intervals spanned zero across all cohorts.
- **C2 Scope**: C2 evaluates prediction sensitivity under matched confidence and degree, not causal explanation correctness.

---

## 2. Platform Operational & Engineering Boundaries

### 2.1. In-Browser Graph Scope Limits
- **Neighborhood Capping**: The browser visualizer restricts extracted subgraphs to $k=2$ hops and a maximum of 150 nodes. Rendering full time-step graphs (which contain up to 5,000+ nodes) in client-side Canvas/SVG degrades frame rates and impairs visual intelligibility.
- **CPU Execution**: Inference is executed on the host CPU with `torch.set_num_threads(2)`. Large batch interventions should not be triggered concurrently under heavy server load.

### 2.2. Legal & Regulatory Non-Enforcement Notice
- **Academic Demonstration Only**: This web platform was created exclusively for academic research demonstration, peer review, and thesis examination.
- **No Operational Reliance**: Under no circumstances should financial institutions, cryptocurrency exchanges, law enforcement agencies, or compliance officers rely upon predictions or explanations from this platform for automated transaction blocking, suspicious activity report (SAR) filing, or asset seizure.
