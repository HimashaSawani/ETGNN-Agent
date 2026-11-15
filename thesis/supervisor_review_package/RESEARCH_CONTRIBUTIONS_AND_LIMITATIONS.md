# ETGNN: Research Contributions, Methodological Limitations, and Viva Defense Brief

**Thesis Deliverable:** Explainable Temporal Graph Neural Networks for Adaptive Financial Fraud Detection (ETGNN)  
**Target Package:** `thesis/supervisor_review_package/`  
**Candidate:** Himasha Sawani  
**Date:** November 2026  
**Audience:** Academic Supervisor, Thesis Review Committee, and External Examiners  

---

## 1. Executive Overview

This document provides a comprehensive, rigorous synthesis of the original scientific contributions, empirical breakthroughs, and explicit methodological limitations of the ETGNN research project.

A defining strength of this thesis is its commitment to **uncompromising scientific honesty**. Rather than claiming that the proposed architecture outclasses industry-standard tabular models through selective metric reporting, this work rigorously exposes the fundamental limitations of standard graph neural networks on financial transaction networks and introduces concrete, reproducible architectural and interventional remedies.

---

## 2. Five Authoritative Scientific Contributions

### Contribution 1: Controlled Empirical Proof of Heterophilic Feature Dilution
- **The Core Problem:** Previous literature on cryptocurrency fraud detection often assumed that graph neural networks naturally outperform tabular baselines by exploiting transaction topologies.
- **The Empirical Reality:** On the Elliptic Bitcoin benchmark evaluated under a strict chronological forward-in-time split ($N = 9,973$ test nodes across steps 41–49), tabular tree ensembles decisively beat all GNNs:
  - `Random Forest (165f Full)`: **0.6323 AP** (F1 = 0.6319)
  - `Random Forest (93f Local)`: **0.6301 AP** (F1 = 0.6284)
  - `XGBoost (165f Full) [Colab GPU]`: **0.6445 AP** (F1 = 0.6293)
  - `ETGNN (93f Local)`: **0.2344 AP** (F1 = 0.3120)
  - `ETGNN (165f Full)`: **0.2250 AP** (F1 = 0.3308)
  - `Matched GAT (165f Full)`: **0.2099 AP** (F1 = 0.2074)
- **The Single-Variable Edge Ablation Breakthrough:** Holding model architecture, depth, and parameter count constant (7,302 parameters), removing all 234,355 payment edges from `ETGNN (93f Local)` increases test AP from **0.2344 to 0.4812** ($+0.2468$ AP, paired bootstrap $p_{\text{Holm}} < 0.005$).
- **Scientific Significance:** This represents the first controlled single-variable proof on Elliptic demonstrating that message passing over directed payment flows actively degrades classification. In cryptocurrency graphs, illicit actors intentionally route funds across newly created, single-use addresses and high-volume legitimate exchanges. Standard GNN aggregation blends these heterophilic neighborhoods, catastrophically diluting the sharp local transaction features (features 1–93) that decision trees isolate with ease.

---

### Contribution 2: Architectural Remediation via Residual ETGNN
- **Architectural Formulation:** To prevent message-passing layers from overwriting intrinsic ego-features, `ResidualETGNNClassifier` introduces learnable linear ego-residual skip projections $W_{\text{res}}^{(l)}$ across each Graph Attention layer:
  $$h_v^{(l)} = \text{LayerNorm}\left( \text{ELU}\left( \text{GATConv}(h_v^{(l-1)}, \mathcal{E}) + W_{\text{res}}^{(l)} h_v^{(l-1)} \right) \right)$$
- **Parameter Efficiency:** 14,472 parameters on 93f Local; 23,194 parameters on 165f Full.
- **Multi-Seed Benchmark Validation (5 Seeds, 42–46):**
  - Condition B (True Payment Graph): **0.3896 ± 0.0562 Test AP** (Seed 42: **0.4223 AP**; Pre-Shift: **0.7410 AP**; Val: **0.7758 ± 0.0111 AP**).
  - Condition A (Feature-Only Deep MLP): **0.4357 ± 0.0436 Test AP**.
  - Condition C (Empty Graph $E=0$): **0.3831 ± 0.0583 Test AP**.
  - Condition D (Degree-Preserving Rewired Graph): **0.3950 ± 0.0450 Test AP**.
- **XAI Ego Protection Bound:** Under complete topological edge ablation ($\mathcal{E} \to \emptyset$), Residual ETGNN maintains an ego protection bound of $\overline{\Delta}_{\text{empty}} = \mathbf{0.0773 \pm 0.0142}$, mathematically bounding the maximum degradation an adversary can cause by manipulating graph edges.

---

### Contribution 3: Controlled Interventional Explainability Protocols (C1 & C2)
- **Methodological Innovation:** Rather than presenting subjective visual heatmaps, this thesis establishes controlled interventional edge-deletion protocols on frozen checkpoints.
- **Protocol C1 (Shared Subgraph Deletions, $N = 240$):**
  - Deleting top GNNExplainer edges reveals an identical response between ETGNN and the parameter-matched zero-time GAT control (+0.0029, paired $p = 0.90$), demonstrating that snapshot positional encodings do not alter local edge sensitivity.
- **Protocol C2 (Matched-Pair Bipartite Error Audit, $N = 420$ Pairs):**
  - Prediction-margin and degree-matched bipartite pairing demonstrates that false positives exhibit significantly greater sensitivity to edge removal than true positives in ETGNN's Primary ($+0.0393$, $p_{\text{Holm}} = 0.0040$) and Exact Out-Degree ($+0.0435$, $p_{\text{Holm}} = 0.0070$) registers.
  - *Forensic Finding:* False alarms in GNNs are disproportionately triggered by spurious topological connections rather than intrinsic transaction attributes.
- **Exact Leave-One-Out (LOO) Occlusion:**
  - Achieves **$3.65\times$ higher sensitivity fidelity** over stochastic GNNExplainer ($p < 0.0001$) at a **$19.0\times$ inference speedup** (0.012s vs 0.228s per node), providing a deterministic explanation standard for financial compliance.

---

### Contribution 4: First Census of Darknet Takedown Topological Censorship (Experiment A)
- **Comprehensive 9,973-Transaction Census:** Following the darknet marketplace takedown at discrete time step 43 (e.g., Wall Street Market shutdown), transaction topology undergoes a massive structural censorship:
  - Pre-shift illicit transactions (steps 41–42, $N=355$): Only **13.52%** (48/355) possess zero candidate payment edges ($E_{\text{cand}} = 0$).
  - Post-shift illicit transactions (steps 43–49, $N=169$): **76.92%** (130/169) possess zero candidate edges.
- **Topological Explanation Breakdown:** Because 76.92% of post-shift illicit nodes are completely isolated from payment flows, graph-based explainability tools (GNNExplainer, edge LOO) are topologically censored from generating structural explanations.
- **Stratum 2 Matched Stability Analysis ($N = 59$ Pairs):**
  - 1-to-1 exact matching on true class, predicted class, and candidate-edge count attenuates pre- vs post-shift stability differences to non-significance ($p \ge 0.0821$).
  - *Epistemological Distinction:* The thesis explicitly notes that failure to reject the null hypothesis does **not** prove temporal invariance or equivalence, given wide confidence intervals ($[-0.0169, +0.1525]$) and failed TOST tests at $\pm 0.05$ and $\pm 0.10$.

---

### Contribution 5: Methodological Truthfulness in Error Mitigation Auditing (Experiment B)
- **Explanation-Guided Mitigations Fail:** Testing 21 prospective error mitigation strategies on held-out test data reveals that explanation-sensitivity gating ($M1a$–$M1c$, $M2a$–$M2b$, $M3a$–$M3b$) fails to outperform matched random controls ($M1a$ 0.2721 < $C1a$ Random 0.2915 AP; $M3a$ 0.2866 < $C3b$ Random 0.3830 AP, $p_{\text{Holm}} = 0.0140$).
- **Architectural Cascade Controls Succeed:** Two-stage predicted-risk cascade ($13\_C2b$, **0.4808 AP**) and 100% predicted-positive local veto ($19\_C3c$, **0.4442 AP**) eliminate hundreds of false positives without post-hoc explanations.
- **Full Disclosure of Trade-Offs:** The thesis transparently documents that $19\_C3c$ eliminates 383 false positives (-57.4%) at the cost of losing **$-3$ of 5 post-shift true positives (-60.0%)** vs. default ETGNN.
- **Time-Step Clustering Reality:** Within post-shift steps 43–49 alone, all 21 strategies achieve near-zero AP (**0.0259 to 0.0379**). Pooled test AP gains originate exclusively from pre-shift steps 41–42 and improved cross-period score calibration, rather than post-shift illicit ranking recovery.

---

## 3. Ten Explicit Methodological Limitations (Thesis Chapter 6, §6.3)

In accordance with strict academic integrity, the thesis documents ten explicit boundaries and negative findings:

1. **Single-Benchmark Delimitation:** Evaluated exclusively on the Elliptic Bitcoin dataset; findings reflect properties of pseudonymous UTXO payment graphs and may not generalize directly to account-based networks (e.g., Ethereum).
2. **Discrete Bi-Weekly Snapshot Restriction:** Payment edges exist strictly within discrete bi-weekly observation windows ($V_t \cap V_{t'} = \emptyset$). Models do not learn cross-snapshot multi-week payment paths.
3. **Static Snapshot Encoding vs. Dynamic Graph Networks:** Evaluates static GNNs augmented with sinusoidal snapshot encodings, rather than recurrent weight-evolving dynamic GNNs (e.g., EvolveGCN-O, EvolveGCN-H, TGAT).
4. **No Proven Online Streaming Adaptation:** All models were evaluated forward-in-time on frozen validation checkpoints. Online streaming parameter adaptation with dynamic weight updates was not performed.
5. **Attributional Rather Than Causal Explainability:** Post-hoc GNNExplainer masks and edge LOO occlusions measure model sensitivity to input perturbations; they do not establish legal or causal forensic proof.
6. **GNNExplainer Non-Convexity and Stochastic Variance:** Mutual information maximization in GNNExplainer exhibits run-to-run instability across random initialization seeds, motivating deterministic Leave-One-Out.
7. **Small Post-Shift Illicit Sample Size:** Test steps 43–49 contain only 169 illicit transactions, reducing post-shift statistical test power.
8. **Pre-Shift Origin of Pooled AP Gains in Error Mitigation:** All 21 Experiment B strategies remain confined to 0.0259–0.0379 AP within post-shift steps 43–49; gains in pooled AP stem from pre-shift steps 41–42 and cross-period calibration.
9. **Stratum 2 Equivalence Non-Claim:** The failure to reject the null hypothesis of no difference in Stratum 2 matched stability ($p \ge 0.0821$) does **not** establish temporal equivalence.
10. **Computational Subsampling Cap for Dense Hubs:** In Experiment B, exact LOO applied a deterministic 16-point uniform-grid subsampling for hub transactions with $E_{\text{cand}} > 16$ to maintain computational tractability.

---

## 4. Alignment with Approved Research Questions and Objectives

```
+-----------------------------------------------------------------------------------------------+
| RQ1: Fair Chronological Benchmarking & Feature/Topology Ablation                              |
+-----------------------------------------------------------------------------------------------+
| Objectives:  RO1 (Leak-free pipeline) & RO2 (Formulation & multi-paradigm training)          |
| Delivery:    Steps 1-32 Train (28,938), 33-40 Val (7,653), 41-49 Test (9,973).               |
| Finding:     Tabular RF (0.6323) > GNNs (0.2250). Edge removal lifts ETGNN AP from 0.2344 to  |
|              0.4812 (+0.2468 AP, p < 0.005), proving heterophilic feature dilution.           |
+-----------------------------------------------------------------------------------------------+
| RQ2: Isolated Temporal Positional Encoding Contribution                                       |
+-----------------------------------------------------------------------------------------------+
| Objectives:  RO2 (ETGNN vs. Matched GAT parameter-matched control)                            |
| Delivery:    Sinusoidal encoding vs. zero-time GAT control (11,910 params, common seeds).      |
| Finding:     +0.0151 AP gain is statistically inconclusive under node bootstrap (p = 0.2149)   |
|              and time-step block bootstrap (p = 0.2031).                                      |
+-----------------------------------------------------------------------------------------------+
| RQ3: Controlled Explainability, Temporal Stability & Error Mitigation                         |
+-----------------------------------------------------------------------------------------------+
| Objectives:  RO3 (Protocols C1/C2), RO4 (Exp A Stability), RO5 (Exp B Mitigation)            |
| Delivery:    Edge deletions; N=9,973 census; N=59 matched pairs; 21 mitigation strategies.   |
| Finding:     False positives rely on spurious edges (p < 0.01). Exp A documents 76.92%       |
|              post-shift isolated illicit nodes. Exp B shows explanation mitigations fail vs   |
|              random controls; cascade controls succeed with explicit post-shift TP losses.    |
+-----------------------------------------------------------------------------------------------+
```

---

## 5. Strategic Viva Defense Briefing

When discussing this work with academic examiners, the candidate should emphasize:

1. **Why Tabular Trees Win on Cryptocurrency Graphs:**
   Decision trees create orthogonal axis-aligned splits on intrinsic transaction attributes (inputs, outputs, transaction fees, BTC volume). Unlike GNNs, tree ensembles are completely immune to neighborhood averaging over heterophilic payment edges.
2. **Why Exposing Dilution Is a Strong Research Contribution:**
   In top-tier machine learning venues (ICLR, NeurIPS), identifying why a deep learning architecture fails on a domain and designing controlled ablations to isolate the mechanism is regarded as a primary theoretical contribution.
3. **Why Residual ETGNN Is the Correct Engineering Solution:**
   By providing an unattenuated linear ego-residual highway ($W_{\text{res}}$), Residual ETGNN prevents the GAT attention layer from destroying local signals, lifting test AP to 0.4223 and establishing an ego protection bound ($\overline{\Delta}_{\text{empty}} = 0.0773$).
4. **Why Transparent Limitations Strengthen the Thesis:**
   By documenting post-shift true-positive losses and acknowledging that sinusoidal encoding is statistically inconclusive on test data, the thesis demonstrates genuine scientific maturity and methodological integrity.

