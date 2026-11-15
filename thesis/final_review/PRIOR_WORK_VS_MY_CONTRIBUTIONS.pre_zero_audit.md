# Prior Work vs. My Research Contributions: Academic Comparison Matrix

**Document ID:** `thesis/final_review/PRIOR_WORK_VS_MY_CONTRIBUTIONS.md`  
**Date:** October 30, 2026  
**Project:** `D:\ETGNN-Fraud-Detection`  
**Official Approved Title:** *Explainable Temporal Graph Neural Networks for Adaptive Financial Fraud Detection*  
**Auditor Designation:** Senior Academic Thesis Editor & Literature Integrity Reviewer  

---

## 1. Academic Grounding & Literature Positioning

To demonstrate rigorous scholarship, an academic thesis must transparently position its methods and findings relative to published literature. This matrix maps the core foundational papers in cryptocurrency forensics, graph neural networks, temporal modeling, and explainability against the specific methods, controls, and findings of this research.

---

## 2. Systematic Academic Prior Work Comparison Matrix

| Key Study & Citation | Original Domain & Core Contribution | Stated Findings & Claims in Prior Work | Known Limitations in Prior Work | What THIS Thesis Does Differently (Originality & Critical Extension) | Novelty Classification |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Weber et al. (2019)**<br>*arXiv / KDD Workshop* | Released the foundational Elliptic Bitcoin transaction benchmark (203k nodes, 234k edges, 49 steps). | Observed that Random Forest (0.65 F1) outperformed standard GCN (0.54 F1) on predicting illicit transactions. | Used fixed F1 at $\tau = 0.50$ (susceptible to class imbalance calibration issues); did not investigate *why* GNNs underperformed or test edge-free ablations. | Re-evaluates under gold-standard **AUPRC** (RF 0.6323 vs. GCN 0.1306, ETGNN 0.2250); executes a controlled edge-removal ablation demonstrating that removing edges increases AP by $+0.2468$, revealing precomputed 1-hop feature redundancy. | **Empirical Deepening & Diagnostic Falsification** |
| **Veličković et al. (2018)**<br>*ICLR 2018* | Formalized Graph Attention Networks (GAT) using multi-head self-attention. | Self-attention weights neighborhood messages dynamically, improving over static GCNs on citation benchmarks. | Evaluated primarily on static homophilic benchmarks (Cora, Citeseer); assumes connected neighbors share labels. | Applies GAT to highly heterophilic Bitcoin payment flows; demonstrates that standard GAT suffers severe precision collapse (0.0893 precision, 0.1155 AUPRC) due to illicit mixing across licit hub nodes. | **Domain Stress-Testing & Failure Analysis** |
| **Vaswani et al. (2017)**<br>*NeurIPS 2017* | Introduced Sinusoidal Positional Encoding in Transformer sequence models. | Uses fixed geometric sine/cosine frequencies to encode token position without learnable parameters. | Applied to ordered 1D sequence tokens; not designed for discrete snapshot graph indices. | Adapts 16-d sinusoidal encoding to discrete graph snapshot indices ($t \in \{1..49\}$); evaluates against a parameter-matched zero-time GAT control across 5 seeds, proving that sinusoidal encoding confers no statistically significant advantage ($p > 0.05$). | **Negative Result & Control Isolation** |
| **Ying et al. (2019)**<br>*NeurIPS 2019* | Formalized `GNNExplainer` for post-hoc GNN node/edge explanation. | Optimizes continuous edge masks $\sigma(\mathbf{m}) \in (0, 1)$ via mutual information and entropy regularization. | Evaluated on synthetic motifs (BA-Shapes) and small molecular graphs; does not benchmark against discrete Leave-One-Out oracles on residual architectures. | Evaluates GNNExplainer on residual GNNs in cryptocurrency AML; proves that continuous mask optimization yields near-uniform masks that perform **no better than random edge deletion ($p = 0.6827$)**, while an Exact LOO Oracle achieves $3.65\times$ higher fidelity. | **Methodological Demarcation & Falsification** |
| **Amara et al. (2022)**<br>*NeurIPS 2022 (GraphFramEx)* | Established a standardized benchmarking framework for GNN explanation evaluation. | Proposed fidelity, sparsity, and stability metrics; highlighted high variance and unreliability across explainer methods. | Focused on standard benchmark datasets; did not evaluate high-degree heterophilic transaction graphs or error registers. | Adapts GraphFramEx fidelity metrics to cryptocurrency AML; introduces Contrast 2 margin-matched correct-vs-incorrect prediction registers and candidate-edge ceiling budget analysis. | **Methodological Extension** |
| **Pareja et al. (2020)**<br>*AAAI 2020 (EvolveGCN)* | Proposed EvolveGCN, using recurrent networks (LSTM/GRU) to evolve GCN weight matrices across snapshot sequences. | Captures dynamic topological evolution across discrete graph snapshots without requiring node identity persistence. | Computationally intensive; evaluated on small snapshots; requires significant GPU memory for RNN unrolling. | Contrasts the snapshot time-encoding approach with recurrent weight evolution; demonstrates that static snapshot encoding fails to capture temporal concept drift (e.g. AlphaBay takedown). | **Comparative Architecture Synthesis** |
| **Rossi et al. (2020)**<br>*ICML Workshop (TGN)* | Proposed Temporal Graph Networks (TGN) for continuous-time dynamic graphs. | Maintains streaming continuous-time memory modules updated at microsecond event timestamps. | Requires microsecond transaction stream timestamps; incompatible with discrete 2-week bi-weekly snapshot datasets like Elliptic1. | Explicitly delimits research scope: explains why continuous-time memory networks cannot be directly deployed on discrete bi-weekly snapshots and formulates TGN as a future work priority. | **Epistemic Delimitation & Future Roadmap** |
| **Dou et al. (2020)**<br>*CIKM 2020 (CARE-GNN)* | Proposed Camouflage-Resistant GNN using reinforcement learning to filter deceptive fraud edges. | Selects relevant neighbors via adaptive thresholds, mitigating camouflage in fraud networks. | Focuses on multi-relation review/ecommerce graphs (YelpChi); requires complex RL policy training. | Addresses financial heterophily via an architectural ego-residual skip connection ($W_{\text{res}}h_v$) rather than complex RL policy optimization, establishing an empirical noise bound of $\bar{\Delta}_{\text{empty}} = 0.0773$. | **Alternative Architectural Solution** |
| **Liu et al. (2021)**<br>*WWW 2021 (PC-GNN)* | Proposed Pick-and-Choose GNN for imbalanced fraud graphs. | Performs class-balanced sub-sampling of neighbor nodes and edges to combat extreme class imbalance. | Re-samples graph topology during training; computationally expensive on large-scale transaction DAGs. | Evaluates loss-weighting and degree-preserving rewired null graphs; proves that ego-residual connections achieve optimization stabilization without altering raw graph topology. | **Methodological Comparison** |
| **Rao et al. (2021)**<br>*KDD 2021 (xFraud)* | Proposed xFraud, an explainable fraud detection framework on heterogeneous graphs. | Incorporates self-attention and targeted edge masking to produce interpretable subgraphs for compliance analysts. | Evaluates on e-commerce transaction graphs; presents generated subgraphs as self-evident explanations without leave-one-out validation. | Subjected post-hoc explainers to adversarial null tests: evaluates GNNExplainer masks against random edge deletion and exact LOO oracles, rejecting claims of causal forensic explanation. | **Critical Methodological Refinement** |
| **Elmougy & Liu (2023)**<br>*KDD 2023 (Elliptic++)* | Released Elliptic++, mapping Elliptic1 transactions to 822k wallet addresses with 56 engineered features. | Claimed external benchmark superiority and multi-aspect graph utility for Bitcoin forensics. | Did not audit feature construction for rolling point-in-time constraints; did not report unseen-actor evaluation. | Forensically inspects raw feature vectors and **discovers static global lifetime feature leakage** (`last_block_appeared_in`, `total_txs`); implements an `unseen_entity_mask` and documents validation-to-test AUPRC collapse ($0.45 \to 0.09$). | **Critical Forensic Audit & Benchmark Discovery** |

---

## 3. Clear Demarcation of Student Work vs. Prior Art

```
┌────────────────────────────────────────────────────────┐
│               FOUNDATIONAL PRIOR ART                   │
│ • GATConv (Veličković 2018)                            │
│ • Sinusoidal Time Encoding (Vaswani 2017)              │
│ • Elliptic Benchmark (Weber 2019)                      │
│ • GNNExplainer (Ying 2019)                             │
│ • GraphFramEx Metrics (Amara 2022)                     │
│ • Double-Edge Swapping (Maslov 2002)                   │
└───────────────────────────┬────────────────────────────┘
                            │ Combined & Critically Extended
                            ▼
┌────────────────────────────────────────────────────────┐
│             ORIGINAL INVESTIGATIVE CONTRIBUTIONS       │
│ 1. Empirical Falsification of GNN Superiority (RF >>) │
│ 2. Discovery of 1-Hop Precomputed Feature Redundancy   │
│ 3. Ego-Residual Skip Formalization & Protection Bound  │
│ 4. Factorial Protocol with Rewired Null Controls       │
│ 5. Holm-Bonferroni FWER Multiplicity Control           │
│ 6. Forensic Discovery of Elliptic++ Lifetime Leakage   │
│ 7. GNNExplainer Indistinguishability from Random (p)   │
│ 8. Leave-One-Out Discrete Oracle Dominance Reference   │
└────────────────────────────────────────────────────────┘
```

---

## 4. Academic Examiner Defense Summary

When questioned by examiners on originality:

> *"This thesis does not claim to have invented Graph Attention Networks, Sinusoidal Positional Encoding, or GNNExplainer. The originality of this research resides in **critical methodological evaluation and diagnostic empirical discovery**. Specifically, this thesis is the first work to systematically demonstrate that precomputed 1-hop aggregate features render graph message passing redundant in Bitcoin transaction graphs, the first to audit the Elliptic++ benchmark and identify future lifetime feature leakage, and the first to show that discrete Leave-One-Out oracles dramatically outperform continuous GNNExplainer relaxations on residual GNN architectures."*
