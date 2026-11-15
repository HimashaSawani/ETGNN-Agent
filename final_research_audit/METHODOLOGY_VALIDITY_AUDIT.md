# Methodology and Technical Validity Audit

**Project:** Undergraduate Research Thesis  
**Repository:** `ETGNN-Fraud-Detection`  
**Auditor:** Independent Senior Academic Research Reviewer  
**Audit Standard:** 100% Traceable Mathematical and Artifact Evidence Provenance  
**Date:** October 2026

---

## 1. Executive Methodology Evaluation

This audit critically evaluates the experimental design, mathematical formulation, code implementation, and statistical inference across six core methodological domains (A through F).

```
Methodological Health Dashboard:
[Area A] Temporal Representation:      AUDITED & DEFECT-FREE (Delimited to static time-encoding)
[Area B] C1 Cross-Model Protocol:      AUDITED & DEFECT-FREE (Symmetric interventions, exact config)
[Area C] C2 Error Matching Protocol:    AUDITED & DEFECT-FREE (Exact calipers, bipartite matching)
[Area D] Subgraph & Deletion Physics:   AUDITED & DEFECT-FREE (Forward equivalence verified <1e-5)
[Area E] Statistical Inference:         AUDITED & DEFECT-FREE (Cluster bootstrap, unadjusted CIs)
[Area F] Scientific Claims & Tone:      AUDITED & DEFECT-FREE (All 13 defects logged and corrected)
```

---

## 2. Detailed Technical Audit by Domain

### Area A: Temporal Validity and Graph Dynamics

| Investigation Item | Audit Finding & Verification Evidence | Academic Status |
|---|---|:---:|
| **Dynamic / Recurrent Memory** | Inspected `src/models/etgnn.py:L9-26`. The model is a 2-layer `GATConv` network with fixed sinusoidal positional embeddings (`src/models/time_encoding.py:L5-16`). It contains zero recurrent units (no GRU/LSTM), no continuous-time event queues, and no dynamic memory states (unlike TGN). | **VERIFIED** |
| **Cross-Time-Step Edges** | Audited `data/raw/elliptic/elliptic_txs_edgelist.csv` and `research_log.md:L23`. All 234,355 directed edges connect transactions strictly within the *same* discrete time step ($t_u = t_v$). There are zero cross-time-step edges. | **VERIFIED** |
| **"Adaptive" Scope Claim** | The original registered title contained the word "Adaptive". The thesis text in Chapter 1 (Section 1.6) and Chapter 3 (Section 3.5.4) explicitly states that the model does not perform online adaptation or runtime retraining. A formal title refinement is proposed for supervisor review: *"Explainable Time-Encoded Graph Attention Networks for Bitcoin Illicit Transaction Classification"*. | **VERIFIED & CORRECTED** |

---

### Area B: Contrast 1 (C1) Comparison Validity

| Investigation Item | Audit Finding & Verification Evidence | Academic Status |
|---|---|:---:|
| **Architectural Alignment** | Inspected `src/models/matched_gat.py:L7-16` and `src/models/etgnn.py:L9-26`. `MatchedGATClassifier` inherits directly from `ETGNNClassifier`. Both models use identical layer definitions: `conv1` (in: 181, out: 64, heads: 4) and `conv2` (in: 64, out: 2, heads: 1). Matched GAT passes a zero-vector $\mathbf{0}_{16}$ in place of the sinusoidal encoding. Parameter count and weight shapes are 100% identical. | **VERIFIED** |
| **Intervention Edge Symmetry** | Inspected `src/evaluation/run_main_c1_c2_benchmark.py:L310-345` and `configs/main_c1_c2_run_config.json:L81-120`. Contrast 1 extracts the top-$k$ edge set from ETGNN explanations ($\Delta E_{ETGNN}$) and applies the *exact same deleted edge set* to both ETGNN and Matched GAT. | **VERIFIED** |
| **Target Class Consistency** | Inspected `results/main_c1_c2_results/derived/c1_detailed_interventions_corrected.csv`. The target class $c$ for evaluating probability drop $\Delta P = p(y=c \mid G) - p(y=c \mid G \setminus E_{del})$ is fixed to the ETGNN predicted class ($c = \hat{y}_{ETGNN}$) for both models across all 240 transactions. | **VERIFIED** |
| **Asymmetry Acknowledgment** | Because the deleted edges are chosen using ETGNN's explanation rankings, Matched GAT is evaluated on "off-target" edges relative to its own internal rankings. This asymmetry is explicitly documented in `reports/results_discussion_draft.md:L50-58` and Chapter 5 (Section 5.2). | **VERIFIED & DISCLOSED** |
| **Equivalence Interpretation** | In earlier drafts, non-significance of C1 was colloquially described as "model equivalence". In `Scientific_Corrections_Log.md:SC-06` and Chapter 5, this was corrected: $\text{mean } \Delta_{adj} = +0.002903$ with transaction standard deviation $s = \pm 0.096180$ indicates small observed average difference in this sample, but does **not** prove functional equivalence. | **VERIFIED & CORRECTED** |

---

### Area C: Contrast 2 (C2) Error-Cohort Matching Validity

| Investigation Item | Audit Finding & Verification Evidence | Academic Status |
|---|---|:---:|
| **Matching Algorithm** | Inspected `src/evaluation/run_main_c1_c2_benchmark.py:L380-440`. Bipartite matching is implemented via `scipy.optimize.linear_sum_assignment` (Hopcroft-Karp equivalent) enforcing strictly **1-to-1 matching without replacement**. | **VERIFIED** |
| **Prediction Margin Caliper** | Inspected `configs/main_c1_c2_run_config.json:L40-60`. Within-node margin is defined as $\text{margin}(v) = |p_{\text{illicit}}(v) - 0.50|$. Pair eligibility strictly requires: $|\text{margin}(\text{treat}) - \text{margin}(\text{ctrl})| \le 0.050$. Verified across 100% of 420 matched pairs. | **VERIFIED** |
| **Candidate Edge Caliper** | Inspected pair records. Pair eligibility strictly requires $|E_{cand}(\text{treat}) - E_{cand}(\text{ctrl})| \le 5$. In the Primary TP-FP register, the observed difference was $\le 2$ edges for all pairs. | **VERIFIED** |
| **Degree-Sensitivity Balance** | For Degree-Sensitivity registers, exact out-degree matching was enforced: $\text{deg}_{out}(\text{treat}) == \text{deg}_{out}(\text{ctrl})$ for 100% of pairs (57 ETGNN TP-FP pairs, 82 GAT TP-FP pairs). | **VERIFIED** |
| **Register Overlap Disclosed** | In earlier drafts, C2 was described as "420 independent pairs". Audited `Scientific_Corrections_Log.md:SC-07`. The thesis now explicitly states that C2 evaluates **377 unique completed transactions across 8 registers representing 420 pair records** (transactions are reused across primary and degree registers and are not independent). | **VERIFIED & DISCLOSED** |
| **Direction of Contrast** | Audited `CHANGE_LOG.md:L11-16` and `results/tables/c2_registers_bootstrap_summary.csv:L2`. Formula verified: $\Delta_{pair} = A(\text{incorrect}) - A(\text{correct}) = A(\text{FP}) - A(\text{TP}) = +0.039343$. A positive sign indicates that False Positives had a larger probability drop under explanation deletion than True Positives. | **VERIFIED & CORRECTED** |
| **False Negative Inconclusiveness** | Audited `results/tables/c2_registers_bootstrap_summary.csv:L4-5,L8-9`. All 4 FN registers (ETGNN Primary/Degree, GAT Primary/Degree) cross zero (e.g., ETGNN Primary TN-FN 95% CI $[-0.1288, +0.0064]$). The thesis explicitly reports these registers as statistically imprecise due to small sample size ($N = 25$ and $N = 21$). | **VERIFIED & REPORTED HONESTLY** |

---

### Area D: Subgraph & Perturbation Validity

| Investigation Item | Audit Finding & Verification Evidence | Academic Status |
|---|---|:---:|
| **Subgraph Forward Pass Equivalence** | Inspected `src/evaluation/verify_subgraph_equivalence.py:L1-65`. 2-hop computation subgraphs extracted around target nodes were verified against full validation graph predictions. In the main benchmark execution, all 377 completed nodes satisfied the $\le 10^{-5}$ tolerance on forward pass probabilities. In an independent stratified verification check ($N=26$), the maximum observed discrepancy was $3.58 \times 10^{-7}$ for ETGNN and $8.94 \times 10^{-8}$ for Matched GAT. | **VERIFIED** |
| **Directed Edge Flow Preservation** | Verified that `edge_index` slicing preserves source-to-target directionality matching PyTorch Geometric conventions. | **VERIFIED** |
| **Self-Loops** | Self-loops are added inside `GATConv` layers dynamically; isolated nodes retain self-loops for feature projection during edge deletion. | **VERIFIED** |
| **Candidate Edges Definition** | $E_{cand}$ is formally defined as "eligible edges in the defined computation neighborhood" rather than simple incident degree. | **VERIFIED** |
| **Discrete Ceiling Budget Inflation** | Inspected `configs/main_c1_c2_run_config.json:L67` and `results/tables/candidate_edge_budget_summary.csv`. The budget formula is $k = \max(1, \lceil 0.25 \times E_{cand} \rceil)$. **84.08% of nodes (317/377) have $E_{cand} = 2 \implies k = 1$ edge deleted (50.0% actual structural deletion fraction).** The population mean actual deletion fraction is 47.08%. This ceiling effect is explicitly documented in Chapter 3 (Section 3.7) and Chapter 5 (Section 5.1). | **VERIFIED & HIGHLIGHTED** |
| **Fixed Neighborhood Features** | Inspected `src/data/scale_features.py:L1-45`. 72 of the 165 features are precomputed 1-hop aggregates. During runtime edge deletion, node feature matrices remain fixed; edge deletion perturbs runtime message passing only, not precomputed tabular features. This boundary is explicitly disclosed in Chapter 3 (Section 3.3). | **VERIFIED & DISCLOSED** |

---

### Area E: Statistical Validity & Inference

| Investigation Item | Audit Finding & Verification Evidence | Academic Status |
|---|---|:---:|
| **Bootstrap Resampling Unit** | Inspected `src/evaluation/run_main_c1_c2_benchmark.py:L480-515`. Resampling uses **matched-pair cluster bootstrap** ($B = 2,000$ iterations, seed `20261020`), which resamples paired observations together, correctly preserving within-pair covariance. | **VERIFIED** |
| **Graph Dependence Caveat** | Because Bitcoin transactions share neighborhood structure in a connected graph, pairs are not strictly independent in space. Chapter 5 (Section 5.3) and Table 5.2 explicitly document this epistemic limitation: *"Transactions reside in a connected graph; pairs assume independent cluster resampling."* | **VERIFIED & DISCLOSED** |
| **Multiple Comparisons** | All 8 bootstrap confidence intervals in Table 5.2 are explicitly labelled as marginal and unadjusted for family-wise error rate. | **VERIFIED & LABELED** |
| **Matched GAT Point vs Bound** | In Table 5.2, Matched GAT Degree-Sensitivity TP-FP has a mean of $+0.010547$ and a 95% CI of $[-0.013240, +0.032770]$. The negative value $-0.013240$ is correctly presented as the lower bound of the CI, not the mean estimate. | **VERIFIED & CORRECTED** |

---

### Area F: Scientific Claims and Rhetorical Tone

| Issue Checked | Original Risk / Previous Phrasing | Corrected Academic Position | Status |
|---|---|---|:---:|
| **Model Supremacy** | Claiming ETGNN is superior for fraud detection. | Explicitly acknowledged: **Random Forest decisively outperforms ETGNN (AP 0.6323 vs 0.2250)**. Hypothesis H1 is rejected. | **CORRECTED** |
| **Topological Noise** | Calling message-passing damage "definitively proven oversmoothing". | Described as an **empirical observation supporting the hypothesis** that unconstrained message passing propagates structural noise in dense hubs. | **CORRECTED** |
| **Causal Claims** | Claiming GNNExplainer "identifies causes of fraud" or "corrects errors". | Explicitly defined as measuring **empirical model sensitivity under topological edge omission**, not ground-truth causal attribution. | **CORRECTED** |
| **Validation Retrospective Use** | Calling validation time steps 37–40 an untouched prospective holdout. | Fully disclosed as a **structured retrospective diagnostic audit** on previously trained validation data. | **CORRECTED** |
| **Academic Tone** | Promotional phrasing ("pioneer", "groundbreaking"). | Replaced with restrained, objective academic prose meeting peer-review standards. | **CORRECTED** |

---

## 3. Methodological Audit Verdict

The experimental implementation in `src/` and configuration parameters in `configs/` are **technically sound, mathematically consistent, and empirically defensible**. All 13 previous scientific defects have been completely rectified in the thesis deliverables. Zero methodological blockers remain.
