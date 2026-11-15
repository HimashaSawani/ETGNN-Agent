# Defensible Academic Contribution Statements

**Repository:** `ETGNN-Fraud-Detection`  
**Purpose:** Standardized, peer-review-grade phrasing for research contributions, empirical findings, and limitations  
**Auditor:** Independent Senior Academic Research Reviewer & Scientific Editor  
**Date:** October 2026

---

## 1. Executive Summary

This document establishes the authoritative, academically defensible formulations of the research contributions, empirical findings, epistemic boundaries, and future directions for the undergraduate research thesis titled **"Explainable Temporal Graph Neural Networks for Adaptive Financial Fraud Detection"** (working refinement: *"Explainable Time-Encoded Graph Attention Networks for Bitcoin Illicit Transaction Classification"*).

These statements are calibrated for direct use in the Abstract, Introduction (Section 1.5), Results Discussion (Chapter 5), Conclusion (Chapter 6), and oral viva defense.

---

## 2. Standardized Contribution Statements

### 2.1 Primary Research Contribution: Matched Error-Cohort XAI Protocol (Contrast 2)
> *"The primary scientific contribution of this thesis is the design and execution of a controlled, matched-pair post-hoc explainability benchmarking protocol (Contrast 2) that evaluates GNNExplainer edge deletion responses across model prediction error cohorts. By employing Hopcroft-Karp bipartite matching on within-node prediction margin ($|\text{margin}| \le 0.050$) and candidate computation edge count ($|E_{cand}| \le 5$, with exact out-degree matching in sensitivity registers), this methodology isolates structural explanation sensitivity from baseline prediction confidence. Resampled via matched-pair cluster bootstrap ($B = 2,000$), empirical results demonstrate that for the time-encoded GAT, False Positives experience a statistically significant larger probability drop under explanation edge deletion than matched True Positives (mean paired difference $+0.039343$, 95% CI $[+0.020234, +0.057410]$). This reveals that model false alarms are disproportionately dependent on sparse, fragile graph edges, addressing a documented blind spot in peer-reviewed GNN explainability literature."*

---

### 2.2 Secondary Research Contribution: Shared-Intervention Temporal Conditioning Audit (Contrast 1)
> *"As a secondary contribution, this research establishes a controlled cross-model perturbation protocol (Contrast 1) evaluating whether adding temporal feature encodings alters a GNN's topological sensitivity under identical post-hoc edge cuts. By applying the exact same top-$k$ edge intervention set derived from ETGNN ($\Delta E_{\text{shared}}$) to both ETGNN and an architecturally aligned, parameter-matched zero-time GAT control across 240 validation transactions, C1 isolates the explanatory impact of temporal conditioning. The observed mean random-adjusted difference was small ($\Delta_{adj} = +0.002903$, 95% bootstrap CI $[-0.009201, +0.015121]$), indicating that temporal conditioning did not induce a substantial differential response to shared edge deletions on average, although notable transaction-level variance ($s = \pm 0.096180$) was observed."*

---

### 2.3 Empirical Finding 1: Tabular Dominance and Hypothesis H1 Outcome
> *"In a rigorous, leak-free chronological evaluation on held-out future time steps (41–49, $N = 9,973$ nodes), standard tabular tree ensembles decisively outperformed all evaluated Graph Neural Networks. Random Forest achieved the highest illicit-class Average Precision ($\text{AP} = 0.6323$), substantially outperforming XGBoost ($\text{AP} = 0.5472$), GraphSAGE ($\text{AP} = 0.2427$), the proposed ETGNN ($\text{AP} = 0.2250$), Matched GAT ($\text{AP} = 0.2099$), and standard GCN ($\text{AP} = 0.1306$). These findings decisively reject Hypothesis H1 and demonstrate that precomputed neighborhood features combined with tabular tree ensembles remain superior to raw message passing for transaction-level Bitcoin forensic classification."*

---

### 2.4 Empirical Finding 2: Topological Message-Passing Ablation
> *"A controlled single-variable ablation on local features (excluding precomputed graph aggregates) revealed that removing graph connectivity from ETGNN caused held-out test Average Precision to surge from $0.2344$ to $0.4812$ (+0.2468 increase). This descriptive comparison provides empirical evidence supporting the hypothesis that unconstrained message passing across intra-timestep payment edges propagates structural noise and aggregates heterogeneous labels from dense transaction hubs, diluting strong local feature signals."*

---

### 2.5 Methodological Transparency: Retrospective Validation Reuse
> *"The post-hoc explainability benchmark (Contrasts C1 and C2) was conducted across 377 completed transactions originating from validation time steps 37–40 (following exploratory pilot development on steps 33–36). Because these transactions reside within the validation split where model early stopping occurred, these evaluations represent structured retrospective diagnostic audits rather than prospective holdout trials. Furthermore, historical test period data (steps 41–49) was inspected during preliminary research stages and is reported under strictly frozen model checkpoints."*

---

### 2.6 Epistemic Boundaries and Operational Definitions
> *"Post-hoc edge deletion responses measure empirical model prediction sensitivity under topological omission; they do not constitute ground-truth causal explanations of criminal behavior, proof of legal culpability, or automated error correction. Furthermore, due to the discrete ceiling budget rule $k = \max(1, \lceil 0.25 \times E_{cand} \rceil)$, transactions with $E_{cand} = 2$ (representing $84.08\%$ of evaluated nodes) experienced an actual structural deletion fraction of $50.0\%$, demonstrating that discrete graph budgets distort nominal perturbation percentages."*

---

### 2.7 Defensible Directions for Future Research
> *"Future research should extend this framework in three principal directions: (1) evaluating continuous-time dynamic memory networks (e.g., TGN, JODIE) on high-frequency cryptocurrency interaction graphs with microsecond inter-block timestamps; (2) benchmarking alternative explainability optimizers (such as Zorro, SubgraphX, or PGExplainer) across matched error cohorts; and (3) establishing joint feature-and-edge perturbation protocols to evaluate multi-modal explanation fidelity."*

---

## 3. Recommended Phrasing Matrix for Oral Viva Defense

| Question from Examiner | Defensible Phrasing for Candidate | What NOT to Say |
|---|---|---|
| *"Why didn't ETGNN beat Random Forest?"* | *"Random Forest achieved 0.6323 AP vs 0.2250 AP because Elliptic's precomputed neighborhood aggregates provide dense statistical summaries without suffering from the label noise propagated during raw message passing across dense transaction hubs. This confirms earlier findings by Weber et al. (2019)."* | *"Our GNN was state-of-the-art and almost beat Random Forest."* |
| *"Is ETGNN really an adaptive temporal model?"* | *"ETGNN is a static Graph Attention Network supplied with fixed 16-dimensional sinusoidal time-step encodings. It operates on discrete snapshots. We explicitly delimited our scope in Section 1.6: it does not perform online parameter adaptation or dynamic continuous-time memory."* | *"Yes, ETGNN adapts in real-time to changing blockchain patterns."* |
| *"What is the main novelty of your thesis?"* | *"Our primary novelty is the experimental protocol of Contrast 2: we are the first to evaluate GNNExplainer responses on matched error cohorts (False Positives vs True Positives) while strictly controlling for prediction margin and local candidate edges using Hopcroft-Karp matching and cluster bootstrap inference."* | *"We invented the first-ever explainable temporal GNN in the world."* |
| *"Does C1 prove that ETGNN and GAT are equivalent?"* | *"No. The mean adjusted difference was small (+0.0029), but transaction variance across nodes was substantial (s = 0.096). We conclude that within this sample, the models showed comparable sensitivity on average, but functional equivalence is not established."* | *"Yes, C1 proves both models are 100% functionally equivalent."* |
| *"Does GNNExplainer correct model errors?"* | *"No. Deleting edges measures local model prediction sensitivity to topological cuts. It demonstrates that False Positives are more fragile (+0.0393 drop difference), but does not repair errors or identify criminal intent."* | *"Yes, GNNExplainer fixes false alarms by removing bad edges."* |
