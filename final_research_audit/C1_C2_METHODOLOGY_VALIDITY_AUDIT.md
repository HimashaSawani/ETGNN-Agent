# C1 and C2 Methodology and Scientific Validity Audit

**Repository:** `ETGNN-Fraud-Detection`  
**Target:** Empirical Protocols Contrast 1 (C1) and Contrast 2 (C2)  
**Auditor:** Independent Senior Academic Research Reviewer & Statistical Referee  
**Date:** October 2026  
**Audit Standard:** 100% Traceable Mathematical and Artifact Evidence Provenance

---

## 1. Executive Summary

This report delivers a deep, point-by-point scientific and statistical validity audit of the two central explainability protocols implemented in the repository:
1. **Contrast 1 (C1):** Temporal Explanation-Sensitivity Comparison ($N = 240$ transactions under shared edge deletions).
2. **Contrast 2 (C2):** Matched Correct-versus-Incorrect Prediction Explanation Contrast ($N = 420$ pair records across 377 unique completed transactions).

```
Protocol Audit Scorecard:
[C1] Architecture & Control Equivalence:  VERIFIED (<1e-10 parameter alignment)
[C1] Intervention Symmetry:              VERIFIED (Shared ETGNN edge cuts, fixed target class)
[C1] Statistical Finding:                VERIFIED (Mean delta_adj = +0.002903, CI spans zero)
[C1] Epistemic Interpretation:           DEFENSIBLE (Small sample difference; equivalence not claimed)
[C2] Hopcroft-Karp Bipartite Matching:   VERIFIED (1-to-1 matching, exact calipers enforced)
[C2] Margin & Degree Calipers:           VERIFIED (|margin| <= 0.050, |E_cand| <= 5, exact out-degree)
[C2] Treatment/Control Direction:        VERIFIED (delta = A(incorrect) - A(correct) = A(FP) - A(TP))
[C2] Statistical Finding:                VERIFIED (ETGNN Primary TP-FP = +0.039343, CI [+0.0202, +0.0574])
[C2] FN Imprecision Reporting:           HONEST & DEFENSIBLE (All 4 FN registers span zero due to N=25)
[C2] Dependence Disclosures:             VERIFIED (Reused nodes and graph autocorrelation disclosed)
```

---

## 2. Phase 4: Contrast 1 (C1) Scientific Validity Audit

Contrast 1 evaluates whether a time-encoded Graph Attention Network (ETGNN) exhibits differential prediction sensitivity to controlled edge deletions compared to an architecturally aligned zero-time comparator (Matched GAT) when subjected to an *identical post-hoc edge intervention set*.

### Detailed Point-by-Point Audit (Items 1–13):

1. **Architecture Equivalence:**  
   - *Verification:* Inspected `src/models/matched_gat.py:L7-16` and `src/models/etgnn.py:L9-26`. `MatchedGATClassifier` directly inherits from `ETGNNClassifier`. Both instantiate identical layer types: `GATConv(181, 16, heads=4)` and `GATConv(64, 2, heads=1)`.
   - *Finding:* Checkpoint weight tensors match dimensionally and structurally.
2. **Exact Time Encoding Implementation:**  
   - *Verification:* Inspected `src/models/time_encoding.py:L5-16`. Uses standard Vaswani-style sinusoidal projection producing a 16-dimensional embedding with frequencies $10000^{2i/16}$ for $t \in [1, 49]$.
3. **Matched GAT Zero-Vector Control:**  
   - *Verification:* In `matched_gat.py:L9-12`, the time features are generated as `torch.zeros((x.shape[0], self.time_dim), dtype=x.dtype, device=x.device)`.
   - *Finding:* Preserves exact input dimension ($165 + 16 = 181$) without adding information.
4. **Training and Checkpoint Comparability:**  
   - *Verification:* Both models trained under identical training scripts (`src/training/train_etgnn.py` and `src/training/train_matched_gat.py`) with identical seed (`42`), learning rate (`0.01`), Adam optimizer, and early stopping on validation AUPRC. Checkpoint hashes verified against `configs/main_c1_c2_run_config.json:L29-44`.
5. **Shared Explanation-Guided Deletion Sets:**  
   - *Verification:* Inspected `src/evaluation/run_main_c1_c2_benchmark.py:L505-520`. The top-$k$ edge set selected by ETGNN ($\Delta E_{\text{shared}}$) is deleted from the computation graph and evaluated on *both* ETGNN and Matched GAT.
6. **Ranking Model Asymmetry:**  
   - *Verification:* ETGNN generates the ranking. Matched GAT is evaluated on "off-target" edges relative to its own internal GNNExplainer masks.
   - *Defensibility:* This asymmetry is explicitly documented in `reports/results_discussion_draft.md:L50-58` and Chapter 5 (Section 5.2). It measures how GAT responds to cuts prioritized by the temporal model.
7. **Random-Deletion Control Implementation:**  
   - *Verification:* For each node, 30 independent random edge draws ($\Delta E_{\text{rand}, r}$) of size $k$ are deleted from $E_{cand}$ using deterministic seeds (`20261020 + node_index`).
   - *Formula Verified:* $A = \Delta P_{\text{top}} - \frac{1}{30} \sum_{r=1}^{30} \Delta P_{\text{rand}, r}$.
8. **Actual Deletion Budgets:**  
   - *Verification:* Budget formula is $k = \max(1, \lceil 0.25 \times E_{cand} \rceil)$.
   - *Discrete Ceiling Impact:* 84.1% of transactions have $E_{cand} = 2 \implies k = 1$ (50.0% actual deletion fraction).
9. **Candidate-Edge Selection:**  
   - *Verification:* $E_{cand}$ is defined as the directed edges present in the 2-hop computation subgraph (`k_hop_subgraph`) around the target node.
10. **Precomputed Aggregate Features Unchanged:**  
    - *Verification:* Columns 94–165 contain precomputed neighborhood aggregates. These remain fixed during edge deletion; deletion perturbs runtime message passing only. This boundary is explicitly disclosed in Chapter 3 (Section 3.3).
11. **Chronological Data Partitioning:**  
    - *Verification:* C1 is evaluated on validation time steps 37–40 (240 nodes). Steps 41–49 (test period) were strictly excluded.
12. **Statistical Estimation & Bootstrap Confidence Intervals:**  
    - *Verification:* Evaluated directly from `results/tables/c1_response_summary.csv:L2`.
    - *Verified Numbers:*
      * $\text{mean } \Delta_{adj} = \mathbf{+0.002903}$
      * Sample standard deviation across transactions: $s = \mathbf{\pm 0.096180}$
      * Median: $-0.001534$
      * Exploratory 95% bootstrap CI: $[-0.009201, +0.015121]$ (zero crossed)
      * Raw difference ($\Delta_{raw}$ without random adjustment): mean $= \mathbf{+0.038491}$ ($s = \pm 0.139170$).
13. **Alignment with Research Question RQ2:**  
    - *Finding:* RQ2 asks whether time encodings alter topological sensitivity. The data shows $\Delta_{adj} \approx +0.0029$, indicating that the average response difference is very small in this sample, but high transaction variance ($s = 0.096$) prevents claiming functional equivalence. C1 is **methodologically sound** and interpreted with scientific restraint.

---

## 3. Phase 5: Contrast 2 (C2) Scientific Validity Audit

Contrast 2 investigates whether GNNExplainer-ranked edge deletions induce differential responses between model prediction errors (False Positives and False Negatives) and model successes (True Positives and True Negatives) under tightly matched prediction confidence and neighborhood structure.

### Detailed Point-by-Point Audit (Items 1–17):

1. **Prediction Category Definitions:**  
   - *Ground Truth:* Illicit ($y = 1$), Licit ($y = 0$). Threshold $\tau = 0.50$.
   - *Definitions:*
     * True Positive (TP): $y = 1, \hat{y} = 1$ ($p \ge 0.50$)
     * False Positive (FP): $y = 0, \hat{y} = 1$ ($p \ge 0.50$)
     * True Negative (TN): $y = 0, \hat{y} = 0$ ($p < 0.50$)
     * False Negative (FN): $y = 1, \hat{y} = 0$ ($p < 0.50$)
2. **FP-versus-TP Matching:**  
   - *Verification:* Evaluates 95 primary matched pairs for ETGNN and 94 primary matched pairs for Matched GAT.
3. **FN-versus-TN Matching:**  
   - *Verification:* Evaluates 25 primary matched pairs for ETGNN and 21 primary matched pairs for Matched GAT.
4. **Prediction-Margin Calipers:**  
   - *Definition:* $\text{margin}(v) = |p_{\text{illicit}}(v) - 0.50|$.
   - *Caliper Enforced:* $|\text{margin}(\text{treat}) - \text{margin}(\text{ctrl})| \le 0.050$.
   - *Verification:* 100% of 420 matched pairs satisfy this caliper.
5. **Candidate-Edge-Count Matching:**  
   - *Caliper Enforced:* $|E_{cand}(\text{treat}) - E_{cand}(\text{ctrl})| \le 5$.
   - *Observed Balance:* In the ETGNN Primary TP-FP register, candidate edge differences were $\le 2$ for all 95 pairs.
6. **Degree-Matched Sensitivity Analysis:**  
   - *Verification:* Sensitivity registers enforce **exact out-degree equality**: $\text{deg}_{out}(\text{treat}) == \text{deg}_{out}(\text{ctrl})$.
   - *Yield:* 57 ETGNN TP-FP pairs, 82 Matched GAT TP-FP pairs, 25 ETGNN TN-FN pairs, 21 Matched GAT TN-FN pairs.
7. **Treatment and Control Orientation:**  
   - *Formula Verified:* $\Delta_{pair} = A(\text{incorrect}) - A(\text{correct}) = A(\text{FP}) - A(\text{TP})$.
   - *Meaning:* Positive sign indicates larger explanation-directed drop (above random) on incorrect predictions.
8. **Explanation-Guided Deletion Effects:**  
   - *Verification:* Deleted edges are derived from model-specific rankings (ETGNN ranking for ETGNN; Matched GAT ranking for Matched GAT).
9. **Random-Deletion Controls:**  
   - *Verification:* Both treatment and control nodes are subtracted by their respective 30-draw random deletion baselines before computing pair difference.
10. **Matched-Pair Difference Calculation:**  
    - *Verification:* Calculated per pair as $\Delta_{pair} = A_{treat} - A_{ctrl}$.
11. **Bootstrap Implementation:**  
    - *Verification:* Matched-pair cluster bootstrap ($B = 2,000$ iterations, seed `20261020`) implemented in `run_main_c1_c2_benchmark.py:L480-515`. Resamples paired observations together, correctly preserving within-pair covariance.
12. **Confidence Interval Construction:**  
    - *Verification:* Empirical percentile interval $[2.5\%, 97.5\%]$ over 2,000 bootstrap resamples.
13. **Multiple-Comparison Limitations:**  
    - *Status:* All 8 bootstrap confidence intervals in Table 5.2 are explicitly labelled as marginal and unadjusted for family-wise error rate.
14. **Graph Dependence Caveat:**  
    - *Verification:* Formally disclosed in Table 5.2 and Section 5.3: *"Transactions reside in a connected graph; pairs assume independent cluster resampling."*
15. **Reused Transactions:**  
    - *Verification:* Total completed unique transactions is 377. Reused across primary and sensitivity registers ($N = 420$ pair records).
16. **Overlapping Matching Registers:**  
    - *Verification:* Primary and Degree-Sensitivity registers comprise different, overlapping subsets (e.g., 57 degree pairs are a subset of candidate-caliper matches). Documented in `CHANGE_LOG.md:L35-39`.
17. **Independent Observation Status:**  
    - *Verification:* Pairs are explicitly described as *pair records*, not 420 independent graph components.

---

## 4. Verification of Core Numerical Findings

All numerical values match underlying artifact files (`results/tables/c2_registers_bootstrap_summary.csv` and `c1_response_summary.csv`) with zero discrepancy:

| Register | Model | Cohort Contrast | Mean Paired Difference | 95% Cluster Bootstrap CI | Zero Crossed? | Scientific Interpretation |
|---|---|:---:|:---:|:---:|:---:|---|
| **Primary** | **ETGNN** | **TP – FP** | **+0.039343** | **[+0.020234, +0.057410]** | **NO** | False Positives exhibit a statistically significant larger probability drop than matched True Positives. |
| **Degree-Sensitivity** | **ETGNN** | **TP – FP** | **+0.043455** | **[+0.019882, +0.069552]** | **NO** | Effect persists under exact out-degree matching. |
| **Primary** | **Matched GAT** | **TP – FP** | **+0.020315** | **[+0.003917, +0.038684]** | **NO** | Positive difference observed in baseline GAT. |
| **Degree-Sensitivity** | **Matched GAT** | **TP – FP** | **+0.010547** | **[-0.013240, +0.032770]** | **YES** | Lower bound crosses zero; sensitive to matching specification. |
| **Primary** | **ETGNN** | **TN – FN** | -0.058027 | [-0.128831, +0.006393] | **YES** | Imprecise due to small sample ($N=25$). Inconclusive. |
| **Degree-Sensitivity** | **ETGNN** | **TN – FN** | -0.019748 | [-0.081593, +0.036312] | **YES** | Imprecise due to small sample ($N=25$). Inconclusive. |
| **Primary** | **Matched GAT** | **TN – FN** | +0.052255 | [-0.046153, +0.162460] | **YES** | Imprecise due to small sample ($N=21$). Inconclusive. |
| **Degree-Sensitivity** | **Matched GAT** | **TN – FN** | +0.072489 | [-0.024721, +0.178587] | **YES** | Imprecise due to small sample ($N=21$). Inconclusive. |

---

## 5. Methodological Validity Verdict

Both C1 and C2 experiments are **scientifically valid, reproducible, and methodologically sound**. The thesis deliverables maintain complete fidelity to the empirical outputs, make zero claims of causal error repair, and transparently disclose all statistical limitations.
