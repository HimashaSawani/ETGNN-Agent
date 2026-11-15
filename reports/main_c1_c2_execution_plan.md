# Main C1 + C2 Execution Plan & Pre-Run Audit Report

**Document Version:** 2.0 (Pre-Run Configuration & Explicit Sample Registers Frozen)  
**Date:** October 20, 2026  
**Status:** Pre-Run Configuration Frozen Prior to Main-Period Execution  
**Hardware Profile:** Verified via 4-Node Technical Pilot Benchmark  
**Configuration Target:** [`configs/main_c1_c2_run_config.json`](file:///d:/ETGNN-Fraud-Detection/configs/main_c1_c2_run_config.json)  
**Scope:** Defines the complete specification, sample registers, uncertainty framework, and precision planning for Contrast 1 (C1) and Contrast 2 (C2) on the Elliptic Bitcoin dataset.

---

## 1. Executive Summary & Resolution of Technical Pilot Review Points

Prior to deploying evaluation on the main validation period (time steps 37–40), all technical and methodological requirements have been resolved, executed on development nodes, and audited:

1. **Explicit Summary Naming & Node 134593 Distinction:**  
   The column previously reported as `advantage_A_gat` was derived from `c1_drop_gat` (probability drops relative to ETGNN's predicted class on ETGNN-ranked edge deletions). It is now explicitly labeled:
   - `c1_advantage_gat_common_target`: GAT advantage on ETGNN-ranked deletions with respect to the common ETGNN target.
   - For Node 134593 ($E=2, k=1$, False Negative in Step 34), probabilities are explicitly labeled:
     - **ETGNN:** $p(\text{class 0}) = 0.551779$, $p(\text{illicit}) = 0.448221 \implies \hat{y}_{\text{ETGNN}} = 0$.
     - **Matched GAT:** $p(\text{class 0}) = 0.428185$, $p(\text{illicit}) = 0.571815 \implies \hat{y}_{\text{GAT}} = 1$.
   - **C1 Common-Target Advantage (Target = 0, ETGNN Ranking):**
     $$\bar{D}_{\text{GE, GAT}}^{(\text{C1})} - \bar{D}_{\text{Rnd, GAT}}^{(\text{C1})} = 0.15685 - 0.09109 = \mathbf{+0.06576057} \quad (\text{Colab GPU})$$
   - **Own-Target Evaluation on the *Same* Interventions (Target = 1):**
     Because $P(Y=1) = 1 - P(Y=0)$, the signed probability drop on class 1 satisfies $D_1 \equiv -D_0$ identically:
     $$\bar{D}_{\text{GE, GAT}}^{(\text{diagnostic})} - \bar{D}_{\text{Rnd, GAT}}^{(\text{diagnostic})} = -0.15685 - (-0.09109) = \mathbf{-0.06576057} \quad (\text{Colab GPU})$$
   - This mathematical symmetry proves that `advantage_A_gat` was strictly a C1 common-target metric, not an intrinsic GAT advantage.

2. **Independent Implementation of C2 for EACH Model:**  
   - GNNExplainer is now instantiated and optimized separately on `m_gat` (`explainer_gat`) for GAT's own original predicted class ($\hat{y}_{\text{GAT}}$) across seeds $[42, 43, 44]$.
   - GAT's own edge mask selects GAT's own top-$k$ edges for deletion, yielding `c2_drop_gat_own_ranking`.
   - GAT's own C2 advantage is computed as `c2_advantage_gat_own_ranking = mean(c2_drop_gat_own_ranking) - mean(c2_rnd_drop_gat_own)`.
   - On Node 134593, seed 42 demonstrates the topological divergence: ETGNN selects local edge 0 (global edge ID `[142517]`), whereas Matched GAT selects local edge 1 (global edge ID `[142518]`).

3. **Preservation of Shared C1 Protocol:**  
   - C1 strictly maintains identical ETGNN-ranked edge deletions and identical seeded random deletion sets (30 draws) across both `etgnn.pt` and `matched_gat.pt`, scored against the common ETGNN target.

4. **Schema & Typo Fix:**  
   - All deleted global edge IDs, ranking models, scoring models, target classes, probabilities, and runtimes are saved in [`detailed_intervention_records.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/technical_pilot_results/detailed_intervention_records.csv) (408 rows) and [`raw_benchmark_records.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/technical_pilot_results/raw_benchmark_records.csv) (140 rows).
   - The runner typo (`"c1_drop_gat": d_c1_gt`) is fixed and committed.
   - Version 1 is preserved in [`experiments/technical_pilot_results_v1/`](file:///d:/ETGNN-Fraud-Detection/experiments/technical_pilot_results_v1/).
   - Cryptographic manifest [`manifest.sha256`](file:///d:/ETGNN-Fraud-Detection/manifest.sha256) and bundle [`ETGNN_Technical_Pilot_Bundle.zip`](file:///d:/ETGNN-Fraud-Detection/ETGNN_Technical_Pilot_Bundle.zip) are rebuilt.

5. **Candid Cryptographic Provenance Reconciliation:**  
   - As documented in [`input_provenance_reconciliation.md`](file:///d:/ETGNN-Fraud-Detection/experiments/technical_pilot_results/input_provenance_reconciliation.md), any earlier reported hashes that differ from the actual physical artifacts are erroneous transcription entries or unrecoverable historical references. We do not claim newline normalization explains unmatched hashes. Physical files on disk are the verified ground truth.

---

## 2. Graph Time-Step Ranges & Test Isolation Disclosure

### 2.1 Time-Step Distributions in Validation Snapshot
The file [`data/processed/validation_graph.pt`](file:///d:/ETGNN-Fraud-Detection/data/processed/validation_graph.pt) (114,002,507 bytes; SHA-256: `261b7ee5951d386bee9843adfc2471726ec9f433341d7c4359c7f66e44cbfc71`) exhibits the following structure:

- **All Graph Nodes in `validation_graph.pt`:**
  - Total node count: **161,603 nodes**.
  - Time-step range: **Steps 1 through 40** (minimum = 1, maximum = 40).
  - *Structural Rationale:* To ensure valid 2-hop message passing for transactions occurring in the validation period without dropping historical topological dependencies, the computation snapshot retains predecessor transactions from steps 1–32 alongside validation steps 33–40.
- **Validation Evaluation Mask (`val_mask`):**
  - Total labelled evaluation nodes: **7,653 nodes**.
  - Time-step range: **Steps 33 through 40** (minimum = 33, maximum = 40; unique steps: $[33, 34, 35, 36, 37, 38, 39, 40]$).
  - Training mask (`train_mask`): 28,938 labelled nodes spanning steps 1 through 32.
  - Test mask (`test_mask`): Empty ($0$ nodes). Test transactions (steps 41–49) are completely absent.

### 2.2 Historical Test Period Disclosure
- The held-out test period (time steps 41–49) was previously inspected during the frozen one-time baseline evaluation on 2026-09-25 (recorded in [`research_log.md`](file:///d:/ETGNN-Fraud-Detection/research_log.md)).
- No test data is touched, re-evaluated, or accessible during this C1/C2 explainability study.
- No model has been tuned or retrained following that test evaluation.

---

## 3. Contrast 1 (C1: Between-Model Comparison) Specification

### 3.1 Primary Contrast vs. Secondary Diagnostic
For each transaction $v$ evaluated under C1, two between-model metrics are computed:

1. **Primary Contrast — Random-Adjusted Advantage Difference ($\delta_{\text{adj}}$):**
   $$\delta_{\text{adj}}(v) = A_{\text{ETGNN}}(v) - A_{\text{GAT, common}}(v)$$
   where:
   $$A_{\text{ETGNN}}(v) = \bar{D}_{\text{GE, ETGNN}}(v) - \bar{D}_{\text{Rnd, ETGNN}}(v)$$
   $$A_{\text{GAT, common}}(v) = \bar{D}_{\text{GE, GAT}}^{(\text{C1})}(v) - \bar{D}_{\text{Rnd, GAT}}^{(\text{C1})}(v)$$
   - *Pre-Registered Role:* Primary test of between-model explanation fidelity. It subtracts each model's architecture-specific noise floor under non-specific random edge removal, isolating the net explanation advantage over random deletion.

2. **Secondary Diagnostic — Raw Shared-Deletion Response Difference ($\delta_{\text{raw}}$):**
   $$\delta_{\text{raw}}(v) = \bar{D}_{\text{GE, ETGNN}}(v) - \bar{D}_{\text{GE, GAT}}^{(\text{C1})}(v)$$
   - *Pre-Registered Role:* Supporting sensitivity diagnostic measuring the unadjusted physical probability drop under identical message-passing edge loss. It verifies whether differences in $\delta_{\text{adj}}$ are driven by the targeted edge rankings themselves or by disparities in random baseline drops.

### 3.2 Explicit C1 Sample Register
Contrast 1 evaluates the **240 unique transactions** derived from the primary ETGNN matched pairs in the main validation period (time steps 37–40):
- **True Positives (TP):** 95 transactions (mean $E_{\text{cand}} = 2.07$, mean margin = $0.319$)
- **False Positives (FP):** 95 transactions (mean $E_{\text{cand}} = 2.02$, mean margin = $0.308$)
- **False Negatives (FN):** 25 transactions (mean $E_{\text{cand}} = 3.00$, mean margin = $0.226$)
- **True Negatives (TN):** 25 transactions (mean $E_{\text{cand}} = 2.80$, mean margin = $0.226$)
- All 240 transaction IDs, node indices, time steps, candidate edge counts, and targets are recorded in `c1_sample_register` inside [`configs/main_c1_c2_run_config.json`](file:///d:/ETGNN-Fraud-Detection/configs/main_c1_c2_run_config.json).

---

## 4. Contrast 2 (C2: Within-Model Caliper-Matched Pairs) Specification

Contrast 2 evaluates within-model paired explanation advantage differences:
$$\Delta_{\text{pair}, i} = A(u_{\text{incorrect}, i}) - A(v_{\text{matched correct}, i})$$

### 4.1 Primary Caliper Matched Pair Registers
Matching is 1-to-1 without replacement with strict calipers: $|\Delta \text{margin}| \le 0.05$ and $|\Delta E_{\text{cand}}| \le 5$.

| Register File | Model | Contrast | Matched Pairs | Max $\Delta \text{Margin}$ | Max $\Delta E_{\text{cand}}$ | Post SMD Margin | Post SMD Edges |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [`caliper_pairs_etgnn_full_tp_fp.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/caliper_pairs_etgnn_full_tp_fp.csv) | `ETGNN_full` | TP–FP | **95** | 0.0470 | 2 | -0.083 | -0.003 |
| [`caliper_pairs_etgnn_full_tn_fn.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/caliper_pairs_etgnn_full_tn_fn.csv) | `ETGNN_full` | TN–FN | **25** | 0.0074 | 1 | -0.002 | +0.011 |
| [`caliper_pairs_matched_gat_tp_fp.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/caliper_pairs_matched_gat_tp_fp.csv) | `Matched_GAT` | TP–FP | **94** | 0.0500 | 3 | -0.129 | +0.008 |
| [`caliper_pairs_matched_gat_tn_fn.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/caliper_pairs_matched_gat_tn_fn.csv) | `Matched_GAT` | TN–FN | **21** | 0.0030 | 1 | +0.000 | +0.000 |

### 4.2 Degree-Balanced Sensitivity Registers
To assess sensitivity to unconstrained out-degree, the secondary register enforces exact out-degree matching ($|\Delta \text{out\_degree}| = 0$). All pair IDs, treat/ctrl indices, txIds, margin differences, edge differences, out-degree differences, and total-degree differences are codified in `configs/main_c1_c2_run_config.json`:

| Register File | Model | Contrast | Matched Pairs | Post SMD Margin | Post SMD Edges | Post SMD Out-Degree | Post SMD Total Degree |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [`degree_sensitivity_pairs_etgnn_full_tp_fp.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/degree_sensitivity_pairs_etgnn_full_tp_fp.csv) | `ETGNN_full` | TP–FP | **57** | -0.002 | +0.016 | **+0.000** | +0.031 |
| [`degree_sensitivity_pairs_etgnn_full_tn_fn.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/degree_sensitivity_pairs_etgnn_full_tn_fn.csv) | `ETGNN_full` | TN–FN | **25** | -0.017 | +0.029 | **+0.000** | +0.022 |
| [`degree_sensitivity_pairs_matched_gat_tp_fp.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/degree_sensitivity_pairs_matched_gat_tp_fp.csv) | `Matched_GAT` | TP–FP | **82** | -0.114 | +0.010 | **+0.000** | +0.035 |
| [`degree_sensitivity_pairs_matched_gat_tn_fn.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/degree_sensitivity_pairs_matched_gat_tn_fn.csv) | `Matched_GAT` | TN–FN | **21** | -0.001 | -0.003 | **+0.000** | +0.007 |

---

## 5. Execution-Node Union Across All Registers

To eliminate redundant computation during execution:
- An exact union was constructed across: C1 sample (240 nodes), ETGNN primary pairs (240 nodes), GAT primary pairs (230 nodes), ETGNN sensitivity pairs (164 nodes), and GAT sensitivity pairs (206 nodes).
- **Total Unique Execution Nodes:** Exactly **377 unique transactions**.
- *Execution Efficiency:* Every node in the union is evaluated with GNNExplainer (seeds 42, 43, 44) and 30 random draws **exactly once**.
- *Sample Preservation:* In post-processing, each pair register and C1 sample draws its corresponding transactions independently, preserving strictly separate analysis samples without cross-contamination.

---

## 6. Uncertainty Analysis & Matched-Pair Cluster Bootstrap

### 6.1 Hierarchical Aggregation Rule
- Explainer seeds ($s \in \{42, 43, 44\}$) and random draws ($r \in \{1, \dots, 30\}$) are within-unit stochastic repetitions.
- They are averaged at the node level *first* to produce scalar node advantages $A(u)$ and $A(v)$.
- Within-pair differences $\Delta_{\text{pair}, i} = A(u_i) - A(v_i)$ form the elementary unit of inferential analysis.

### 6.2 Matched-Pair Cluster Bootstrap
- Resampling with replacement is conducted at the **matched-pair level**:
  Draw $N_{\text{pairs}}$ pairs with replacement: $\{(u_j^*, v_j^*)\}_{j=1}^{N_{\text{pairs}}}$.
- Recompute bootstrap mean: $\bar{\Delta}^* = \frac{1}{N_{\text{pairs}}} \sum_{j=1}^{N_{\text{pairs}}} \Delta_{\text{pair}, j}^*$.
- Execute $B = 2,000$ iterations to construct empirical bootstrap standard errors $\widehat{\text{SE}}_{\text{boot}}$ and $95\%$ percentile / BCa confidence intervals.
- Non-parametric Wilcoxon signed-rank tests are computed alongside paired Student's $t$-tests.
- Multi-seed Jaccard stability $J(S_a, S_b)$ is computed across seeds to measure explainer continuous-mask optimization dispersion.

### 6.3 Graph-Dependence Limitation Disclosure
- Transactions in the validation snapshot are embedded in a connected transaction graph within each time step.
- While matching controls for prediction margin and computation subgraph size ($E_{\text{cand}}$), matched nodes may share overlapping 2-hop neighborhoods or historical predecessors.
- Standard i.i.d. assumptions represent an asymptotic approximation. The pair cluster bootstrap accounts for matched pairing variance, but unmodeled graph autocorrelation remains an epistemic limitation of transductive graph benchmarking.

---

## 7. Precision Planning (MDES as a Function of SD)

To avoid unsupported empirical claims prior to measuring effect variance in the main period, statistical precision is formalized strictly as a function of the unknown population standard deviation $s_{\Delta}$:
$$\text{MDES} = (t_{\alpha/2, \text{df}} + t_{\beta, \text{df}}) \cdot \frac{s_{\Delta}}{\sqrt{N_{\text{pairs}}}} = c \cdot s_{\Delta}$$
evaluated at $\alpha = 0.05$ (two-sided) and $80\%$ statistical power ($1 - \beta = 0.80$):

| Estimand Register | Matched Pairs ($N$) | Degrees of Freedom | $t_{\alpha/2}$ ($\alpha=0.05$) | $t_{\beta}$ ($1-\beta=0.80$) | Standardized Precision Coefficient ($c$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`ETGNN_full` TP–FP Primary** | **95** | 94 | 1.9855 | 0.8447 | **$\text{MDES} = 0.2903 \cdot s_{\Delta}$** |
| **`ETGNN_full` TN–FN Primary** | **25** | 24 | 2.0639 | 0.8569 | **$\text{MDES} = 0.5842 \cdot s_{\Delta}$** |
| **`Matched_GAT` TP–FP Primary** | **94** | 93 | 1.9858 | 0.8448 | **$\text{MDES} = 0.2918 \cdot s_{\Delta}$** |
| **`Matched_GAT` TN–FN Primary** | **21** | 20 | 2.0860 | 0.8599 | **$\text{MDES} = 0.6428 \cdot s_{\Delta}$** |
| **`Degree Sensitivity` ETGNN TP–FP** | **57** | 56 | 2.0032 | 0.8493 | **$\text{MDES} = 0.3778 \cdot s_{\Delta}$** |
| **`Degree Sensitivity` ETGNN TN–FN** | **25** | 24 | 2.0639 | 0.8569 | **$\text{MDES} = 0.5842 \cdot s_{\Delta}$** |
| **`Degree Sensitivity` Matched GAT TP–FP** | **82** | 81 | 1.9897 | 0.8460 | **$\text{MDES} = 0.3131 \cdot s_{\Delta}$** |
| **`Degree Sensitivity` Matched GAT TN–FN** | **21** | 20 | 2.0860 | 0.8599 | **$\text{MDES} = 0.6428 \cdot s_{\Delta}$** |

### Methodological Interpretation:
1. TP–FP contrasts achieve high resolution ($\text{MDES} \le 0.292 \cdot s_{\Delta}$).
2. Due to the scarcity of illicit transactions, TN–FN contrasts achieve moderate resolution ($\text{MDES} \sim 0.58\text{--}0.64 \cdot s_{\Delta}$).
3. In accordance with pre-registration, failures to reject the null on TN–FN will be reported with explicit reference to these bounds rather than claiming evidence of absent effect.

---

## 8. Strict Freezing & Review Deliverables

> [!CAUTION]
> **Execution Boundary:**
> - The technical pilot benchmark has been updated, executed, and audited on the 4 development nodes (time steps 33–36).
> - All main-run configuration artifacts ([`configs/main_c1_c2_run_config.json`](file:///d:/ETGNN-Fraud-Detection/configs/main_c1_c2_run_config.json), sample IDs, pair registers, power tables) are frozen.
> - **Execution is halted.** Main-period deletions (time steps 37–40) will NOT be launched until formal approval is granted.
> - Models will not be retrained; test period (steps 41–49) remains completely sealed.
