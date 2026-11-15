# Dated Frozen Protocol Amendment (v2.1): Comparative Evaluation of GNN Explanations (C1 & C2)

**Document Version:** 2.1 (Fully Verified Automated Balance Tables & Assertion Audits)  
**Date of Audit & Freezing:** October 20, 2026  
**Status:** Protocol Frozen Prior to Experimental Execution  
**Target Architectures:** Full-Feature ETGNN (`etgnn.pt`) vs. Aligned No-Time Matched GAT (`matched_gat.pt`)  
**Secondary Baseline:** Local-Feature ETGNN (`etgnn_local.pt`)  
**Scope:** Replaces the unadjusted 80-node descriptive plan with the matched comparative designs C1 and C2.

---

## 1. Executive Summary & Design Alignment

This amended protocol establishes the formal methodological specifications for two comparative evaluations (**C1** and **C2**) on the Elliptic Bitcoin dataset:
1. **Contrast 1 (C1: Between-Model Same-Perturbation Contrast):** Compares `etgnn.pt` against `matched_gat.pt` on the **exact same transactions**, **same candidate-edge universe**, **same removed-edge sets**, **same budget**, and a **common target class** (the frozen ETGNN original predicted class).
2. **Contrast 2 (C2: Within-Model Caliper-Matched Contrast):** Within each model, pairs incorrect predictions (False Positives and False Negatives) with balanced correct predictions (True Positives and True Negatives) using **caliper-constrained 1-to-1 matching without replacement** on prediction margin ($|p - 0.5|$) and candidate edge count ($E_{\text{cand}}$). Contrast is measured as the explanation advantage difference:
   $$\Delta_{\text{pair}} = A(u_{\text{incorrect}}) - A(v_{\text{matched correct}})$$
   where $A(v) = \bar{D}_{\text{GE}}(v) - \bar{D}_{\text{Rnd}}(v)$.

---

## 2. Checkpoint Provenance & Exact C1 Alignment

### 2.1 Confirmation of the Primary C1 Model Pair
The primary C1 evaluation is strictly between:
1. **`etgnn.pt`** (Full-feature temporal model)
2. **`matched_gat.pt`** (Full-feature no-time baseline)

Both checkpoints reside in `results/checkpoints/` and have been verified:

| Attribute | `etgnn.pt` | `matched_gat.pt` | Verification Status |
| :--- | :--- | :--- | :--- |
| **File Path** | [`results/checkpoints/etgnn.pt`](file:///d:/ETGNN-Fraud-Detection/results/checkpoints/etgnn.pt) | [`results/checkpoints/matched_gat.pt`](file:///d:/ETGNN-Fraud-Detection/results/checkpoints/matched_gat.pt) | Verified on disk |
| **File Size** | 50,929 bytes | 51,333 bytes | Verified on disk |
| **SHA-256 Hash** | `b2f5535f3b6b1675109dae205934e761d2ad1b80b5409564bbe5889edb79dadc` | `1e9b43ea1cc25a26dd8bbe953c49a8782a0d9c27b5e0da17afe726fd21405074` | Cryptographically verified |
| **Feature Set** | Full 165 features (93 local + 72 neighborhood) | Full 165 features (identical ordering & unscaled raw features) | Verified identical |
| **Time Handling** | 16-dim sinusoidal encoding: $\sin(t/10000^{2i/16}), \cos(t/10000^{2i/16})$ | 16-dim zero vector: `torch.zeros((x.shape[0], 16))` | Verified in forward pass |
| **Input Dims** | 165 features + 16 time = 181 dimensions | 165 features + 16 zero = 181 dimensions | Verified in layer weights |
| **Architecture** | Layer 1: GATConv(181, 16, heads=4); Layer 2: GATConv(64, 2, heads=1) | Layer 1: GATConv(181, 16, heads=4); Layer 2: GATConv(64, 2, heads=1) | Verified identical |
| **Dropout** | $p = 0.3$ | $p = 0.3$ | Verified identical |
| **Training Seed** | `torch.manual_seed(42)` | `torch.manual_seed(42)` | Verified in training code |
| **Optimizer** | Adam ($\text{lr}=0.01$, $\text{weight\_decay}=5\times 10^{-4}$) | Adam ($\text{lr}=0.01$, $\text{weight\_decay}=5\times 10^{-4}$) | Verified identical |
| **Class Weights** | $w = [1.0, N_{\text{licit}} / N_{\text{illicit}}]$ | $w = [1.0, N_{\text{licit}} / N_{\text{illicit}}]$ | Verified identical |
| **Stopping Rule** | Best validation AUPRC (patience = 8 epochs) | Best validation AUPRC (patience = 8 epochs) | Verified identical |
| **Decision Rule** | Frozen threshold $\tau = 0.50$ | Frozen threshold $\tau = 0.50$ | Verified identical |

### 2.2 Epistemic Limitations & Provenance Boundary
> [!WARNING]
> **Conditional Inference Disclosure:**
> - There is exactly **one checkpoint per model** on disk (`etgnn.pt` and `matched_gat.pt`). No training-seed replication checkpoints (e.g. seeds 43, 44) exist in the repository.
> - Consequently, all empirical findings are strictly **conditional on these specific checkpoint instances**. We do not claim to have demonstrated an isolated, universal causal effect of time encoding across random model initializations.

### 2.3 Status of Local ETGNN (`etgnn_local.pt`)
- `etgnn_local.pt` was trained on **93 local features** (+ 16 time dims = 109 dims; SHA-256: `2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082`).
- No aligned 93-feature GAT baseline checkpoint (`gat_local.pt` or `matched_gat_local.pt`) exists in the repository.
- Therefore, **`etgnn_local.pt` is strictly excluded from between-model C1 contrasts** to avoid confounding temporal encoding with feature dimensionality (93 vs 165). It is retained solely as an independent within-model C2 comparator to check consistency with the historical pilot.

---

## 3. Physical Graph Verification & Candidate Edge Rationale

### 3.1 Verification Across All 161,603 Graph Positions
- **Identity of Indices:** Exactly 0 positions deviate between `val_graph["original_node_index"]` and graph position tensor ($161,603$ identical matches).
- **Transaction ID Mapping:** Verified 1-to-1 against [`data/processed/node_mapping.csv`](file:///d:/ETGNN-Fraud-Detection/data/processed/node_mapping.csv).
- **Historical Pilot Exclusions:** All 20 historical pilot nodes were verified by both `node_index` and `txId` (10 in dev steps 33–36, 10 in main steps 37–40) and are confirmed excluded.
- **Physical Self-Loops:** `contains_self_loops(val_graph["edge_index"]) == False` (exactly 0 physical self-loops).
- **Subgraph Numerical Reproducibility:** $\max |p_{\text{full}} - p_{\text{subgraph}}| \le 1.79 \times 10^{-7}$ (verified across 50 nodes, passing the $10^{-5}$ tolerance).

### 3.2 Candidate Edge Universe & The $E_{\text{cand}} \ge 2$ Rationale
The deletion budget is computed as:
$$k = \max(1, \lceil \alpha \cdot E_{\text{cand}} \rceil)$$
The requirement $E_{\text{cand}} \ge 2$ is enforced due to ranking non-discriminability, not ceiling rounding:
- For **$E_{\text{cand}} = 1$**:
  - Nominal budget $\alpha = 0.25 \implies k = \lceil 0.25 \times 1 \rceil = 1$ ($100\%$ edge removal).
  - Nominal budget $\alpha = 0.10 \implies k = \lceil 0.10 \times 1 \rceil = 1$ ($100\%$ edge removal).
  - Nominal budget $\alpha = 0.50 \implies k = \lceil 0.50 \times 1 \rceil = 1$ ($100\%$ edge removal).
  - **Ranking Non-Discriminability:** When $E=1$, there is only a single edge in the computation graph. Every ranking method (GNNExplainer, Random, Degree, Occlusion) selects the exact same edge. Hence, $E=1$ is mathematically incapable of distinguishing explanation quality.
- For small candidate sets, discrete ceiling rounding alters the effective fraction:

| Candidate Edges $E_{\text{cand}}$ | $\alpha = 0.10$ ($k$, actual %) | $\alpha = 0.25$ ($k$, actual %) | $\alpha = 0.50$ ($k$, actual %) | Ranking Discriminability |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $k=1$ (100.0%) | $k=1$ (100.0%) | $k=1$ (100.0%) | **None (all methods pick edge 1)** |
| **2** | $k=1$ (50.0%) | $k=1$ (50.0%) | $k=1$ (50.0%) | **Binary choice (picks 1 of 2)** |
| **3** | $k=1$ (33.3%) | $k=1$ (33.3%) | $k=2$ (66.7%) | Substantive ranking choice |
| **4** | $k=1$ (25.0%) | $k=1$ (25.0%) | $k=2$ (50.0%) | Substantive ranking choice |
| **5** | $k=1$ (20.0%) | $k=2$ (40.0%) | $k=3$ (60.0%) | Budget-sensitive |
| **10** | $k=1$ (10.0%) | $k=3$ (30.0%) | $k=5$ (50.0%) | Fully continuous |

---

## 4. Nature of Graph Perturbation & C1 Contrast Specification

### 4.1 Frozen-Input Message-Passing Sensitivity
> [!IMPORTANT]
> **Nature of Deletion:**
> - In the Elliptic dataset, features 94–165 are pre-computed historical neighborhood aggregations.
> - When candidate edges $\Delta E$ are deleted from `edge_index`, the node feature matrix $X$ is **NOT recomputed**. Features 94–165 remain static at their full-graph values.
> - Therefore, C1 measures **frozen-input message-passing deletion sensitivity**, not the ablation of all historical topological signal.

### 4.2 Shared Deletion & Baseline Contrasts
For each transaction $v$:
1. **Shared ETGNN-Ranked Deletion:**
   - Top-$k$ edges $\Delta E_{\text{GE}}$ are selected by the frozen ETGNN GNNExplainer edge ranking.
   - The **exact same edge set** $\Delta E_{\text{GE}}$ is removed from both `etgnn.pt` and `matched_gat.pt`.
2. **Shared Random Deletion Reference:**
   - For each draw $r \in \{1, \dots, 30\}$, a random edge set $\Delta E_{\text{rnd}}^{(r)}$ of size $k$ is drawn using a fixed seed.
   - The **exact same random edge set** is removed from both models.
3. **Common Target Class:**
   - Probability drops for both models are evaluated strictly with respect to ETGNN's original predicted class $\hat{y}^* = \hat{y}_{\text{ETGNN}}(v)$:
     $$D_{\text{model}}(v, \Delta E) = p_{\text{model}}(Y = \hat{y}^* \mid G) - p_{\text{model}}(Y = \hat{y}^* \mid G \setminus \Delta E)$$
4. **Explainer Seed Aggregation:**
   - Seeds 42, 43, 44 are **averaged at the node level FIRST**:
     $$\bar{D}_{\text{GE}}(v) = \frac{1}{3} \sum_{s=1}^3 D_{\text{GE}}(v, s), \quad A(v) = \bar{D}_{\text{GE}}(v) - \bar{D}_{\text{Rnd}}(v)$$
   - Seeds are not treated as independent observations in hypothesis testing.

---

## 5. Caliper-Constrained Matching Specification & Mathematical Properties

### 5.1 Formal Matching Rules
Matching is performed 1-to-1 without replacement separately for TP–FP and TN–FN contrasts:
1. **Covariates:** $\mathbf{X} = [\text{margin} = |p - 0.5|, \quad E_{\text{cand}}]$.
2. **Standardized Euclidean Distance:**
   $$d(i, j) = \sqrt{\left(\frac{\text{margin}_i - \text{margin}_j}{s_{\text{pooled, pre}}^{\text{margin}}}\right)^2 + \left(\frac{E_i - E_j}{s_{\text{pooled, pre}}^{E}}\right)^2}$$
3. **Constant Pre-Match Denominator for SMD:**
   $$s_{\text{pooled, pre}} = \sqrt{\frac{s_{\text{treat, pre}}^2 + s_{\text{ctrl, pre}}^2}{2}}$$
4. **Admissibility Calipers:**
   A candidate pair $(i, j)$ is admissible if and only if:
   $$|\text{margin}_i - \text{margin}_j| \le 0.05 \quad \text{AND} \quad |E_{\text{cand}, i} - E_{\text{cand}, j}| \le 5$$
5. **Exact Maximum Cardinality Verification:**
   The matching cardinality of our linear sum assignment implementation was formally proven to match the exact mathematical maximum cardinality computed by the **Hopcroft-Karp algorithm** (`scipy.sparse.csgraph.maximum_bipartite_matching`).

### 5.2 Assertion Verification Results
All row-by-row data assertions were executed via [`src/evaluation/generate_verified_report_tables.py`](file:///d:/ETGNN-Fraud-Detection/src/evaluation/generate_verified_report_tables.py):

| Model | Contrast | Available Treat | Available Ctrl | Matched Pairs | Hopcroft-Karp Max Card | Treat Nodes Unique | Ctrl Nodes Unique | Caliper Margin Max Diff | Caliper Edges Max Diff | Degree Identity Verified |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ETGNN_full`** | **TP_FP** | 250 | 99 | **95** | 95 | 95 (100%) | 95 (100%) | **0.0470** | **2** | **Passed** |
| **`ETGNN_full`** | **TN_FN** | 26 | 1,312 | **25** | 25 | 25 (100%) | 25 (100%) | **0.0074** | **1** | **Passed** |
| **`Matched_GAT`** | **TP_FP** | 486 | 104 | **94** | 94 | 94 (100%) | 94 (100%) | **0.0500** | **3** | **Passed** |
| **`Matched_GAT`** | **TN_FN** | 21 | 1,076 | **21** | 21 | 21 (100%) | 21 (100%) | **0.0030** | **1** | **Passed** |
| **`ETGNN_local`** | **TP_FP** | 177 | 100 | **92** | 92 | 92 (100%) | 92 (100%) | **0.0499** | **4** | **Passed** |
| **`ETGNN_local`** | **TN_FN** | 25 | 1,385 | **25** | 25 | 25 (100%) | 25 (100%) | **0.0087** | **4** | **Passed** |

*Verification Summary:*
- Zero repeated nodes in any register.
- Treat and control node sets are completely disjoint.
- Strict unrounded calipers satisfied on 100% of matched pairs.
- Total degree identity: $\text{total\_degree} = \text{in\_degree} + \text{out\_degree}$ confirmed row-by-row with 0 violations across all 7,653 rows of every feasibility CSV and matched pair register.

---

## 6. Empirical Matching Balance & Diagnostic Evidence

### 6.1 Explanation of Numbers Changed from Previous Draft
In the previous v2.0 text, the balance table suffered from a transcription error where intermediate unconstrained numbers (`in_degree` 1.222, `out_degree` 0.343, `total_degree` 2.566) were erroneously copied into the markdown table instead of the true CSV outputs. In this v2.1 document, **all table values are generated directly by script from the verified CSV records**, with the exact identity:
$$\text{mean}(\text{total\_degree}) = \text{mean}(\text{in\_degree}) + \text{mean}(\text{out\_degree})$$
verified to within $10^{-6}$ for every model, contrast, and group.

### 6.2 Verified Balance Table (Generated Automatically from CSV Data)

| Model | Contrast | Covariate | Pre Pooled SD | Pre Treat Mean | Pre Ctrl Mean | Pre SMD | Post Treat Mean | Post Ctrl Mean | Post SMD | Pre VR | Post VR |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ETGNN_full`** | **TP_FP** | **margin** | 0.1222 | 0.257 | 0.325 | **-0.561** | 0.308 | 0.319 | **-0.083** | 1.036 | 0.852 |
| | | **candidate_edge_count** | 17.0812 | 7.912 | 2.687 | **+0.306** | 2.021 | 2.074 | **-0.003** | 14.509 | 0.157 |
| | | **in_degree** | 9.2574 | 3.988 | 1.525 | **+0.266** | 1.063 | 1.000 | **+0.007** | 5.275 | $\infty$ |
| | | **out_degree** | 0.5434 | 1.012 | 0.323 | **+1.267** | 0.990 | 0.337 | **+1.201** | 1.447 | 1.076 |
| | | **total_degree** | 9.1957 | 5.000 | 1.849 | **+0.343** | 2.053 | 1.337 | **+0.078** | 5.215 | 1.324 |
| **`ETGNN_full`** | **TN_FN** | **margin** | 0.1415 | 0.230 | 0.422 | **-1.355** | 0.226 | 0.226 | **-0.002** | 1.356 | 0.987 |
| | | **candidate_edge_count** | 17.8633 | 5.192 | 7.421 | **-0.125** | 3.000 | 2.800 | **+0.011** | 0.247 | 1.059 |
| | | **in_degree** | 7.0077 | 1.769 | 3.361 | **-0.227** | 1.680 | 1.920 | **-0.034** | 0.012 | 0.490 |
| | | **out_degree** | 1.3675 | 0.538 | 1.255 | **-0.524** | 0.520 | 1.080 | **-0.410** | 0.154 | 1.561 |
| | | **total_degree** | 7.1647 | 2.308 | 4.617 | **-0.322** | 2.200 | 3.000 | **-0.112** | 0.021 | 1.000 |
| **`Matched_GAT`** | **TP_FP** | **margin** | 0.1088 | 0.176 | 0.278 | **-0.935** | 0.249 | 0.264 | **-0.129** | 0.571 | 0.800 |
| | | **candidate_edge_count** | 17.2327 | 8.212 | 3.394 | **+0.280** | 2.415 | 2.277 | **+0.008** | 7.811 | 1.417 |
| | | **in_degree** | 9.4914 | 4.327 | 1.606 | **+0.287** | 1.383 | 1.085 | **+0.031** | 5.902 | 3.920 |
| | | **out_degree** | 0.6170 | 0.969 | 0.337 | **+1.025** | 0.957 | 0.362 | **+0.966** | 2.110 | 1.849 |
| | | **total_degree** | 9.4638 | 5.296 | 1.942 | **+0.354** | 2.340 | 1.447 | **+0.094** | 5.860 | 2.061 |
| **`Matched_GAT`** | **TN_FN** | **margin** | 0.1815 | 0.285 | 0.399 | **-0.630** | 0.285 | 0.285 | **+0.000** | 1.713 | 1.000 |
| | | **candidate_edge_count** | 16.0215 | 2.286 | 7.178 | **-0.305** | 2.286 | 2.286 | **+0.000** | 0.001 | 0.672 |
| | | **in_degree** | 6.4299 | 1.429 | 3.071 | **-0.255** | 1.429 | 1.381 | **+0.007** | 0.010 | 0.747 |
| | | **out_degree** | 1.4721 | 0.524 | 1.328 | **-0.546** | 0.524 | 0.809 | **-0.194** | 0.149 | 3.471 |
| | | **total_degree** | 6.6316 | 1.952 | 4.399 | **-0.369** | 1.952 | 2.191 | **-0.036** | 0.020 | 1.646 |
| **`ETGNN_local`** | **TP_FP** | **margin** | 0.0943 | 0.147 | 0.218 | **-0.753** | 0.193 | 0.209 | **-0.172** | 0.769 | 0.740 |
| | | **candidate_edge_count** | 9.4530 | 6.237 | 2.810 | **+0.363** | 2.707 | 2.217 | **+0.052** | 3.793 | 4.331 |
| | | **in_degree** | 4.7127 | 2.915 | 1.580 | **+0.283** | 1.576 | 1.065 | **+0.108** | 0.638 | 4.884 |
| | | **out_degree** | 0.6369 | 0.932 | 0.310 | **+0.977** | 1.033 | 0.304 | **+1.144** | 2.433 | 2.929 |
| | | **total_degree** | 4.7141 | 3.848 | 1.890 | **+0.415** | 2.609 | 1.370 | **+0.263** | 0.641 | 2.942 |
| **`ETGNN_local`** | **TN_FN** | **margin** | 0.1256 | 0.189 | 0.393 | **-1.627** | 0.189 | 0.189 | **-0.003** | 0.428 | 1.006 |
| | | **candidate_edge_count** | 18.6872 | 4.800 | 7.661 | **-0.153** | 4.800 | 4.960 | **-0.009** | 0.236 | 0.871 |
| | | **in_degree** | 7.6358 | 1.560 | 3.531 | **-0.258** | 1.560 | 3.160 | **-0.210** | 0.008 | 0.011 |
| | | **out_degree** | 1.3337 | 0.600 | 1.253 | **-0.489** | 0.600 | 1.080 | **-0.360** | 0.164 | 1.220 |
| | | **total_degree** | 7.7522 | 2.160 | 4.784 | **-0.338** | 2.160 | 4.240 | **-0.268** | 0.015 | 0.020 |

---

## 7. Honest Disclosure of Out-Degree Imbalance & Degree-Balanced Sensitivity Match

### 7.1 Empirical Limitation Disclosure: Out-Degree Imbalance
The balance table demonstrates that while `margin`, `candidate_edge_count`, and `in_degree` achieve post-match balance ($|\text{SMD}| \le 0.083$ for `ETGNN_full`), **`out_degree` exhibits a persistent empirical imbalance** ($\text{post SMD} = +1.201$ for TP–FP):
- *Empirical Limitation:* Candidate edge extraction focuses on directed incoming computation subgraphs (`flow="source_to_target"`). An empirical asymmetry exists in the dataset between True Positives (mean out-degree = $0.323$) and False Positives (mean out-degree = $1.012$). Out-degree is not constrained by the primary matching specification.
- *Primary Estimand Choice:* The primary estimand matches on `margin` and `candidate_edge_count` because GNNExplainer deletions operate on the incoming computation edges. **We explicitly disclaim that degree has been controlled by the primary match**, and document the out-degree difference strictly as an observed structural limitation of the primary design.

### 7.2 Degree-Balanced Sensitivity Match (Pre-Specified Sensitivity Estimand)
To quantify sensitivity to out-degree imbalance without relying on deletion outcomes, an alternative **degree-balanced sensitivity match** was computed adding an exact out-degree matching restriction ($|\Delta \text{out\_degree}| = 0$):

| Model | Contrast | Matched Pairs | Treat Coverage | Ctrl Coverage | Post SMD Margin | Post SMD Edges | Post SMD Out-Degree | Post SMD Total-Degree |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ETGNN_full`** | **TP_FP** | **57** | 22.8% | 57.6% | **-0.002** | **+0.016** | **+0.000** | **+0.031** |
| **`ETGNN_full`** | **TN_FN** | **25** | 96.2% | 1.9% | **-0.017** | **+0.029** | **+0.000** | **+0.022** |
| **`Matched_GAT`** | **TP_FP** | **82** | 16.9% | 78.8% | **-0.114** | **+0.010** | **+0.000** | **+0.035** |
| **`Matched_GAT`** | **TN_FN** | **21** | 100.0% | 2.0% | **-0.001** | **-0.003** | **+0.000** | **+0.007** |
| **`ETGNN_local`** | **TP_FP** | **72** | 40.7% | 72.0% | **-0.181** | **+0.088** | **+0.000** | **+0.153** |
| **`ETGNN_local`** | **TN_FN** | **25** | 100.0% | 1.8% | **+0.002** | **+0.000** | **+0.000** | **-0.222** |

*Interpretation:* Requiring exact out-degree balance drops TP–FP coverage from 95 pairs to 57 pairs for `ETGNN_full`, but achieves **zero out-degree bias ($\text{SMD} = 0.000$)** and perfect balance on all degree dimensions ($\text{total\_degree SMD} = +0.031$). Both primary and sensitivity pair files are preserved for post-experimental reporting.

---

## 8. Technical Pilot Plan (Development Steps 33–36 Only)

### 8.1 Runtime Disclosure
> [!NOTE]
> **Runtime Status:** Execution runtime on target hardware has **not yet been measured**. Previous extrapolations are superseded by this empirical technical pilot plan.

### 8.2 Technical Pilot Scope
- **Purpose:** Benchmark execution latency, CPU memory, and GPU VRAM footprint on hardware before deploying large evaluation jobs.
- **Explicit Framing:** The 4 benchmark nodes serve purely as a technical runtime profile. **They are NOT presented as a matched-pair effect-variance pilot**, nor will development-period outcomes be used to estimate effect sizes or stopping rules.
- **Location:** Strictly restricted to development time steps **33–36**.
- **Sample Selection:** 4 isolated validation nodes from steps 33–36 (disjoint from historical pilot nodes):
  - TP: `node_index` 131350 (`txId` 93564753, step 33, $E=2$, margin=0.351)
  - FP: `node_index` 130659 (`txId` 92512070, step 33, $E=118$, margin=0.233)
  - FN: `node_index` 134593 (`txId` 288474786, step 34, $E=2$, margin=0.052)
  - TN: `node_index` 130674 (`txId` 92724813, step 33, $E=6$, margin=0.368)
- **Script:** [`src/evaluation/technical_pilot_plan.py`](file:///d:/ETGNN-Fraud-Detection/src/evaluation/technical_pilot_plan.py).
- **Target Sample File:** [`experiments/matched_c1_c2_audit/technical_pilot_sample.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/technical_pilot_sample.csv).
- **Execution Boundary:** Prepared and verified, but **NOT EXECUTED**.

---

## 9. Review Deliverables & Updated Bundle

The complete audit evidence is packaged in:
- **ZIP File:** [`experiments/matched_c1_c2_audit/C1_C2_Audit_Review_Bundle.zip`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/C1_C2_Audit_Review_Bundle.zip) (4.06 MB)

**Contents of Bundle:**
1. **Scripts:** `run_c1_c2_audit_v2.py`, `generate_verified_report_tables.py`, `technical_pilot_plan.py`
2. **Protocol Amendment:** `reports/c1_c2_protocol_amendment.md` (v2.1)
3. **Assertion & Diagnostics Tables:**
   - `assertion_verification_results.csv`
   - `verified_caliper_balance_diagnostics.csv`
   - `verified_caliper_within_pair_differences.csv`
   - `degree_sensitivity_balance_summary.csv`
   - `automated_report_tables.md`
4. **Feasibility CSVs:** `feasibility_etgnn_full.csv`, `feasibility_matched_gat.csv`, `feasibility_etgnn_local.csv`
5. **Primary Caliper Matched Pairs:**
   - `caliper_pairs_etgnn_full_tp_fp.csv` (95 pairs)
   - `caliper_pairs_etgnn_full_tn_fn.csv` (25 pairs)
   - `caliper_pairs_matched_gat_tp_fp.csv` (94 pairs)
   - `caliper_pairs_matched_gat_tn_fn.csv` (21 pairs)
   - `caliper_pairs_etgnn_local_tp_fp.csv` (92 pairs)
   - `caliper_pairs_etgnn_local_tn_fn.csv` (25 pairs)
6. **Degree Sensitivity Matched Pairs:**
   - `degree_sensitivity_pairs_etgnn_full_tp_fp.csv` (57 pairs)
   - `degree_sensitivity_pairs_etgnn_full_tn_fn.csv` (25 pairs)
   - `degree_sensitivity_pairs_matched_gat_tp_fp.csv` (82 pairs)
   - `degree_sensitivity_pairs_matched_gat_tn_fn.csv` (21 pairs)
7. **Plots:** Caliper overlap scatter plots and marginal distribution histograms.

---

## 10. Strict STOP Enforcement

Execution is halted. No main deletion experiments, GNNExplainer optimization runs, or technical pilot executions have been started. All notebook utilities for batch checkpointing, resume state, and backup are preserved.
