# Validation Feasibility Pilot Report (Version 2 - Evidence Reconciled)
## Preliminary Exploratory Audit of Edge-Attribution Perturbation Responses

**Versioned Directory**: `experiments/pilot_validation_audit_v2/`  
**Initial Recording**: 2026-10-19 20:55:51 UTC (`experiments/pilot_validation_audit/`)  
**Evidence Audit & Reconciliation**: 2026-10-20  
**Evaluated Architecture**: Local-Feature Time-Encoded GAT (`ETGNNClassifier`, input dimension 93, hidden dimension 64, 4 attention heads, dropout 0.3)  
**Evaluated Checkpoint**: `results/checkpoints/etgnn_local.pt` (SHA256: `2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082`)  
**Data Evaluated**: Validation graph snapshot `data/processed/validation_graph.pt` (Time steps 33–40 only)  
**Test Data Status**: Time steps 41–49 represent **previously inspected test data** from prior baseline evaluation. **No new test inference, hyperparameter tuning, or retraining was performed.** Independent verification scripts access frozen test arrays strictly for read-only numerical consistency audits.

---

## 1. Executive Summary of Reconciled Findings

1. **Deletion Drop Response**: Across all 20 pilot nodes, deleting the top 25% GNNExplainer edges produced a mean signed drop of **+0.1216 $\pm$ 0.0973** (macro) in original-class confidence, exceeding uniform-random edge deletion (**+0.0087 $\pm$ 0.0271**) and source out-degree ranking (**-0.0062 $\pm$ 0.1237**). Exhaustive single-edge occlusion ranking produced a drop of **+0.2027 $\pm$ 0.0954**.
2. **False Positive Correction (Repeated-Seed Analysis)**:
   - On the 5 selected False Positive transactions, deleting top GNNExplainer edges flipped the predicted class from illicit to licit in **13 of 15 repeated-seed runs** (86.7%).
   - At the transaction level: **5 of 5** transactions showed correction in at least one seed (any-seed: 100%), **4 of 5** showed correction in at least two seeds (majority-seed: 80.0%), and **4 of 5** showed correction across all three seeds (all-seed: 80.0%).
   - Transaction 154421 corrected only under seed 43 (prob dropped from 0.645 to 0.487), while failing to cross the threshold under seeds 42 (0.588) and 44 (0.565).
3. **True Negative Error Induction**:
   - Deleting explanation edges induced unwanted classification errors on True Negatives in **3 of 15 seed runs** (20.0%), concentrated entirely on Node 148195 across all three seeds (licit prob dropped from 0.782 to ~0.444).
   - In comparison, Single-Edge Occlusion induced errors in **40.0%** of TN cases (2/5) and **20.0%** of TP cases (1/5). Uniform-random deletion induced **0.0%** errors on TN and TP.
4. **Optimization Instability vs. Random Reference**:
   - Across three explainer seeds (42, 43, 44), mean pairwise Jaccard similarity of selected edges was **0.3702** (ranging from **0.2041** on TP to **0.5470** on TN).
   - In comparison, the exact expected random overlap under finite-sample hypergeometric sampling is **0.1547** (observed empirical random across 30 draws: **0.1545**).
   - While GNNExplainer stability is above random baseline, substantial stochastic variation remains, particularly on True Positives (0.2041, barely above the 0.1547 random reference).
5. **No Evidence of Causation or Population Generalizability**:
   - These results do **not** prove noisy-neighbor causation, global optimality, or population-wide accuracy improvements.
   - The observed asymmetry between FP correction (86.7%) and FN correction (20.0%) is heavily confounded by sample selection: FP nodes were located much closer to the decision threshold (mean margin = 0.0579) and had larger candidate edge sets (mean 108.8) than FN nodes (mean margin = 0.2831, mean candidate edges 31.8).

---

## 2. Experimental Setup and Candidate Edge Reconciliation

### 2.1 Reconciled Candidate Edge Count: Exactly 2,712
The candidate edges $E_{\mathrm{cand}}$ are defined as all edges within each transaction's 2-hop directed computation subgraph (`flow="source_to_target"`).

An earlier preliminary draft stated "2,544 candidate edges"; the comprehensive evidence audit confirms this was a manual text drafting addition error (diff: -168). The empirical sum across `pilot_nodes.csv`, the row count of `candidate_edges.csv`, and direct PyG graph extraction code all strictly confirm **2,712 candidate edges**:
- **True Positives (5 nodes)**: 233, 212, 90, 72, 73 = **680 edges**
- **False Positives (5 nodes)**: 147, 111, 153, 61, 72 = **544 edges**
- **False Negatives (5 nodes)**: 81, 45, 11, 16, 6 = **159 edges**
- **True Negatives (5 nodes)**: 367, 286, 363, 154, 159 = **1,329 edges**
- **Total across 20 nodes**: **2,712 candidate edges**

### 2.2 Equal Budget Rule
Every method strictly evaluated an identical candidate budget per node:
$$k = \max(1, \lceil 0.25 \times E_{\mathrm{cand}} \rceil)$$
All 700 perturbation trials in `raw_interventions.csv` strictly respect this equality constraint.

---

## 3. Detailed Empirical Results

### Table 1: Comparative Method Summary (Macro vs. Micro Pooled)

| Method | Macro Mean Drop ($\pm$ Std) | Pooled Mean Drop ($\pm$ Std) | FP Correction Rate (Runs) | FN Correction Rate (Runs) | TP Error Induction | TN Error Induction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNNExplainer (3 seeds)** | **+0.1216 $\pm$ 0.0973** | +0.1216 $\pm$ 0.0972 | **13/15 (86.7%)** | 3/15 (20.0%) | **0/15 (0.0%)** | 3/15 (20.0%) |
| **Single-Edge Occlusion** | **+0.2027 $\pm$ 0.0954** | +0.2027 $\pm$ 0.0954 | **5/5 (100.0%)** | 1/5 (20.0%) | 1/5 (20.0%) | 2/5 (40.0%) |
| **Uniform-Random (30 draws)** | +0.0087 $\pm$ 0.0271 | +0.0087 $\pm$ 0.0594 | 29/150 (19.3%) | 2/150 (1.3%) | 0/150 (0.0%) | 0/150 (0.0%) |
| **Source Out-Degree** | -0.0062 $\pm$ 0.1237 | -0.0062 $\pm$ 0.1237 | 0/5 (0.0%) | 0/5 (0.0%) | 0/5 (0.0%) | 0/5 (0.0%) |

### Table 2: False Positive Outcomes per Transaction and per Seed

| Node Index | Original $P(\mathrm{illicit})$ | Prediction Margin | Seed 42 $P_{\mathrm{del}}$ | Seed 42 Flip | Seed 43 $P_{\mathrm{del}}$ | Seed 43 Flip | Seed 44 $P_{\mathrm{del}}$ | Seed 44 Flip | Seeds Corrected | Any-Seed ( $\ge 1/3$ ) | Majority ( $\ge 2/3$ ) | All-Seed (3/3) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **130690** | 0.5248 | 0.0248 | 0.3102 | Yes | 0.3091 | Yes | 0.3203 | Yes | 3/3 | Yes | Yes | Yes |
| **130935** | 0.5724 | 0.0724 | 0.3246 | Yes | 0.3145 | Yes | 0.3278 | Yes | 3/3 | Yes | Yes | Yes |
| **133801** | 0.5329 | 0.0329 | 0.3834 | Yes | 0.3639 | Yes | 0.3829 | Yes | 3/3 | Yes | Yes | Yes |
| **136379** | 0.5144 | 0.0144 | 0.4149 | Yes | 0.4249 | Yes | 0.4508 | Yes | 3/3 | Yes | Yes | Yes |
| **154421** | 0.6453 | 0.1453 | 0.5883 | No | 0.4872 | Yes | 0.5646 | No | 1/3 | Yes | No | No |
| **Total** | | | | **4/5** | | **5/5** | | **4/5** | **13/15 (86.7%)** | **5/5 (100%)** | **4/5 (80%)** | **4/5 (80%)** |

### Table 3: Quadrant Distributional Confounders

| Quadrant | $N$ Nodes | Mean Margin $|P - 0.5|$ | Margin Range | Mean $E_{\mathrm{cand}}$ | $E_{\mathrm{cand}}$ Range | Mean In-Degree |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **False Positives (FP)** | 5 | **0.0579** | [0.0144, 0.1453] | 108.8 | [61, 153] | 52.2 |
| **False Negatives (FN)** | 5 | **0.2831** | [0.0481, 0.4864] | 31.8 | [6, 81] | 20.4 |
| **True Positives (TP)** | 5 | 0.2085 | [0.0696, 0.2699] | 136.0 | [72, 233] | 101.4 |
| **True Negatives (TN)** | 5 | 0.2745 | [0.2093, 0.3399] | 265.8 | [154, 367] | 154.0 |

---

## 4. Methodological Interpretation & Critical Limitations

1. **Non-Random Convenience Sampling**:
   The 20 pilot nodes were selected with an `incoming_degree >= 2` requirement and sorted by degree. This oversamples dense subgraph topologies and transactions with high structural visibility.
2. **Confounded Asymmetry**:
   The fact that 86.7% of FP seed runs flipped compared to 20.0% of FN runs cannot be claimed as an established intrinsic property of the architecture. The selected FP nodes were clustered within 0.058 of the 0.50 threshold, whereas FN nodes required massive probability shifts ($\sim$0.283) to cross the threshold.
3. **Continuous Mask Instability**:
   The mean Jaccard stability of 0.3702 confirms that continuous-mask GNNExplainer optimization exhibits high stochasticity across seeds. Edge attribution must be reported with multi-seed error bounds.
4. **Descriptive, Non-Causal Nature**:
   Edge deletion perturbation measures localized response in a frozen model; it does not establish ground-truth causal transaction flows or prove noisy-neighbor causation in real-world Bitcoin networks.
