# Phase 4 Controlled Experiment Results: Incremental Relational Value Benchmark (Programmatically Verified)

**Document Status**: Official Research Extension Deliverable (Programmatically Regenerated & Audited in Phase B)  
**Repository**: `ETGNN-Fraud-Detection`  
**Execution Record**: Exactly 40/40 Controlled Trials Completed (`task-565`, runtime: 26.78 min)  
**Authoritative Data Source**: [`research_extension/PHASE4_EXPERIMENT_MATRIX.csv`](file:///d:/ETGNN-Fraud-Detection/research_extension/PHASE4_EXPERIMENT_MATRIX.csv) and `results/phase4_experiments/tables/*.json` (100% programmatic parity verified)  
**Strict Safeguards**: Chronological training steps 1–32 ($28,938$ labelled nodes: $25,536$ licit, $3,402$ illicit); Validation steps 33–40 ($7,653$ labelled nodes: $7,034$ licit, $619$ illicit). Historical test steps 41–49 unaccessed during Phase 4.

---

## 1. Executive Summary of Audited Findings

Phase 4 evaluated the controlled factorial question:
*Under a matched ego-residual Graph Attention Network (`ResidualETGNNClassifier`), does the true intra-timestep Bitcoin transaction graph in Elliptic provide incremental validation predictive value relative to degree-preserving rewired controls, empty-edge controls, and non-graph Deep MLPs?*

### The Audited Headline Findings:
1. **On the Local 93-Feature Representation (No Precomputed Aggregates)**:
   - **Exploratory Topological Stabilization Over Rewired Controls**: Condition B (`ResidualETGNNClassifier`, True Graph) achieved **$0.7758 \pm 0.0111$** validation AUPRC vs. **$0.7252 \pm 0.0420$** for Condition D (Degree-Preserving Rewired Graph).
   - The seed-paired gain ($ar{\Delta} = \mathbf{+0.0506 \pm 0.0322}$, $95\% 	ext{ CI } [+0.0107, +0.0906]$) is positive across **5 out of 5 seeds** with an unadjusted paired $t$-test $p = 0.0245$ (Cohen's $d_z = 1.57$). However, after **Holm–Bonferroni multiple-testing correction** across the 6 primary factorial contrasts, the adjusted p-value is **$p_{	ext{Holm}} = 0.1223$** ($p_{	ext{Holm}} = 0.0734$ within the 3-contrast 93f family), and the lower 95% confidence bound ($+0.0107$) falls below the pre-registered $+0.020$ practical significance threshold.
   - **Inconclusive Relative to Empty Edges and Deep MLP on AUPRC**: Condition B ($0.7758 \pm 0.0111$) did not statistically outperform the parameter-matched Empty-Edge control (Condition C: **$0.6938 \pm 0.1669$**, unadjusted $p = 0.3297$) or the feature-only Deep MLP (Condition A: **$0.7616 \pm 0.0563$**, unadjusted $p = 0.6503$). Therefore, under pre-registered Category 1 rules (which required statistically beating both Rewired and Empty controls), the topological advantage is classified as **exploratory internal evidence of optimization stabilization**, not confirmatory superiority over non-graph models.
   - Condition B achieved the lowest cross-seed AUPRC standard deviation ($\sigma = 0.0111$, $5	imes$ lower than the MLP's $\sigma = 0.0563$) and the highest uncalibrated F1 score at $	au = 0.50$ (**$0.6122 \pm 0.0533$** vs. **$0.5247 \pm 0.0878$** for Deep MLP).
2. **On the Full 165-Feature Representation (Precomputed Aggregates Included)**:
   - **True Graph Provides Zero Advantage Over Random Rewiring**: Condition B (**$0.5812 \pm 0.0639$**) performed similarly to Condition D (**$0.5890 \pm 0.0281$**, $ar{\Delta} = -0.0077$, unadjusted $p = 0.7537$).
   - **Feature-Only Deep MLP Dominates (Confirmatory / Robust Descriptive Evidence)**: Condition A (**$0.7189 \pm 0.0016$**) outperformed Condition B by $ar{\Delta} = -0.1377$ AUPRC (unadjusted $p = 0.0083$, **Holm-adjusted $p_{	ext{Holm}} = 0.0498$**), surviving family-wise error control across all 6 contrasts.
3. **Mechanistic Interpretation & Epistemic Boundary**:
   - Features `93..164` (72 features) in the 165f dataset are precomputed 1-hop neighborhood summary statistics. While combining GAT message passing with precomputed neighborhood statistics degrades accuracy relative to the 93f setting, we note that Condition C (Empty Edges on 165f, which performs *zero* neighbor message passing) also degraded to **$0.6150 \pm 0.0832$** relative to the 165f MLP ($0.7189$). Thus, the 165f GAT degradation reflects an interaction between colinear precomputed features and GAT self-attention parameterization rather than purely topological over-smoothing.

---

## 2. Comprehensive 40-Trial Performance Tables (100% Programmatically Generated)

All metrics are computed on the chronological validation partition (time steps 33–40, $7,653$ labelled nodes: $7,034$ licit, $619$ illicit) after training on time steps 1–32 ($28,938$ labelled nodes: $25,536$ licit, $3,402$ illicit) using class weights $[1.0, 7.5062]$:

### 2.1 Local 93-Feature Representation (20 Trials)

| Run ID | Cond | Description | Seed | Params | Best Ep / Tot | Wall (s) | Val AUPRC | Val F1 | Precision | Recall | ROC-AUC | TP | FP | TN | FN |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P4-Cond_A_93f_s42` | A | Deep MLP | 42 | 22,466 | 16 / 24 | 27.4s | **0.8043** | 0.4758 | 0.3271 | 0.8724 | 0.9432 | 540 | 1111 | 5923 | 79 |
| `P4-Cond_A_93f_s43` | A | Deep MLP | 43 | 22,466 | 16 / 24 | 25.3s | **0.7905** | 0.5455 | 0.3982 | 0.8659 | 0.9478 | 536 | 810 | 6224 | 83 |
| `P4-Cond_A_93f_s44` | A | Deep MLP | 44 | 22,466 | 16 / 24 | 22.2s | **0.7718** | 0.4694 | 0.3202 | 0.8788 | 0.9461 | 544 | 1155 | 5879 | 75 |
| `P4-Cond_A_93f_s45` | A | Deep MLP | 45 | 22,466 | 16 / 24 | 24.7s | **0.7782** | 0.4628 | 0.3135 | 0.8837 | 0.9447 | 547 | 1198 | 5836 | 72 |
| `P4-Cond_A_93f_s46` | A | Deep MLP | 46 | 22,466 | 1 / 9 | 18.0s | **0.6633** | 0.6700 | 0.8970 | 0.5347 | 0.8836 | 331 | 38 | 6996 | 288 |
| *Condition A Mean ± SD* | | | | *22,466* | *13.0* | *23.5s* | **0.7616 ± 0.0563** | *0.5247 ± 0.0878* | *0.4512 ± 0.2515* | *0.8071 ± 0.1524* | *0.9331 ± 0.0277* | | | | |
| `P4-Cond_B_93f_s42` | B | Residual True | 42 | 14,472 | 48 / 50 | 78.3s | **0.7713** | 0.5993 | 0.4578 | 0.8675 | 0.9570 | 537 | 636 | 6398 | 82 |
| `P4-Cond_B_93f_s43` | B | Residual True | 43 | 14,472 | 32 / 40 | 52.4s | **0.7708** | 0.6119 | 0.4679 | 0.8837 | 0.9532 | 547 | 622 | 6412 | 72 |
| `P4-Cond_B_93f_s44` | B | Residual True | 44 | 14,472 | 34 / 42 | 65.3s | **0.7844** | 0.6142 | 0.4804 | 0.8514 | 0.9561 | 527 | 570 | 6464 | 92 |
| `P4-Cond_B_93f_s45` | B | Residual True | 45 | 14,472 | 50 / 50 | 73.1s | **0.7627** | 0.5432 | 0.3940 | 0.8740 | 0.9516 | 541 | 832 | 6202 | 78 |
| `P4-Cond_B_93f_s46` | B | Residual True | 46 | 14,472 | 47 / 50 | 67.4s | **0.7900** | 0.6925 | 0.5828 | 0.8530 | 0.9619 | 528 | 378 | 6656 | 91 |
| *Condition B Mean ± SD* | | | | *14,472* | *42.2* | *67.3s* | **0.7758 ± 0.0111** | *0.6122 ± 0.0533* | *0.4766 ± 0.0681* | *0.8659 ± 0.0138* | *0.9560 ± 0.0040* | | | | |
| `P4-Cond_C_93f_s42` | C | Residual Empty | 42 | 14,472 | 2 / 10 | 9.1s | **0.4094** | 0.2731 | 0.1611 | 0.8950 | 0.8766 | 554 | 2884 | 4150 | 65 |
| `P4-Cond_C_93f_s43` | C | Residual Empty | 43 | 14,472 | 28 / 36 | 34.9s | **0.7645** | 0.4806 | 0.3290 | 0.8918 | 0.9481 | 552 | 1126 | 5908 | 67 |
| `P4-Cond_C_93f_s44` | C | Residual Empty | 44 | 14,472 | 6 / 14 | 11.4s | **0.6830** | 0.3094 | 0.1859 | 0.9208 | 0.9265 | 570 | 2496 | 4538 | 49 |
| `P4-Cond_C_93f_s45` | C | Residual Empty | 45 | 14,472 | 50 / 50 | 49.0s | **0.7954** | 0.5645 | 0.4147 | 0.8837 | 0.9537 | 547 | 772 | 6262 | 72 |
| `P4-Cond_C_93f_s46` | C | Residual Empty | 46 | 14,472 | 43 / 50 | 50.9s | **0.8167** | 0.5904 | 0.4421 | 0.8885 | 0.9584 | 550 | 694 | 6340 | 69 |
| *Condition C Mean ± SD* | | | | *14,472* | *25.8* | *31.0s* | **0.6938 ± 0.1669** | *0.4436 ± 0.1455* | *0.3066 ± 0.1287* | *0.8960 ± 0.0145* | *0.9327 ± 0.0336* | | | | |
| `P4-Cond_D_93f_s42` | D | Residual Rewired | 42 | 14,472 | 26 / 34 | 44.1s | **0.7181** | 0.3684 | 0.2323 | 0.8901 | 0.9302 | 551 | 1821 | 5213 | 68 |
| `P4-Cond_D_93f_s43` | D | Residual Rewired | 43 | 14,472 | 36 / 44 | 65.3s | **0.6903** | 0.4861 | 0.3367 | 0.8740 | 0.9336 | 541 | 1066 | 5968 | 78 |
| `P4-Cond_D_93f_s44` | D | Residual Rewired | 44 | 14,472 | 38 / 46 | 61.4s | **0.7778** | 0.4844 | 0.3329 | 0.8885 | 0.9471 | 550 | 1102 | 5932 | 69 |
| `P4-Cond_D_93f_s45` | D | Residual Rewired | 45 | 14,472 | 31 / 39 | 52.2s | **0.6814** | 0.3944 | 0.2535 | 0.8885 | 0.9312 | 550 | 1620 | 5414 | 69 |
| `P4-Cond_D_93f_s46` | D | Residual Rewired | 46 | 14,472 | 48 / 50 | 62.2s | **0.7584** | 0.5846 | 0.4409 | 0.8675 | 0.9494 | 537 | 681 | 6353 | 82 |
| *Condition D Mean ± SD* | | | | *14,472* | *35.8* | *57.1s* | **0.7252 ± 0.0420** | *0.4636 ± 0.0858* | *0.3192 ± 0.0824* | *0.8817 ± 0.0103* | *0.9383 ± 0.0092* | | | | |

---

### 2.2 Full 165-Feature Representation (20 Trials)

| Run ID | Cond | Description | Seed | Params | Best Ep / Tot | Wall (s) | Val AUPRC | Val F1 | Precision | Recall | ROC-AUC | TP | FP | TN | FN |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P4-Cond_A_165f_s42` | A | Deep MLP | 42 | 31,682 | 6 / 14 | 19.3s | **0.7218** | 0.3647 | 0.2274 | 0.9208 | 0.9351 | 570 | 1937 | 5097 | 49 |
| `P4-Cond_A_165f_s43` | A | Deep MLP | 43 | 31,682 | 5 / 13 | 17.8s | **0.7187** | 0.3053 | 0.1817 | 0.9548 | 0.9405 | 591 | 2662 | 4372 | 28 |
| `P4-Cond_A_165f_s44` | A | Deep MLP | 44 | 31,682 | 6 / 14 | 15.4s | **0.7184** | 0.3441 | 0.2104 | 0.9451 | 0.9385 | 585 | 2196 | 4838 | 34 |
| `P4-Cond_A_165f_s45` | A | Deep MLP | 45 | 31,682 | 5 / 13 | 13.1s | **0.7175** | 0.2925 | 0.1723 | 0.9693 | 0.9470 | 600 | 2883 | 4151 | 19 |
| `P4-Cond_A_165f_s46` | A | Deep MLP | 46 | 31,682 | 2 / 10 | 19.6s | **0.7183** | 0.3757 | 0.2364 | 0.9144 | 0.9335 | 566 | 1828 | 5206 | 53 |
| *Condition A Mean ± SD* | | | | *31,682* | *4.8* | *17.0s* | **0.7189 ± 0.0016** | *0.3365 ± 0.0364* | *0.2056 ± 0.0280* | *0.9409 ± 0.0230* | *0.9389 ± 0.0053* | | | | |
| `P4-Cond_B_165f_s42` | B | Residual True | 42 | 23,688 | 22 / 30 | 69.7s | **0.5980** | 0.3955 | 0.2534 | 0.8998 | 0.9211 | 557 | 1641 | 5393 | 62 |
| `P4-Cond_B_165f_s43` | B | Residual True | 43 | 23,688 | 11 / 19 | 27.7s | **0.5684** | 0.3655 | 0.2306 | 0.8805 | 0.9050 | 545 | 1818 | 5216 | 74 |
| `P4-Cond_B_165f_s44` | B | Residual True | 44 | 23,688 | 12 / 20 | 32.3s | **0.6342** | 0.4093 | 0.2668 | 0.8788 | 0.9264 | 544 | 1495 | 5539 | 75 |
| `P4-Cond_B_165f_s45` | B | Residual True | 45 | 23,688 | 7 / 15 | 19.9s | **0.4771** | 0.3513 | 0.2246 | 0.8061 | 0.8764 | 499 | 1723 | 5311 | 120 |
| `P4-Cond_B_165f_s46` | B | Residual True | 46 | 23,688 | 14 / 22 | 35.0s | **0.6285** | 0.4582 | 0.3167 | 0.8288 | 0.9235 | 513 | 1107 | 5927 | 106 |
| *Condition B Mean ± SD* | | | | *23,688* | *13.2* | *36.9s* | **0.5812 ± 0.0639** | *0.3960 ± 0.0418* | *0.2584 ± 0.0368* | *0.8588 ± 0.0395* | *0.9105 ± 0.0208* | | | | |
| `P4-Cond_C_165f_s42` | C | Residual Empty | 42 | 23,688 | 11 / 19 | 34.8s | **0.4706** | 0.3599 | 0.2239 | 0.9176 | 0.9016 | 568 | 1969 | 5065 | 51 |
| `P4-Cond_C_165f_s43` | C | Residual Empty | 43 | 23,688 | 11 / 19 | 30.5s | **0.6747** | 0.3714 | 0.2317 | 0.9354 | 0.9306 | 579 | 1920 | 5114 | 40 |
| `P4-Cond_C_165f_s44` | C | Residual Empty | 44 | 23,688 | 16 / 24 | 47.0s | **0.6635** | 0.4154 | 0.2720 | 0.8788 | 0.9293 | 544 | 1456 | 5578 | 75 |
| `P4-Cond_C_165f_s45` | C | Residual Empty | 45 | 23,688 | 9 / 17 | 22.5s | **0.6220** | 0.3503 | 0.2155 | 0.9370 | 0.9216 | 580 | 2112 | 4922 | 39 |
| `P4-Cond_C_165f_s46` | C | Residual Empty | 46 | 23,688 | 15 / 23 | 22.6s | **0.6440** | 0.4090 | 0.2742 | 0.8045 | 0.9112 | 498 | 1318 | 5716 | 121 |
| *Condition C Mean ± SD* | | | | *23,688* | *12.4* | *31.5s* | **0.6150 ± 0.0832** | *0.3812 ± 0.0294* | *0.2435 ± 0.0277* | *0.8947 ± 0.0556* | *0.9189 ± 0.0124* | | | | |
| `P4-Cond_D_165f_s42` | D | Residual Rewired | 42 | 23,688 | 22 / 30 | 48.3s | **0.5806** | 0.3906 | 0.2519 | 0.8691 | 0.9140 | 538 | 1598 | 5436 | 81 |
| `P4-Cond_D_165f_s43` | D | Residual Rewired | 43 | 23,688 | 10 / 18 | 25.2s | **0.5816** | 0.3470 | 0.2146 | 0.9063 | 0.9098 | 561 | 2053 | 4981 | 58 |
| `P4-Cond_D_165f_s44` | D | Residual Rewired | 44 | 23,688 | 15 / 23 | 43.2s | **0.5815** | 0.3663 | 0.2314 | 0.8788 | 0.9127 | 544 | 1807 | 5227 | 75 |
| `P4-Cond_D_165f_s45` | D | Residual Rewired | 45 | 23,688 | 17 / 25 | 59.2s | **0.5638** | 0.3974 | 0.2606 | 0.8368 | 0.9105 | 518 | 1470 | 5564 | 101 |
| `P4-Cond_D_165f_s46` | D | Residual Rewired | 46 | 23,688 | 17 / 25 | 50.1s | **0.6373** | 0.4161 | 0.2791 | 0.8174 | 0.9128 | 506 | 1307 | 5727 | 113 |
| *Condition D Mean ± SD* | | | | *23,688* | *16.2* | *45.2s* | **0.5890 ± 0.0281** | *0.3835 ± 0.0271* | *0.2475 ± 0.0251* | *0.8617 ± 0.0351* | *0.9119 ± 0.0017* | | | | |

---

## 3. Summary Performance Matrix & Overall Ranking

### Ranking on Local 93 Features:
1. **Condition B (Residual ETGNN, True Graph, 14,472 params)**: AUPRC **$0.7758 \pm 0.0111$** | F1 ($	au=0.5$): **$0.6122 \pm 0.0533$** | Precision: **$0.4766 \pm 0.0684$** | Recall: **$0.8659 \pm 0.0136$**
2. **Condition A (Deep MLP, Feature Only, 22,466 params)**: AUPRC **$0.7616 \pm 0.0563$** | F1 ($	au=0.5$): **$0.5247 \pm 0.0878$** | Precision: **$0.4512 \pm 0.2515$** | Recall: **$0.8071 \pm 0.1524$**
3. **Condition D (Residual ETGNN, Rewired Graph, 14,472 params)**: AUPRC **$0.7252 \pm 0.0420$** | F1 ($	au=0.5$): **$0.4636 \pm 0.0858$** | Precision: **$0.3192 \pm 0.0825$** | Recall: **$0.8817 \pm 0.0102$**
4. **Condition C (Residual ETGNN, Empty Edges, 14,472 params)**: AUPRC **$0.6938 \pm 0.1669$** | F1 ($	au=0.5$): **$0.4436 \pm 0.1455$** | Precision: **$0.3066 \pm 0.1292$** | Recall: **$0.8960 \pm 0.0145$**

### Ranking on Full 165 Features:
1. **Condition A (Deep MLP, Feature Only, 31,682 params)**: AUPRC **$0.7189 \pm 0.0016$** | F1 ($	au=0.5$): **$0.3365 \pm 0.0364$** | Precision: **$0.2056 \pm 0.0280$** | Recall: **$0.9409 \pm 0.0232$**
2. **Condition C (Residual ETGNN, Empty Edges, 23,688 params)**: AUPRC **$0.6150 \pm 0.0832$** | F1 ($	au=0.5$): **$0.3812 \pm 0.0294$** | Precision: **$0.2435 \pm 0.0277$** | Recall: **$0.8947 \pm 0.0556$**
3. **Condition D (Residual ETGNN, Rewired Graph, 23,688 params)**: AUPRC **$0.5890 \pm 0.0281$** | F1 ($	au=0.5$): **$0.3835 \pm 0.0271$** | Precision: **$0.2475 \pm 0.0251$** | Recall: **$0.8617 \pm 0.0352$**
4. **Condition B (Residual ETGNN, True Graph, 23,688 params)**: AUPRC **$0.5812 \pm 0.0639$** | F1 ($	au=0.5$): **$0.3960 \pm 0.0418$** | Precision: **$0.2584 \pm 0.0369$** | Recall: **$0.8588 \pm 0.0394$**
