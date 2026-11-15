# Validation Feasibility Pilot Report: Edge-Only Attribution Under Structural Liability

**Experiment Directory**: `experiments/pilot_validation_audit/`  
**Execution Timestamp**: 2026-10-19 20:55:51 UTC  
**Target Architecture**: Local-Feature Time-Encoded GAT (`ETGNNClassifier`, input dimension 93, hidden dimension 64, 4 attention heads, dropout 0.3)  
**Evaluated Checkpoint**: `results/checkpoints/etgnn_local.pt` (SHA256: `2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082`)  
**Data Evaluated**: Validation graph snapshot `data/processed/validation_graph.pt` (time steps 33–40 only)  
**Strict Isolation**: Test steps 41–49 were **not accessed**; existing checkpoints and tables were **not modified**.

---

## 1. Executive Summary of Observed Findings

1. **Deletion Drop Advantage**: Across all 20 pilot nodes, deleting the top 25% GNNExplainer edges produced a mean signed drop of **+0.1216 $\pm$ 0.0972** in original-class confidence, outperforming uniform-random edge deletion (**+0.0087 $\pm$ 0.0594**) and source out-degree ranking (**-0.0062 $\pm$ 0.1237**). However, exhaustive **single-edge occlusion ranking** achieved an even stronger drop (**+0.2027 $\pm$ 0.0954**).
2. **Asymmetric Conditional Correction on Errors**:
   - On **False Positives (FP)**, deleting top-ranked GNNExplainer edges flipped the predicted class from illicit (1) to licit (0) in **86.7% of runs** (13 of 15 seed runs), compared to only **19.3%** for uniform-random deletion and **0.0%** for source out-degree ranking.
   - On **False Negatives (FN)**, deletion achieved only a **20.0% correction rate** under GNNExplainer and **1.3%** under random draws.
3. **Explainer Optimization Instability**: Across three random explainer seeds (42, 43, 44), the mean pairwise Jaccard similarity of selected top-25% edges was **0.3702** (ranging from **0.2041** on TP to **0.5470** on TN). GNNExplainer edge masks exhibit high stochastic variation across optimization runs despite identical input subgraphs.
4. **Computational Costs**:
   - GNNExplainer (50 epochs): Generation = **0.620 sec/node**, Evaluation = **0.009 sec/node**.
   - Single-Edge Occlusion: Generation = **0.777 sec/node**, Evaluation = **0.047 sec/node**.
   - Uniform-Random (30 draws): Generation = **0.00014 sec/draw**, Evaluation = **0.008 sec/draw**.
   - Source Out-Degree: Generation = **0.00008 sec/node**, Evaluation = **0.008 sec/node**.

---

## 2. Inventory and Verification Audit

### 2.1 Preprocessing and Mapping Pipeline
- `src/data/create_id_map.py`: Maps 203,769 transactions to contiguous 0-indexed integers (`data/processed/node_mapping.csv`).
- `src/data/map_edges.py`: Maps 234,355 directed edges (`data/processed/mapped_edges.csv`).
- `src/data/map_labels.py`: Aligns classes (`1` = illicit, `0` = licit, `-1` = unknown) (`data/processed/node_labels.csv`).
- `src/data/scale_features.py`: Fits `StandardScaler` on time steps 1–32 only (130,628 nodes), writes `data/processed/scaled_features.npy`.
- `src/data/build_stage_graphs.py`: Builds stage snapshots (`train_graph.pt`, `validation_graph.pt`, `test_graph.pt`).
- `src/data/check_edge_times.py`: Asserts `(source_time == target_time).all()`; verifies 100% of edges are contemporaneous.

### 2.2 Split Masks and Data Partitions
- Partitioned via `data/processed/nodes_with_masks.csv`:
  - **Train (Steps 1–32)**: 28,938 labelled nodes (25,536 licit, 3,402 illicit).
  - **Validation (Steps 33–40)**: 7,653 labelled nodes (7,034 licit, 619 illicit).
  - **Held-Out Test (Steps 41–49)**: 9,973 labelled nodes (9,449 licit, 524 illicit).
  - **Unknown Nodes (`label == -1`)**: 157,205 across all periods, strictly excluded from supervised masks and metric calculations.

### 2.3 Model Definitions and Evaluated Checkpoint
- Architecture: `ETGNNClassifier` in `src/models/etgnn.py`.
- Evaluated Checkpoint: `results/checkpoints/etgnn_local.pt` (32,901 bytes).
  - Trained on 93 local features (cols 0–92) + 16 sinusoidal time dimensions. Excludes 72 precomputed graph aggregates.
  - Frozen weights validated on steps 33–40: Validation AUPRC = 0.5791, F1 = 0.4759.

### 2.4 Missing Artifacts Identified
1. **Validation Prediction Vectors**: No `.npz` file containing saved validation probability arrays existed (only test probabilities were saved in `data/processed/final_test_probabilities.npz`). Validation probabilities had to be evaluated live on `validation_graph.pt`.
2. **Training Loss Logs**: No raw stdout training logs exist in `results/checkpoints/` or `results/tables/`.

---

## 3. Experimental Design and Feasibility Protocol

### 3.1 Node Selection (20 Labelled Nodes)
- Selected 5 nodes per quadrant from `validation_graph.pt` where `val_mask` is True, `time_step.between(33, 40)`, and incoming degree $\ge 2$:
  - **5 True Positives (TP)**: Nodes 148166 (step 37), 130726 (step 33), 133786 (step 34), 133832 (step 34), 133884 (step 34).
  - **5 False Positives (FP)**: Nodes 130690 (step 33), 130935 (step 33), 136379 (step 35), 154421 (step 39), 133801 (step 34).
  - **5 False Negatives (FN)**: Nodes 142234 (step 36), 142515 (step 36), 148265 (step 37), 157211 (step 40), 157389 (step 40).
  - **5 True Negatives (TN)**: Nodes 148292 (step 37), 154471 (step 39), 148195 (step 37), 154373 (step 39), 151484 (step 38).
- **Full/Subgraph Inference Equivalence**: Verified for all 20 nodes using `torch.abs(full_probs - sub_probs).max()`. Max observed discrepancy across all 20 nodes was **$1.49 \times 10^{-8}$**, confirming mathematical identity.

### 3.2 Candidate Edges and Intervention Budget
- For each node $v$, candidate edges $E_{\mathrm{cand}}$ are the edges in its 2-hop computational subgraph (`flow="source_to_target"`).
- Candidate edge count ranged from 6 (node 157389) to 367 (node 148292). Total candidate edges across 20 nodes = 2,544.
- **Budget Rule**: $k = \max(1, \lceil 0.25 \times E_{\mathrm{cand}} \rceil)$. All methods evaluated identical candidate budgets:
  - Node 157389 ($E=6$): $k=2$ (33.3%).
  - Node 148265 ($E=11$): $k=3$ (27.3%).
  - Node 148292 ($E=367$): $k=92$ (25.1%).
  - Zero nodes triggered `is_full_set` ($k == E$) or `insufficient_candidates` ($E < 2$).

### 3.3 Comparative Ranking Methods
1. **GNNExplainer (Edge-Only)**: Optimized for 50 epochs with `node_mask_type=None` across 3 seeds (42, 43, 44). Node attributes and sinusoidal time encodings remained unmasked and fixed.
2. **Uniform-Random Subsets**: 30 independent draws of size $k$ without replacement using recorded seed `20261019 + r * 1000 + (node_id % 1000)`.
3. **Source Out-Degree Ranking**: Edges ranked by the out-degree of their source transaction in `validation_graph.pt` (stable tie-breaking by canonical edge index).
4. **Single-Edge Occlusion**: Every candidate edge $e_i$ deleted individually; ranked by individual drop in original-class confidence $\Delta_i = P(Y=c) - P(Y=c \mid E \setminus \{e_i\})$.

---

## 4. Aggregate Results and Quantitative Comparison

### Table 1: Comparative Intervention Metrics (Across 700 Evaluations)

| Method | Mean Signed Deletion Response | Std Deletion Response | Mean Retention Prob | FP Correction Rate | FN Correction Rate | TP Error Induction Rate | TN Error Induction Rate | Mean Gen Time | Mean Eval Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNNExplainer (3 seeds)** | **+0.1216** | 0.0972 | 0.8235 | **86.7%** | 20.0% | **0.0%** | 20.0% | 0.620 s | 0.009 s |
| **Single-Edge Occlusion** | **+0.2027** | 0.0954 | 0.8917 | **100.0%** | 20.0% | 20.0% | 40.0% | 0.777 s | 0.047 s |
| **Uniform-Random (30 draws)** | +0.0087 | 0.0594 | 0.6925 | 19.3% | 1.3% | 0.0% | 0.0% | 0.0001 s | 0.008 s |
| **Source Out-Degree** | -0.0062 | 0.1237 | 0.6033 | 0.0% | 0.0% | 0.0% | 0.0% | 0.00008 s | 0.008 s |

### Table 2: Breakdown by Quadrant (Means Across Nodes)

| Category | GNNExplainer Drop Mean | Uniform-Random Drop Mean | Degree Drop | Occlusion Drop | Paired Advantage (GE - Random) | GNNExplainer Jaccard Stability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **False Positives (FP)** | **+0.1601** | -0.00003 | -0.1125 | +0.2443 | **+0.1602** | 0.4075 |
| **True Negatives (TN)** | **+0.2190** | -0.0037 | +0.0102 | +0.2512 | **+0.2227** | 0.5470 |
| **True Positives (TP)** | +0.0548 | +0.0103 | -0.0082 | +0.1194 | +0.0445 | 0.2041 |
| **False Negatives (FN)** | +0.0525 | +0.0284 | +0.0858 | +0.1959 | +0.0241 | 0.3221 |

---

## 5. Methodological Analysis & Pilot Insights

1. **GNNExplainer vs. Occlusion**:
   Single-edge occlusion produced higher confidence drops (+0.2027 vs +0.1216) than GNNExplainer. This occurs because GNNExplainer optimizes a continuous relaxation that does not directly simulate discrete subset removal. However, GNNExplainer achieved 0.0% error induction on TP nodes, whereas occlusion induced errors on 20% of TP cases.
2. **Degree Ranking Failure**:
   Source out-degree ranking performed poorly (mean drop -0.0062; in FPs, it increased confidence in the error by -0.1125). In financial transaction graphs, high out-degree fan-out transactions (e.g. mining pool distributions or exchanges) do not inherently represent the most influential fraud attributions.
3. **The False Positive Pathology**:
   In 4 of the 5 FP cases, deleting the top 25% explanation edges dropped the illicit probability from $>0.50$ down to $\le 0.40$, successfully correcting the model error. This supports the structural liability hypothesis: the model's false alarms were induced by noisy neighbor message passing, which edge explanation successfully isolated.

---

## 6. Reproducibility Command

To reproduce this feasibility audit from scratch:
```powershell
# From project root D:\ETGNN-Fraud-Detection
.\.venv\Scripts\python.exe -m src.evaluation.run_validation_pilot
```
All outputs are saved to `experiments/pilot_validation_audit/`:
- `config.json`: Complete environment and parameter snapshot.
- `pilot_nodes.csv`: 20 selected nodes and candidate subgraph stats.
- `candidate_edges.csv`: 2,544 candidate edges with degree annotations.
- `raw_interventions.csv`: 700 individual intervention trials.
- `node_summary.csv`: Per-node summarized repetitions and paired statistics.
- `aggregate_summary.csv`: Macro comparative metrics.
- `plots/pilot_per_node_paired_deletion_responses.png`: 20-panel paired comparison plot.
- `plots/pilot_paired_advantage_bars.png`: Paired advantage bar plot.
