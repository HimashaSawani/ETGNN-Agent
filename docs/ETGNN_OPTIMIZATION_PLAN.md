# ETGNN Controlled Model Performance Improvement Study
## Comprehensive Research Audit, Performance Diagnosis, and Bounded Experiment Plan

**Repository:** `ETGNN-Fraud-Detection`  
**Author / Investigator:** Senior GNN Researcher & PyTorch Geometric Systems Engineer  
**Date:** October 2026  
**Status:** Phase 1 (Baseline Audit) & Phase 2 (Performance Diagnosis) Complete; Experiment Plan Formulated  
**Target Metric:** Validation AUPRC (Area Under the Precision-Recall Curve) on time steps 33–40  

---

## 1. Executive Summary

This study investigates whether the existing Explainable Temporal Graph Neural Network (**ETGNN**) model can achieve superior validation performance without compromising scientific validity, violating temporal ordering, or engaging in post-hoc data snooping. 

All analyses, baselines, and proposed interventions adhere to strict methodological safeguards:
1. **Zero Test-Set Snooping**: Model selection, hyperparameter tuning, and structural ablations are restricted exclusively to the validation split (time steps 33–40). The held-out test split (time steps 41–49) remains completely untouched during configuration search.
2. **Preservation of Core Artifacts**: All original checkpoints (`results/checkpoints/etgnn.pt`), frozen thesis results (Chapters 1–6), Contrast 1 (C1) and Contrast 2 (C2) registers, and web demo services remain completely unmodified.
3. **Dedicated Experiment Isolation**: All new exploratory models, training logs, and checkpoints are directed to a quarantined directory: `results/optimization_experiments/`.
4. **Empirically Measured Compute Bound**: Prior to launching full experiments, hardware capacity, memory footprint, and CPU execution times were empirically profiled. A single training run on the local CPU requires ~8.27 seconds per epoch (total ~3.5 to 5.0 minutes with early stopping), consuming ~280 MB resident RAM.

---

## 2. Phase 1 — Baseline Audit & Exact Reproduction

### 2.1. Dataset Split & Chronological Integrity Audit
The repository operates on the **Elliptic Bitcoin Dataset** (203,769 transaction nodes and 234,355 directed edges across 49 discrete time steps):

| Split Name | Time Step Range | Total Transaction Nodes | Labelled Nodes | Licit (0) | Illicit (1) | Illicit Class Ratio |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | $t \in [1, 32]$ | 130,628 | 28,938 | 25,536 | 3,402 | 11.76% (1 : 7.51) |
| **Validation** | $t \in [33, 40]$ | 161,603 cumulative | 7,653 (in $t \in [33, 40]$) | 7,034 | 619 | 8.09% (1 : 11.36) |
| **Test** | $t \in [41, 49]$ | 203,769 cumulative | 9,973 (in $t \in [41, 49]$) | 9,449 | 524 | 5.25% (1 : 18.03) |

**Verification Findings:**
- **Zero Temporal Leakage in Graph Slices**: Edges in the Elliptic dataset exist strictly within individual time steps. In `build_stage_graphs.py`, stage-specific graph slices enforce `times[source] <= cutoff` and `times[target] <= cutoff`, preventing message passing from future time steps into past computations.
- **Feature Normalization Integrity**: In `scale_features.py`, the `StandardScaler` was strictly fitted on rows where `time_step <= 32`. No validation or test features leaked into the scaler statistics.
- **Feature Provenance**: The 165 input features comprise **93 local features** (transaction fees, input/output counts, local transaction properties) and **72 precomputed one-hop neighborhood aggregates** (mean, std, min, max of neighbor features).

### 2.2. Baseline Model Architecture & Training Configuration
The original baseline ETGNN (`src/models/etgnn.py` and `src/training/train_etgnn.py`) is specified as:
- **Input Dimensions**: 165 features + 16-dimensional sinusoidal time encoding = 181 dimensions.
- **Layer 1**: `GATConv(181, 16, heads=4, dropout=0.3)` $\to$ output dimension 64, followed by `.relu()` and `dropout=0.3`.
- **Layer 2**: `GATConv(64, 2, heads=1, concat=False, dropout=0.3)`.
- **Loss Function**: `torch.nn.CrossEntropyLoss(weight=[1.0, 7.506])`, where class 1 weight is $\frac{25536}{3402} \approx 7.506$.
- **Optimizer**: Adam with learning rate $\eta = 0.01$, weight decay $\lambda = 5 \times 10^{-4}$.
- **Seed & Threading**: `torch.manual_seed(42)`, `torch.set_num_threads(2)`.
- **Early Stopping**: Monitored on validation AUPRC with minimum delta $\Delta = 0.0001$ and patience $P = 8$ epochs, evaluated up to 50 epochs.

### 2.3. Checkpoint Numerical Identity Verification
We verified the exact frozen checkpoint `results/checkpoints/etgnn.pt` by evaluating it on `data/processed/validation_graph.pt` using `scratch/verify_baseline_eval.py`:

| Metric | Computed from Checkpoint | Stored in `etgnn_validation.json` | Absolute Discrepancy |
|---|:---:|:---:|:---:|
| **Best Epoch** | — | 40 | Exact |
| **Precision** | 0.3909836065573771 | 0.3909836065573771 | **0.0000000000** |
| **Recall** | 0.7705977382875606 | 0.7705977382875606 | **0.0000000000** |
| **F1 Score** | 0.5187601957585645 | 0.5187601957585645 | **0.0000000000** |
| **ROC-AUC** | 0.9202537134426234 | 0.9202537134426234 | **0.0000000000** |
| **AUPRC** | **0.5487728712276526** | **0.5487728712276526** | **0.0000000000** |

**Conclusion:** The baseline is 100% reproducible with zero floating-point error. The original state is preserved and locked.

---

## 3. Phase 2 — Performance Diagnosis (Evidence vs. Hypotheses)

### 3.1. Empirically Observed Evidence (Repository Facts)
1. **Severe Multi-Stage Class Imbalance**: Labelled illicit transactions constitute only 11.76% in training, dropping to 8.09% in validation, and 5.25% in test.
2. **Loss Weighting Trade-Off**: The inverse-frequency weight (7.51) aggressively boosts recall (77.1%) at the expense of precision (39.1%) at threshold 0.5. False positives outnumber true positives by nearly 1.5 to 1.
3. **The GNN Underperformance Paradox**: 
   - Non-graph Random Forest achieves **Val AUPRC 0.9102**, F1 0.8750.
   - Non-graph XGBoost achieves **Val AUPRC 0.8425**, F1 0.6578.
   - ETGNN (with graph message passing) achieves only **Val AUPRC 0.5488**, F1 0.5188.
   - Crucially, when ETGNN is evaluated on 93 local features with **empty edges** (`train_etgnn_local_no_edges.py`), validation AUPRC jumps to **0.8282**!
4. **Graph Directionality & Structural Bottlenecks**:
   - Bitcoin transaction edges represent directed payment flows ($u \to v$).
   - Standard `GATConv` only aggregates incoming edges (from payers to payees). Nodes with in-degree 0 receive no neighbor messages.
   - 84.1% of transactions have exactly 2 computation edges, and 95.8% select only 1 edge under XAI budget rules.
5. **Feature Conflation**: The input matrix already includes 72 precomputed one-hop neighborhood features. A 2-layer GNN computes a 2-hop convolution over an existing 1-hop convolution, resulting in 3-hop receptive field diffusion.

### 3.2. Unverified Hypotheses for Controlled Testing
- **Hypothesis 1 (Loss Calibration & Ranking)**: Inverse-frequency loss weighting (`weight=7.506`) forces logits into extreme margins, impairing continuous rank discrimination measured by AUPRC. Moderate weighting (e.g., square-root ratio $\approx 2.74$, or unweighted CE with post-hoc decision thresholding) will improve ranking order and AUPRC.
- **Hypothesis 2 (Connectivity & Bidirectional Message Passing)**: In financial fraud, money laundering involves bidirectional relationships (source of funds and destination of funds). Converting the directed graph to bidirectional (`to_undirected(edge_index)`) or incorporating residual/self-connection preservation will provide richer structural context.
- **Hypothesis 3 (Oversmoothing & Feature Dilution)**: Standard GAT without residual connections smooths distinct fraud features into the licit majority. Adding a skip connection ($x_{proj} + \text{GAT}(x)$) or higher dropout (0.4–0.5) will preserve local transaction specificity.
- **Hypothesis 4 (Optimization Dynamics)**: Learning rate $\eta = 0.01$ is high for multi-head attention on tabular graph features, causing coarse parameter oscillation. A gentler learning rate ($\eta = 0.003$ or $0.005$) with cosine decay or lower weight decay ($10^{-4}$) will enable better convergence.
- **Hypothesis 5 (Temporal Encoding Granularity)**: Fixed sinusoidal 16-dim encoding was concatenated naively. Evaluating time dimensions (8, 16, 32) and additive vs. concatenated projections will clarify whether the model benefits from richer temporal signals.

---

## 4. Phase 3 & 4 — Compute Profiling & Resource-Bounded Experiment Plan

### 4.1. Hardware Specification & Empirical Compute Benchmark
Execution environment profiled on host system:
- **Processor**: 4 physical cores (8 logical threads), CPU-only PyTorch (2.14.0+cpu, CUDA not available).
- **System Memory**: 7.65 GB total RAM, with ~1.04 GB available free memory.
- **Empirical Epoch Benchmark** (`scratch/benchmark_epoch.py` with `num_threads=2`):
  - Model & data loading time: **0.92 seconds**.
  - Train forward + backward pass: **6.32 seconds / epoch**.
  - Validation forward + AUPRC evaluation: **1.95 seconds / epoch**.
  - Total time per epoch: **8.27 seconds**.
  - Maximum process memory: **257.0 MB to 283.1 MB** (occupying ~27% of available RAM, safely zero OOM risk).
  - Maximum early-stopping training run (~20–40 epochs): **2.8 to 5.5 minutes**.

### 4.2. Experiment Matrix (Bounded Set of 16 Controlled Configurations)

To ensure maximum scientific utility without exceeding compute limits, we define 16 bounded experimental runs across 6 research axes:

| Run ID | Research Axis | Architectural / Training Change | Rationale | Est. Epochs | Est. Time (min) |
|---|---|---|---|:---:|:---:|
| **EXP-00** | *Reference Baseline* | Standard ETGNN ($\eta=0.01$, $\lambda=5\text{e-}4$, $d=0.3$, $w=7.51$, heads=4) | Re-verify baseline under identical protocol | ~40 | ~5.5 |
| **EXP-01** | 1. Learning Rate | $\eta = 0.003$, weight decay $\lambda = 5\text{e-}4$ | Less aggressive step size | ~30 | ~4.1 |
| **EXP-02** | 1. Learning Rate | $\eta = 0.005$, weight decay $\lambda = 5\text{e-}4$ | Intermediate step size | ~30 | ~4.1 |
| **EXP-03** | 1. Weight Decay | $\eta = 0.005$, weight decay $\lambda = 1\text{e-}4$ | Reduced L2 penalty on attention weights | ~30 | ~4.1 |
| **EXP-04** | 2. Dropout | Dropout $p = 0.2$ (vs 0.3) | Reduce feature suppression | ~30 | ~4.1 |
| **EXP-05** | 2. Dropout | Dropout $p = 0.5$ (vs 0.3) | Stronger regularization against overfitting | ~30 | ~4.1 |
| **EXP-06** | 2. Capacity & Res | Hidden dim = 96 (Layer 1 output 96) | Higher representational width | ~30 | ~4.8 |
| **EXP-07** | 2. Skip Connection | Residual linear projection: $h = \text{ReLU}(\text{GAT}(x) + W_{res} x)$ | Mitigate oversmoothing / preserve ego-features | ~30 | ~4.5 |
| **EXP-08** | 3. Attention Heads | Heads = 2 (dim 32/head = 64) | Less attention variance on small neighborhood | ~30 | ~3.8 |
| **EXP-09** | 3. Attention Heads | Heads = 8 (dim 8/head = 64) | Finer-grained multi-subspace attention | ~30 | ~5.2 |
| **EXP-10** | 4. Class Weighting | Unweighted loss ($w = [1.0, 1.0]$) | Test if unweighted CE produces better ranking | ~30 | ~4.1 |
| **EXP-11** | 4. Class Weighting | Sqrt class weight ($w = [1.0, \sqrt{7.51}] \approx [1.0, 2.74]$) | Balanced penalty without extreme margin shift | ~30 | ~4.1 |
| **EXP-12** | 5. Connectivity | Bidirectional edges: `to_undirected(edge_index)` | Message passing along input AND output cash flows | ~30 | ~5.8 |
| **EXP-13** | 5. Connectivity | Reversed edges: `edge_index.flip(0)` | Upstream vs downstream flow comparison | ~30 | ~4.1 |
| **EXP-14** | 6. Temporal Dim | Time encoding dimension = 8 (vs 16) | Compact temporal signature | ~30 | ~4.1 |
| **EXP-15** | 6. Temporal Dim | Time encoding dimension = 32 (vs 16) | Richer multi-scale temporal frequency | ~30 | ~4.2 |

**Total Estimated Compute Budget:**
- 16 runs $\times$ ~4.2 minutes average $\approx$ **67 minutes total sequential execution time**.
- Memory peak: $\le 350\text{ MB}$, zero GPU dependence, zero interference with running web services.

### 4.3. Multi-Seed Validation for Promising Candidates
Any configuration that improves upon the baseline validation AUPRC ($> 0.5488$) by at least $+0.0100$ will be subjected to multi-seed verification across seeds **`[42, 43, 44]`** to distinguish true structural superiority from random initialization artifacts.

---

## 5. Phase 5 & 6 — Model Selection Protocol & Holdout Disclaimers

### 5.1. Model Selection Criteria
The selection of the superior model will be governed strictly by:
1. **Primary Metric**: Validation AUPRC on labelled transactions in time steps 33–40.
2. **Secondary Metrics**: Validation F1 score, Precision-Recall balance, and multi-seed stability (variance across seeds 42, 43, 44).
3. **Ablation Transparency**: If an edge-free or simplified model outperforms GAT variants, this result will be documented honestly as evidence of graph relational noise or oversmoothing, rather than obscured.

### 5.2. Mandatory Holdout Epistemic Notice (Phase 6)
> **Crucial Methodological Disclaimer**:  
> The held-out test split (time steps 41–49) has already been evaluated in earlier project analysis (yielding baseline test AUPRC 0.2250 for ETGNN, 0.6323 for RF). Consequently, any subsequent evaluation of optimized models on test steps 41–49 **cannot be characterized as a fresh, untouched, unbiased holdout**. Any such evaluation must be explicitly classified as **exploratory post-hoc verification**.

---

## 6. Execution Safety Rules
1. Never overwrite `results/checkpoints/etgnn.pt`. New checkpoints will be saved to `results/optimization_experiments/checkpoints/`.
2. Never modify `results/tables/etgnn_validation.json` or `results/tables/final_test_results.csv`.
3. Never alter the web backend default model until an optimized checkpoint has been comprehensively validated and explicitly approved.
