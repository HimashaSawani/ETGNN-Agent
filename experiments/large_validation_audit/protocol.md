# Pre-Registered Protocol: Large-Scale Validation Evaluation (N=80 Nodes)
## Edge-Attribution Perturbation Responses Under Stratified Random Sampling

**Document Status**: Pre-Registered Before Execution  
**Target Sample**: $N = 80$ validation transactions (20 True Positives, 20 False Positives, 20 False Negatives, 20 True Negatives)  
**Sample File**: `experiments/large_validation_audit/sampled_nodes_80.csv`  
**Candidate Edges File**: `experiments/large_validation_audit/candidate_edges_80.csv` (Total 673 candidate edges)  
**Evaluated Architecture**: Local-Feature Time-Encoded GAT (`ETGNNClassifier`, input_dim=93, hidden_dim=64, 4 attention heads, dropout 0.3)  
**Evaluated Checkpoint**: `results/checkpoints/etgnn_local.pt` (SHA256: `2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082`)  
**Data Partition**: Validation graph `data/processed/validation_graph.pt` (Time steps 33–40 only)  
**Test Data Guarantee**: Time steps 41–49 are strictly **unaccessed** and excluded from the evaluation bundle.

---

## 1. Rationale and Research Questions

The preliminary feasibility pilot ($N=20$ nodes, 5 per quadrant) identified two key phenomena:
1. **False Positive Correction**: 86.7% of repeated-seed GNNExplainer trials flipped FP predictions to licit.
2. **Explainer Optimization Instability**: Multi-seed Jaccard stability was 0.3702 overall and 0.2041 on True Positives.

However, the pilot was severely confounded by its convenience sampling scheme (selecting the highest in-degree nodes), which caused:
- False Positives to be clustered extremely close to the decision boundary (mean margin = 0.058).
- False Positives to have much larger candidate edge sets (mean 108.8) than False Negatives (mean 31.8).

This large-scale follow-up study resolves these selection biases by employing **reproducible stratified random sampling** across the eligible validation population.

### Pre-Registered Hypotheses:
- **H1 (Edge Specificity)**: GNNExplainer top-25% edge deletion produces a higher signed drop in original-class confidence than uniform-random deletion ($\bar{\Delta}_{\mathrm{GE}} - \bar{\Delta}_{\mathrm{Rnd}} > 0$) with a 95% confidence interval strictly excluding zero.
- **H2 (Confounded FP/FN Asymmetry)**: When controlling for prediction margin and neighborhood size, the difference in correction rates between False Positives and False Negatives will significantly attenuate compared to the pilot's unadjusted rates (86.7% vs. 20.0%).
- **H3 (Stochastic Instability)**: GNNExplainer continuous-mask optimization will continue to exhibit substantial multi-seed Jaccard dispersion, remaining well below deterministic alignment ($J < 0.60$) despite exceeding theoretical random expectation ($J_{\mathrm{rnd}} \approx 0.15$).

---

## 2. Sampling and Eligibility Rules

### 2.1 Eligibility Criteria
A node $v$ in `validation_graph.pt` is eligible if:
1. It belongs to validation time steps: $\mathrm{time\_step}(v) \in [33, 40]$ and $\mathrm{val\_mask}(v) == \mathrm{True}$.
2. It has ground-truth supervision: $y(v) \in \{0, 1\}$ (unknown labels $-1$ excluded).
3. Its 2-hop directed computation subgraph (`flow="source_to_target"`) contains candidate edges: $E_{\mathrm{cand}}(v) \ge 2$.

### 2.2 Eligible Population Counts (Validation Steps 33–40)
- **True Positives (TP)**: 226 eligible nodes
- **False Positives (FP)**: 483 eligible nodes
- **False Negatives (FN)**: 44 eligible nodes
- **True Negatives (TN)**: 2,698 eligible nodes
- **Total Eligible**: 3,451 nodes

### 2.3 Pre-Registered Sample (N=80)
Using PRNG seed `20261020` without replacement, exactly 20 nodes were sampled from each quadrant:
- **20 True Positives (TP)**: Mean $E_{\mathrm{cand}} = 4.65$ (range 2–53); Mean margin = $0.2351$.
- **20 False Positives (FP)**: Mean $E_{\mathrm{cand}} = 7.25$ (range 2–34); Mean margin = $0.1314$.
- **20 False Negatives (FN)**: Mean $E_{\mathrm{cand}} = 5.95$ (range 2–45); Mean margin = $0.2129$.
- **20 True Negatives (TN)**: Mean $E_{\mathrm{cand}} = 15.80$ (range 2–141); Mean margin = $0.3964$.
- **Total Candidate Edges across 80 nodes**: **673 edges**.

---

## 3. Intervention Protocol & Removal Budget

Every method evaluated on node $i$ strictly uses an identical edge removal budget:
$$k_i = \max(1, \lceil 0.25 \times E_{\mathrm{cand}, i} \rceil)$$

### Evaluated Methods:
1. **GNNExplainer (3 seeds)**:
   - Optimized for 50 epochs with `node_mask_type=None`, `edge_mask_type="object"`.
   - Node features (93 cols) and 16-dim sinusoidal time encodings remain fixed and unmasked.
   - Evaluated across 3 seeds: 42, 43, 44.
   - Candidate edges ranked descending by continuous mask weight (stable tie-breaking).
2. **Uniform-Random Edge Deletion (30 draws)**:
   - 30 independent draws of size $k_i$ without replacement per node.
   - Deterministic seed sequence: `20261020 + r * 1000 + (node_id % 1000)`.
3. **Source Out-Degree Ranking**:
   - Edges ranked descending by the source transaction's out-degree in `validation_graph.pt`.
   - Evaluates whether simple structural hub fan-out explains attribution.

---

## 4. Pre-Registered Metrics and Statistical Aggregation

### 4.1 Unit of Analysis: Node-Level Macro Means
Repeated seeds (3 for GNNExplainer, 30 for random) are **within-node repetitions**. They must **not** be treated as independent transactions.

For node $i$:
$$\bar{\Delta}_{\mathrm{GE}, i} = \frac{1}{3} \sum_{s=1}^3 \Delta_{\mathrm{GE}, i, s}, \quad \bar{\Delta}_{\mathrm{Rnd}, i} = \frac{1}{30} \sum_{r=1}^{30} \Delta_{\mathrm{Rnd}, i, r}$$
$$\text{Paired Drop Advantage: } D_i = \bar{\Delta}_{\mathrm{GE}, i} - \bar{\Delta}_{\mathrm{Rnd}, i}$$

### 4.2 Statistical Significance & Confidence Intervals
- Mean Paired Advantage across $N$ nodes:
  $$\bar{D} = \frac{1}{N} \sum_{i=1}^N D_i, \quad \mathrm{SE} = \frac{s_D}{\sqrt{N}}$$
  Report 95% Confidence Interval: $\bar{D} \pm t_{0.975, N-1} \times \mathrm{SE}$.
- Stratified 95% CIs reported separately for TP, FP, FN, and TN.

### 4.3 Correction and Error-Induction Definitions
- **FP Correction**:
  - Run-level rate: $m_{\mathrm{flip}} / (3 \times N_{\mathrm{FP}})$.
  - Any-seed rate: fraction of FP nodes with $\ge 1$ seed correcting the error.
  - Majority-seed rate: fraction of FP nodes with $\ge 2$ seeds correcting the error.
  - All-seed rate: fraction of FP nodes with all 3 seeds correcting the error.
- **FN Correction**: Defined symmetrically ($P_{\mathrm{illicit}}$ rising above 0.50).
- **TP and TN Error Induction**: Defined as perturbation causing a previously correct prediction to cross the 0.50 threshold into an error.

### 4.4 Stability Reference
- Pairwise Jaccard similarity across the 3 explainer seeds:
  $$J_{\mathrm{GE}, i} = \frac{1}{3} \sum_{s < s'} \frac{|S_{i,s} \cap S_{i,s'}|}{|S_{i,s} \cup S_{i,s'}|}$$
- Compared against the **exact theoretical hypergeometric random overlap**:
  $$\mathbb{E}[J_{\mathrm{rnd}, i}] = \sum_{x=\max(0, 2k_i - E_i)}^{k_i} \frac{x}{2k_i - x} \frac{\binom{k_i}{x}\binom{E_i - k_i}{k_i - x}}{\binom{E_i}{k_i}}$$
  and the empirical 30-draw random Jaccard.
