# Methodology

## Research design

This study evaluates illicit Bitcoin transaction classification using
the original Elliptic dataset. The proposed ETGNN is compared with
Random Forest, XGBoost, GCN, GraphSAGE, and GAT. The primary outcome
is illicit-class area under the precision-recall curve (AUPRC).
Precision, recall, F1-score, and ROC-AUC are secondary metrics.

## Title and Architectural Framing: "Adaptive" Check

The codebase implements a **time-encoded static Graph Attention Network**,
not a dynamic, recurrent, or continuous-time event-memory temporal GNN
(such as TGN, JODIE, or DyRep). Specifically:
- Time is represented via a fixed 16-dimensional sinusoidal projection of the
  discrete transaction time step index ($t \in \{1 \dots 49\}$) concatenated
  to static node features prior to standard GAT layers.
- The model weights remain completely static during evaluation; there is no
  runtime weight adaptation, continual learning, online drift update, or
  dynamic graph rewiring mechanism.
- Consequently, any characterization of the model as "Adaptive" is **empirically
  unsupported by the implemented architecture**.
- **Supervisor Recommendation:** We formally recommend revising any prospective
  thesis or paper title containing "Adaptive" to:
  > *"Explainable Time-Encoded Graph Attention Networks for Bitcoin Illicit Transaction Classification"*
  This recommendation is flagged for formal supervisor and committee approval
  prior to document finalization.

## Dataset and labels

The dataset contains 203,769 transactions, 234,355 directed edges,
and 49 discrete time steps. Each transaction has an identifier, a time
step, and 165 numeric feature columns. The numeric columns consist
of 93 local features and 72 precomputed one-hop neighborhood
aggregates. Class 1 denotes illicit transactions and Class 2 denotes licit transactions;
"unknown" denotes unlabelled transactions that can supply features during
graph message passing but are excluded from supervised loss and evaluation metrics.
For binary model training and evaluation, Class 1 is mapped to target label $y = 1$
(illicit) and Class 2 is mapped to $y = 0$ (licit). Classification decisions are
evaluated at the canonical frozen decision threshold $\tau = 0.50$, where transactions
with model probability $p(\text{illicit}) > 0.50$ are predicted as illicit ($\hat{y} = 1$)
and those with $p(\text{illicit}) \le 0.50$ are predicted as licit ($\hat{y} = 0$).

Transaction identifiers were mapped to feature-row indices before
constructing graph edges. Duplicate identifiers, missing edge
endpoints, feature alignment, missing values, and infinite values
were checked. Every observed edge connects transactions within the
same time step; the dataset does not provide cross-time-step edges.

## Chronological evaluation & partition structure

Time steps 1-32 were used for training, 33-40 for validation, and
41-49 for held-out testing. The feature scaler was fitted using
nodes from training time steps only, then applied to all periods.
Separate graph snapshots were constructed using nodes and edges
available up to each period's cutoff. Supervised masks include only
labelled nodes and do not overlap.

### Follow-up Partitioning: Development vs. Main Benchmark
Downstream explainability and intervention audits were conducted strictly
within the validation window (time steps 33–40) rather than a pristine,
untouched holdout:
- **Development (Steps 33–36):** Used for initial plumbing, pipeline development,
  and pilot checks under the frozen canonical decision threshold $\tau = 0.50$
  (no threshold tuning was performed; the standard 0.50 threshold was retained).
- **Main C1 & C2 Benchmark (Steps 37–40):** Frozen follow-up audit evaluating
  377 completed transactions. Because these transactions originate from the
  validation period where model checkpoints were selected, these experiments
  constitute structured retrospective audits, not prospective test trials.

The experiment assumes that within-step graph edges and the released
precomputed features are available at the end of each time step. It
does not evaluate immediate prediction at transaction arrival.

## Baselines and proposed model

Random Forest and XGBoost use transaction feature vectors. Since
the full feature set contains neighborhood aggregates, these
baselines are not entirely free of graph-derived information.
A local-feature-only Random Forest provides a separate ablation.

GCN, GraphSAGE, and GAT use the aligned graph and chronological
snapshots. The proposed ETGNN has two graph-attention layers. A fixed
16-dimensional sinusoidal encoding of each transaction's time step
is concatenated with its numeric features before graph attention.
The encoding can be calculated for time steps unseen during training.
The graph-attention layers exchange messages only along the observed
within-step edges. ETGNN does not maintain a recurrent graph state
or model observed edges between different time steps.

GNN models were trained with class-weighted cross-entropy and Adam.
Checkpoints were selected using validation AUPRC with early stopping.
A single frozen checkpoint per model family was selected for final test
evaluation and explanation benchmarking.

## Explainability and Intervention Protocols (C1 and C2)

Post-hoc explainability was investigated using GNNExplainer on extracted
two-hop computation subgraphs:
- **Main Run Subgraph Equivalence Coverage:** All 377 completed evaluation
  transactions underwent runtime forward-pass equivalence assertions verifying
  that 2-hop computation subgraph probabilities match full-graph predictions
  within a strict tolerance of $10^{-5}$ on both ETGNN and Matched GAT.
- **Verification Sample Bounds:** An additional stratified verification across
  all candidate neighborhood sizes ($E_{cand} \in \{2..7\}$) observed maximum
  discrepancies bounded by $3.58 \times 10^{-7}$ (ETGNN) and $8.94 \times 10^{-8}$
  (Matched GAT) within that tested sample.

### Contrast 1 Protocol: Shared ETGNN-Ranked Interventions
Contrast 1 (C1) evaluates $N = 240$ transactions from steps 37–40 to measure
the differential deletion response between ETGNN and Matched GAT.
Crucially, **C1 evaluates responses to shared ETGNN-ranked deletion sets**,
not each model's own independently chosen edge rankings:
- Both models are subjected to the identical deletion of the top $k$ edges
  selected by GNNExplainer on ETGNN.
- Both models are evaluated on the identical target class (ETGNN's predicted class).
- For each model, random deletion serves as a baseline comparator estimating
  average response under 30 seeded random edge draws under identical budget $k$.
- The primary outcome is the random-adjusted difference:
  $$\Delta_{adj} = (\Delta_{GE, ETGNN} - \Delta_{Rnd, ETGNN}) - (\Delta_{GE, Matched\_GAT} - \Delta_{Rnd, Matched\_GAT})$$
- The observed mean random-adjusted response difference was small ($\Delta_{adj} = 0.002903$).
  The sample standard deviation across transactions is $s = \pm 0.096180$ (variance
  across the 240 transactions, not a standard error or confidence interval).
- Any additional bootstrap hypothesis testing on C1 must be designated as exploratory.

### Contrast 2 Protocol: Within-Model Advantages & Matched Pairs
Contrast 2 (C2) tests whether GNNExplainer edge masks yield larger deletion
responses on erroneous classifications than on correct classifications within each model:
- Unlike C1, in Contrast 2 **Matched GAT uses its own GNNExplainer edge ranking
  and its own predicted target class** to compute its own within-model advantage.
- Within-node advantage is defined as:
  $$\text{Adv}_M(u) = \Delta_{GE, M}(u) - \frac{1}{30}\sum_{d=1}^{30}\Delta_{Rnd, M}^{(d)}(u)$$
- Pairs are constructed between True Positives and False Positives (TP–FP) and
  True Negatives and False Negatives (TN–FN) using caliper matching on predicted
  probability margin and candidate edge count ($N = 420$ pairs across 8 registers).
- For any transaction node $v$, its prediction margin is explicitly defined as its distance from the decision boundary:
  $$\text{margin}(v) = |p_{\text{illicit}}(v) - 0.50|$$
- The pair eligibility condition for candidate matching between an incorrect transaction and a correct transaction within the same time step is:
  $$|\text{margin}(\text{incorrect}) - \text{margin}(\text{correct})| \le 0.050$$
  and candidate computation-neighbourhood edge caliper:
  $$|E_{cand}(\text{incorrect}) - E_{cand}(\text{correct})| \le 5$$
  Note that the matching condition is $|\text{margin}(\text{incorrect}) - \text{margin}(\text{correct})| \le 0.050$, not $|p_{\text{illicit}} - 0.50| \le 0.050$ (which defines individual node margin). Furthermore, the candidate-edge caliper $\le 5$ is the protocol-configured eligibility bound, which is distinguished from the smaller differences observed in selected pairs (for example, in the primary ETGNN TP–FP register, observed candidate-edge differences were bounded within $\le 2$).
- The paired difference contrast is defined as:
  $$\Delta_{pair} = A(\text{incorrect}) - A(\text{correct})$$
  For FP–TP pairs, $\Delta_{pair} = A(\text{FP}) - A(\text{TP})$. A positive estimate
  indicates that the explanation-minus-random deletion response was larger for
  false positives than true positives.
- Two matching specifications are evaluated within the same time step using Hopcroft-Karp maximum bipartite matching:
  1. *Primary Caliper Matching:* Matches treated (incorrect) and control (correct)
     transactions subject to the configured protocol calipers:
     $$|\text{margin}(\text{incorrect}) - \text{margin}(\text{correct})| \le 0.050$$
     and
     $$|E_{cand}(\text{incorrect}) - E_{cand}(\text{correct})| \le 5$$
     (retaining the source-confirmed candidate-edge caliper $\le 5$, distinguished from the smaller differences bounded within $\le 2$ observed in selected ETGNN TP–FP pairs).
  2. *Degree-Balanced Sensitivity Matching:* Enforces **exact out-degree matching**
     ($\text{out\_degree}_{treat} = \text{out\_degree}_{ctrl}$ for $100\%$ of pairs)
     under the identical configured calipers on prediction margin ($|\text{margin}(\text{incorrect}) - \text{margin}(\text{correct})| \le 0.050$)
     and candidate edge count ($|E_{cand}(\text{incorrect}) - E_{cand}(\text{correct})| \le 5$).
- Uncertainty is estimated via matched-pair cluster bootstrap ($B = 2,000$ iterations,
  seed `20261020`), resampling transaction pairs as clusters. Preserving pairs
  accounts for within-pair correlation, but does not resolve spatial dependence
  between different pairs sharing graph context. The 8 intervals are marginal
  and unadjusted for multiple comparisons.

### Structural Discreteness in Deletion Budgets
The protocol specified a nominal deletion fraction $\alpha = 25\%$ of eligible
edges in the defined 2-hop computation neighbourhood ($E_{cand}$). The discrete
ceiling removal rule is:
$$k = \max(1, \lceil \alpha \cdot E_{cand} \rceil)$$
- 317 out of 377 transactions (84.1%) have exactly $E_{cand} = 2$.
- Deleting 1 edge ($k = 1$) from these neighbourhoods imposes an actual deletion
  fraction of $k / E_{cand} = 50.0\%$, a two-fold inflation over the nominal 25% rule.
- Deletion perturbations must be understood as discrete removals on sparse
  subgraphs, not infinitesimal marginal perturbations.

### Methodological & Epistemic Constraints
1. **Static Aggregate Features:** 72 of the 165 node features are precomputed
   one-hop neighborhood aggregates. Deleting graph edges perturbs message
   passing during GNN inference but leaves the precomputed tabular aggregates
   intact, confounding topological and feature attribution.
2. **Graph Dependence:** Transactions reside in a connected graph structure.
   While cluster bootstrap resamples pairs, spatial autocorrelation across
   subgraphs is not fully eliminated.
3. **Multiple Marginal Testing:** The eight bootstrap intervals are marginal
   estimates without family-wise error rate corrections.
4. **Epistemic Boundaries:** Deletion perturbations measure model sensitivity
   under edge removal; they do not establish causal explanations, ground-truth
   faithfulness, or automatic error correction. Non-significant differences
   do not demonstrate functional equivalence.
