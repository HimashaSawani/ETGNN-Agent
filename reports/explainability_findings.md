# Explainability findings

## Scope and definitions

Initial exploratory analysis examined GNNExplainer on four hand-selected ETGNN
predictions from validation time steps 33-40: one true positive (TP), false
positive (FP), false negative (FN), and true negative (TN). This was subsequently
expanded to a frozen multi-register benchmark across 377 completed validation
transactions (time steps 37–40, following exploratory development on steps 33–36)
comparing ETGNN against a parameter-matched zero-time GAT.

For an original predicted class $c$, the signed deletion response is:
$$p(c \mid \text{original graph}) - p(c \mid \text{graph after deletion})$$
A positive value means the deletion reduced support for $c$; a negative value means
its probability increased. This perturbation response is an empirical measure of
model sensitivity under structural edge omission, not proof of explanation quality,
harmful-edge identification, or causal error correction.

Mask sparsity at threshold $t$ is $1 - \text{count}(\text{mask} \ge t) / \text{count}(\text{mask})$.
Because GNNExplainer masks are continuous, 0.3, 0.5, and 0.7 are descriptive
sensitivity points rather than natural cutoffs.

---

## Initial Four-Node Case Audit

- **Mask integrity:** Every center-feature mask contains 165 numeric entries. Edge masks equal subgraph edge counts: TP 90, FP 159, FN 11, TN 154.
- **Top-five edge deletion:** Lowered predicted-class probability for FP (0.1217, causing class flip) and TN (0.0245), but increased probability for TP (-0.0125) and FN (-0.0509).
- **Controlled comparison:** Top deletion was clearly more disruptive than bottom or seeded-random deletion only for FP. Across the other three cases, responses were mixed.

---

## Main Multi-Register C1 & C2 Benchmark

### 1. Subgraph Equivalence Assertion Coverage
During the main benchmark execution, runtime assertions verified that 2-hop
computation subgraph forward passes reproduced full-graph prediction probabilities
within a tolerance of $10^{-5}$ across all 377 completed nodes for both models.
An additional stratified verification sample ($N=26$ across candidate neighborhood
sizes $E_{cand} \in \{2..7\}$) observed discrepancies bounded by $3.58 \times 10^{-7}$
for ETGNN and $8.94 \times 10^{-8}$ for Matched GAT.

### 2. Contrast 1: Response to Shared ETGNN-Ranked Deletions
Contrast 1 evaluates the random-adjusted deletion response difference between
ETGNN and Matched GAT across $N = 240$ transactions under shared ETGNN top-mask deletions:
$$\Delta_{adj} = \text{Adv}_{ETGNN} - \text{Adv}_{Matched\_GAT}$$
where $\text{Adv}_M = \Delta_{GE, M} - \frac{1}{30}\sum_{d=1}^{30}\Delta_{Rnd, M}^{(d)}$,
with random deletion serving as a baseline comparator.
The unadjusted raw difference is $\Delta_{raw} = \Delta_{GE, ETGNN} - \Delta_{GE, Matched\_GAT}$.

- **Adjusted Difference:** The observed mean random-adjusted response difference was small in this selected sample of frozen checkpoints: $\text{mean } \Delta_{adj} = 0.002903$ (sample standard deviation $s = \pm 0.096180$ across transactions, median = $-0.001534$, IQR = $0.063236$).
- **Raw Difference:** $\text{mean } \Delta_{raw} = 0.038491$ ($s = \pm 0.139170$, median = $0.006023$).
- **Sample Means:** Mean advantage over random comparator was $0.020239$ for ETGNN and $0.017336$ for Matched GAT.
- These results do not establish that the models are equivalent, nor do they demonstrate a causal time-encoding effect. Any additional C1 bootstrap analysis must be labelled exploratory.

### 3. Eligible Edges and Ceiling Deletion Rule
The frozen deletion protocol enforced the discrete ceiling rule:
$$k = \max(1, \lceil 0.25 \times E_{cand} \rceil)$$
where $E_{cand}$ denotes eligible edges in the defined 2-hop computation neighbourhood.
Because neighbourhoods are sparse:
- **317 out of 377 transactions (84.1%) have exactly $E_{cand} = 2$.**
- Deleting 1 edge ($k = 1$) from a 2-edge neighbourhood imposes an **actual deletion fraction of 50.0%**, a 2.0× inflation over the nominal 25% target.
- For $E_{cand} = 3$ ($N=26$): $k = 1$ (actual 33.3%).
- For $E_{cand} = 4$ ($N=18$): $k = 1$ (actual 25.0%).
- For $E_{cand} = 5$ ($N=10$): $k = 2$ (actual 40.0%).
- For $E_{cand} = 6$ ($N=2$): $k = 2$ (actual 33.3%).
- For $E_{cand} = 7$ ($N=4$): $k = 2$ (actual 28.6%).
- Population mean actual deletion fraction was $47.08\%$ (median $50.0\%$).
Perturbations reflect discrete edge removals on sparse subgraphs.

### 4. Contrast 2: Matched-Pair Bootstrap Across All 8 Registers
Contrast 2 evaluates within-node explainer advantages across matched pairs under
Primary Caliper Matching and Degree-Balanced Sensitivity Matching.
The implemented paired difference contrast is:
$$\Delta_{pair} = A(\text{incorrect}) - A(\text{correct})$$
- For FP–TP pairs: $\Delta_{pair} = A(\text{FP}) - A(\text{TP})$
- For TN–FN pairs: $\Delta_{pair} = A(\text{FN}) - A(\text{TN})$

#### Worked Pair Example from Saved Records
In pair `ETGNN_full_TP_FP_001`:
- Treat Node: $u = 148295$ (FP, time step 37). Advantage $A(\text{FP}) = -0.013590 - 0.020087 = -0.033677$.
- Control Node: $v = 152012$ (TP, time step 38). Advantage $A(\text{TP}) = 0.046399 - 0.023608 = +0.022791$.
- Paired Difference: $\Delta_{pair} = A(\text{FP}) - A(\text{TP}) = -0.033677 - (+0.022791) = -0.056467$.

Across the matched registers, pairs were resampled using matched-pair bootstrap
($B = 2,000$ iterations, seed `20261020`). Preserving pairs accounts for within-pair
correlation, but does not resolve spatial dependence between pairs sharing graph context.
All eight confidence intervals are marginal and unadjusted for multiple testing.

| Register | Match Type | Pairs ($N$) | Mean Paired Diff | Bootstrap SE | 95% Marginal CI | Zero Crossed? |
|---|---|---:|---:|---:|---:|:---:|
| ETGNN Primary TP–FP | Primary | 95 | +0.039343 | 0.009585 | [+0.020234, +0.057410] | No |
| ETGNN Degree-Sensitivity TP–FP | Degree | 57 | +0.043455 | 0.012658 | [+0.019882, +0.069552] | No |
| Matched GAT Primary TP–FP | Primary | 94 | +0.020315 | 0.008923 | [+0.003917, +0.038684] | No |
| Matched GAT Degree-Sensitivity TP–FP | Degree | 82 | +0.010547 | 0.011610 | [-0.013240, +0.032770] | **Yes** |
| ETGNN Primary TN–FN | Primary | 25 | -0.058027 | 0.034398 | [-0.128831, +0.006393] | **Yes** |
| ETGNN Degree-Sensitivity TN–FN | Degree | 25 | -0.019748 | 0.029750 | [-0.081593, +0.036312] | **Yes** |
| Matched GAT Primary TN–FN | Primary | 21 | +0.052255 | 0.051517 | [-0.046153, +0.162460] | **Yes** |
| Matched GAT Degree-Sensitivity TN–FN | Degree | 21 | +0.072489 | 0.051881 | [-0.024721, +0.178587] | **Yes** |

#### Empirical Interpretations:
1. **Direction of ETGNN FP–TP Difference:**
   Within the matched sample, the explanation-minus-random deletion response was **larger for false positives than true positives**
   ($\text{mean } A(\text{FP}) - A(\text{TP}) = +0.039343$ in primary matching, $+0.043455$ in degree-sensitivity matching).
   Neither interval crosses zero. Top-mask edge deletions reduced predicted-class probabilities more sharply for false positives than true positives.
   This must not be described as proof of explanation quality, identification of harmful edges, or successful error correction.
2. **Matching Specification Sensitivity in Matched GAT:**
   In Matched GAT, primary matching yielded a positive estimate ($+0.020315$, 95% CI $[+0.003917, +0.038684]$),
   whereas in degree-sensitivity matching the estimate attenuated to $+0.010547$ with its 95% CI crossing zero
   ($[-0.013240, +0.032770]$; $-0.013240$ is the lower bound of the CI, not the mean). Because primary and
   degree-sensitivity registers represent different, overlapping matched samples, changes cannot be attributed
   solely to degree adjustment, nor does zero-crossing status prove that the two models differ significantly.
3. **Statistical Imprecision in False Negative (FN) Comparisons:**
   All four TN–FN registers yield wide confidence intervals crossing zero due to small sample sizes ($N = 25$ for ETGNN,
   $N = 21$ for Matched GAT). Non-significance reflects sample-size limitations and must not be interpreted as functional equality.

---

## Methodological Limitations

1. **Previously Used Validation Data:** The 377 transactions evaluated in C1 and C2 were sampled from the
   chronological validation period (steps 37–40, following steps 33–36 used for development), where model checkpoints
   and thresholds were selected. Furthermore, historical test split performance (steps 41–49) was previously inspected.
   These evaluations are structured follow-up audits, not uninspected prospective trials.
2. **Single Checkpoint per Model:** Evaluations used a single checkpoint per model family, conflating inductive bias
   with checkpoint-specific stochasticity.
3. **Fixed Precomputed Aggregate Features:** 72 of the 165 node features are precomputed neighborhood aggregates;
   deleting graph edges perturbs message passing but leaves tabular aggregates static.
4. **Graph Dependence:** Transactions belong to a connected graph; matched-pair bootstrap resamples transaction pairs
   without fully resolving spatial autocorrelation across subgraphs.
5. **Multiple Marginal Confidence Intervals:** The 8 bootstrap CIs are marginal and unadjusted for family-wise error rates.
6. **No Claims of Causal Correction:** Deletion perturbation measures model sensitivity under edge omission; it does
   not demonstrate causal error correction, explanation faithfulness, or ground-truth explanation validity. Non-significant
   differences must not be conflated with equivalence.
