# Research questions, hypotheses, and contributions

## Research objective

Evaluate whether adding a fixed representation of transaction time to a graph
attention classifier improves chronological illicit-transaction classification
on Elliptic, and compare the result with graph and tree-based baselines without
overstating the temporal or explanatory capabilities of the implementation.

## Research questions

1. How does the implemented time-encoded GAT (ETGNN) compare with Random Forest,
   XGBoost, GCN, GraphSAGE, and GAT on held-out future Elliptic time steps when
   illicit-class AUPRC is the primary metric?
2. On validation data, does fixed sinusoidal time encoding improve AUPRC over a
   parameter-matched zero-time GAT across the tested seeds?
3. What do local GNNExplainer masks and controlled edge-deletion perturbations
   reveal about selected validation TP, FP, FN, and TN predictions, and what
   limitations prevent causal or population-level interpretation?

## Hypotheses and outcomes

- H1: ETGNN will outperform the strongest baseline on held-out AUPRC. Not
  supported: ETGNN achieved 0.2250 versus Random Forest's 0.6323.
- H2: Time encoding will improve validation AUPRC over a matched zero-time GAT.
  Partially supported: ETGNN was higher for two of three paired seeds and had a
  higher three-seed mean (0.5292 versus 0.4781), but no significance test or
  repeated test-set study supports a general effect claim.
- H3: High-mask edges will consistently support selected predictions. Not
  supported in the four cases: top-edge deletion reduced predicted-class
  probability for FP and TN but increased it for TP and FN.

## Evidence-bounded contribution

The project contributes a reproducible chronological comparison, validation
ablations separating time encoding, local/precomputed features, and graph
edges, plus a cautious local explanation audit. It does not establish a novel
temporal-GNN architecture, superior fraud detection, causal explanations, or
generalization beyond Elliptic.
