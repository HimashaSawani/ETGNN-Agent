# Literature review

## Elliptic and fraud baselines

Weber et al. (2019) introduced the Elliptic transaction graph for anti-money
laundering research. Its 203,769 transaction nodes are connected by 234,355
directed payment-flow edges and organized into 49 time steps. The published 166
features comprise time and transaction descriptors, including 72 one-hop
neighborhood aggregates. Their experiments compared logistic regression,
Random Forest, multilayer perceptrons, and GCN variants, with Random Forest the
strongest reported classifier. This matters because a graph model should be
compared with strong tabular baselines and because the full-feature Random
Forest already receives graph-derived aggregates.

Random Forest combines randomized decision trees (Breiman, 2001). XGBoost is a
scalable gradient-boosting system with sparsity-aware learning (Chen and
Guestrin, 2016). Both provide important nonlinear tabular comparisons. Neither
is described here as graph-information-free when using all Elliptic features.

## Graph representation learning

GCN uses localized graph convolution to combine node features and structure
(Kipf and Welling, 2017). GraphSAGE learns neighborhood aggregation functions
that can generate representations for unseen nodes (Hamilton et al., 2017).
GAT assigns learned attention weights within graph neighborhoods (Velickovic et
al., 2018). These methods motivate the three graph baselines. ETGNN retains GAT
message passing and concatenates a deterministic sinusoidal encoding of the
discrete time step with each node's numeric features.

This mechanism is narrower than temporal graph learning in the literature.
EvolveGCN uses recurrent units to evolve GCN parameters across a graph sequence
(Pareja et al., 2020), while Temporal Graph Networks combine memory modules and
graph operators over timed event sequences (Rossi et al., 2020). ETGNN has
neither recurrent state nor event memory. Repository checks show that all
Elliptic edges used here join nodes in the same time step. The method is best
described as a time-feature-augmented GAT operating within completed snapshots.

## Imbalance, time, and evaluation

Only a minority of labelled Elliptic transactions are illicit. Precision-recall
analysis focuses on positive-class retrieval and is informative under class
imbalance (Saito and Rehmsmeier, 2015), motivating AUPRC as the primary metric.
The chronological split places training before validation and validation before
test. This tests later-period generalization and avoids future labels in model
selection, but a single future holdout does not identify the cause of drift.

## Explanation Methods and Evaluation Paradigms

GNNExplainer optimizes continuous feature and edge masks associated with a
model prediction by maximizing mutual information between the original prediction
and predictions over masked subgraphs (Ying et al., 2019). While widely utilized,
compact mask weights describe local model associations rather than establishing
causal error mechanisms or ground-truth explanation faithfulness.

In the broader interpretability literature, Hooker et al. (2019) demonstrated
via the ROAR framework that feature importance attributions must be evaluated
against explicit random degradation baselines, because uncalibrated degradation
often reflects non-specific model sensitivity rather than true explanation quality.
In the graph domain, subsequent benchmarking studies (e.g., Luo et al., 2020;
Amara et al., 2022) formalized deletion fidelity (measuring prediction degradation
under the omission of top-ranked explanation edges versus unranked or random edges).
However, prior literature predominantly restricts explanation evaluation to two
paradigms: (a) evaluating synthetic graphs with known ground-truth motifs, or
(b) self-evaluating correct classifications on static benchmarks.

### Contrast 1 in Context: Shared-Intervention Model Comparisons
Most GNN explainability studies evaluate each model against its own explanations.
Contrast 1 (C1) introduces a shared-intervention design: subjecting a baseline
model (Matched zero-time GAT) to the exact top-mask edge deletions identified
by a time-augmented model (ETGNN) on the same nodes and target class, while
calibrating each model against its own 30-draw seeded random edge deletion baseline.
This evaluates whether adding sinusoidal time encodings alters local structural
reliance beyond parameter-matched graph attention alone. In our frozen sample,
the observed mean random-adjusted difference was small ($\Delta_{adj} = 0.002903$,
sample SD $\pm 0.096180$); equivalence was not established.

### Contrast 2 in Context: Paired Error-vs-Correct Explanations
Explanation evaluation in graph anomaly detection has rarely examined whether
explanation sensitivity behaves differently between correct classifications and
systematic model errors. Contrast 2 (C2) addresses this question by constructing
matched pairs between erroneous classifications (False Positives, False Negatives)
and correct classifications (True Positives, True Negatives), strictly matched
on prediction margin ($\text{margin}(v) = |p_{\text{illicit}}(v) - 0.50|$, with pair eligibility condition $|\text{margin}(\text{incorrect}) - \text{margin}(\text{correct})| \le 0.050$) and candidate computation-neighbourhood edges (protocol-configured caliper $|E_{cand}(\text{incorrect}) - E_{cand}(\text{correct})| \le 5$, with smaller differences $\le 2$ observed in selected ETGNN TP–FP pairs, alongside exact out-degree matching in the sensitivity register).
Within the matched sample, the explanation-minus-random response was larger for
false positives than true positives (mean paired difference $+0.039343$, 95% marginal
CI $[+0.020234, +0.057410]$), while licit-domain FN–TN comparisons remained
statistically imprecise due to small cohort sizes ($N=25$).

## Provisional Research Gap and Epistemic Boundaries

No directly equivalent study was identified in the reviewed literature that
evaluates fixed sinusoidal time encodings in GAT against parameter-matched zero-time
controls on Elliptic under strict chronological splits, alongside shared-intervention
between-model contrasts (C1) and margin/degree-matched paired error contrasts (C2).

However, this research gap is strictly provisional and evidence-bounded:
1. *Unresolved Algorithm Comparisons:* We benchmarked GNNExplainer with continuous
   mask optimization; whether alternative explainers (e.g., PGExplainer, SubgraphX)
   exhibit similar error-versus-correct response disparities remains unresolved.
2. *Unresolved Architectural Breadth:* Evaluations examined a single frozen checkpoint
   pair across the validation horizon; whether these findings generalize across
   different random initializations or distinct temporal GNN families (e.g., EvolveGCN,
   TGN) is not empirically demonstrated.
3. *Absence of Prior Work:* The absence of an identical experimental setup in existing
   literature reflects a localized empirical investigation rather than a theoretical
   proof of a foundational paradigm gap.

## References

Amara, K., Ying, Z., Zhang, Z., Han, Z., Zhao, Y., Shan, Y., Brandes, U., Schemm, S., & Zhang, C. (2022). GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks. *Proceedings of the First Learning on Graphs Conference*, in *Proceedings of Machine Learning Research*, 198:44:1-44:23. https://proceedings.mlr.press/v198/amara22a.html

Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5-32.
https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
*Proceedings of KDD 2016*, 785-794. https://doi.org/10.1145/2939672.2939785

Hooker, S., Erhan, D., Kindermans, P. J., & Been, K. (2019). A benchmark for
interpretability methods. *Advances in Neural Information Processing Systems (NeurIPS 2019)*,
32. https://proceedings.neurips.cc/paper/2019/hash/fe4b8555ed8f8ab149aa792f6da600f4-Abstract.html

Luo, D., Cheng, W., Xu, D., Yu, F., Zong, B., Chen, H., & Zhang, X. (2020).
Parameterized explainer for graph neural networks. *Advances in Neural Information
Processing Systems (NeurIPS 2020)*, 33, 19620-19631.
https://proceedings.nips.cc/paper_files/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation
learning on large graphs. *Advances in Neural Information Processing Systems
30*. https://proceedings.neurips.cc/paper/6703-inductive-representation-learning-on-large-graphs

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph
convolutional networks. *ICLR 2017*. https://arxiv.org/abs/1609.02907

Pareja, A., et al. (2020). EvolveGCN: Evolving graph convolutional networks for
dynamic graphs. *AAAI, 34*(04), 5363-5370.
https://doi.org/10.1609/aaai.v34i04.5984

Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M.
(2020). Temporal graph networks for deep learning on dynamic graphs.
https://arxiv.org/abs/2006.10637

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more
informative than the ROC plot when evaluating binary classifiers on imbalanced
datasets. *PLOS ONE, 10*(3), e0118432.
https://doi.org/10.1371/journal.pone.0118432

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y.
(2018). Graph attention networks. *ICLR 2018*.
https://arxiv.org/abs/1710.10903

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson,
T., & Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting
with graph convolutional networks for financial forensics.
https://arxiv.org/abs/1908.02591

Ying, Z., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019).
GNNExplainer: Generating explanations for graph neural networks. *Advances in
Neural Information Processing Systems 32*.
https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html
