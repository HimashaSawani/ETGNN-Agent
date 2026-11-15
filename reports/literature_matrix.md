# Literature matrix

| Source | Verified contribution | Relation to this implementation |
|---|---|---|
| Breiman (2001) | Defines Random Forest as an ensemble of randomized tree predictors. | Supports the strongest tabular baseline; this implementation uses class-balanced sampling and 100 trees. |
| Chen and Guestrin (2016) | Presents scalable gradient tree boosting with sparsity-aware learning. | Supports the XGBoost baseline with class weighting and validation early stopping. |
| Kipf and Welling (2017) | Introduces localized first-order graph convolution for semi-supervised node classification. | Basis for the GCN baseline on chronological graph snapshots. |
| Hamilton, Ying, and Leskovec (2017) | Introduces inductive neighborhood aggregation for unseen nodes. | Basis for GraphSAGE, although this study evaluates completed period snapshots rather than streaming deployment. |
| Velickovic et al. (2018) | Introduces masked self-attention over graph neighborhoods. | Direct architectural basis for GAT and the two-layer ETGNN. |
| Weber et al. (2019) | Releases Elliptic with 203,769 transaction nodes, 234,355 payment-flow edges, 49 time steps, and 166 published features; reports RF as strongest in its experiments. | Defines the dataset and motivates strong tree baselines. The published features include time and 72 precomputed neighborhood aggregates. |
| Ying et al. (2019) | Introduces GNNExplainer to identify compact subgraphs and feature subsets associated with a GNN prediction. | Supports local masks for four validation cases; no ground-truth explanation labels are available here. |
| Pareja et al. (2020) | EvolveGCN evolves GCN parameters with recurrent mechanisms for dynamic graphs. | Clarifies what ETGNN is not: ETGNN has fixed time encoding and no recurrent parameter evolution. |
| Rossi et al. (2020) | Temporal Graph Networks model dynamic graphs as sequences of timed events with memory and graph operators. | Clarifies that end-of-time-step ETGNN snapshots are not event-stream or memory-based temporal learning. |
| Saito and Rehmsmeier (2015) | Demonstrates the value of precision-recall analysis for imbalanced binary classification. | Supports illicit-class AUPRC as the primary metric; held-out prevalence is 5.25%. |

The matrix does not establish novelty. It positions the implemented method as a
GAT supplied with fixed time features and distinguishes it from recurrent,
memory-based, and event-stream temporal graph models.
