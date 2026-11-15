# Frozen final evaluation protocol

- Dataset: original Elliptic transaction dataset.
- Training: time steps 1–32; validation: 33–40; test: 41–49.
- Supervised metrics include only labelled nodes; unknown=-1 is excluded.
- Feature scaler was fitted using time steps 1–32 only.
- Graph snapshots include only nodes and edges available by each cutoff.
- Primary metric: illicit-class AUPRC.
- Secondary metrics: ROC-AUC, illicit precision, recall, and F1.
- Report F1 using the fixed 0.5 threshold for every model.
- Clearly label any validation-tuned threshold results separately.
- Evaluate the held-out test period once after model choices are frozen.
- Include Random Forest, XGBoost, GCN, GraphSAGE, GAT, and ETGNN.
- Include matched/no-edge/local-feature analyses as ablations.
- Report that model and ablation choices were explored extensively
  using the validation period; test-period class counts were inspected during EDA, but no model was fitted or evaluated on test labels.
- Do not claim statistical significance or verified explanation accuracy.

## Prediction timing limitation
Edges and precomputed neighborhood features within each time step are
treated as available at the end of that time step. These experiments
do not demonstrate instant prediction when a transaction first arrives.
