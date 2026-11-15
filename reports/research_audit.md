# Research completion audit

## Completed and verified

- Data alignment checks record 203,769 nodes, 234,355 within-time-step edges,
  49 time steps, and no missing edge endpoints.
- The chronological split is consistently train 1-32, validation 33-40, and
  held-out test 41-49; scaling was fitted on training-period nodes only.
- Frozen test results and per-time-step results are present. Random Forest is
  strongest by test AUPRC (0.6323); ETGNN reaches 0.2250.
- Full-feature provenance is documented: 165 numeric inputs comprise 93 local
  columns and 72 precomputed one-hop neighborhood aggregates.
- Validation ablations cover matched zero-time GAT seeds, local-only models,
  and an ETGNN no-edge control.
- Four validation GNNExplainer cases contain complete 165-value center-feature
  masks and edge masks matching their two-hop subgraphs.
- Top, bottom, and seeded-random edge deletion plus threshold-sensitive mask
  sparsity are reproducible from tracked scripts and tables.
- Research questions, literature matrix/review, methodology, results,
  discussion, limitations, conclusion, demo, and reproduction commands are
  documented.

## Resolved inconsistencies

- ETGNN is described as a GAT with fixed sinusoidal time encoding, not as a
  recurrent, event-stream, or online temporal GNN.
- Random Forest is not described as graph-information-free because the full
  Elliptic feature set contains neighborhood aggregates.
- Explanation probability changes are separated from claims of fidelity, and
  continuous-mask sparsity is reported at three explicit thresholds.
- Draft headings were removed from the finalized methodology and results files.
- Legacy text encodings were normalized so split ranges render consistently.
- The four saved full-mask JSON files are tracked inputs rather than being
  hidden by the generated-explanations ignore rule.

## Evidence gaps that remain

- No experiment identifies the cause of the large validation-to-test decline.
- Four selected explanation cases and one random control per case cannot
  establish explanation accuracy, stability, or population-level faithfulness.
- No ground-truth explanation annotations exist, and feature/time perturbation
  was not performed.
- One dataset and one frozen future test window do not establish external
  generalizability or statistical significance.
- A university-specific formatting rubric, required report length, and
  presentation/slides requirement are not available in the repository; those
  submission-format checks require the module specification.
