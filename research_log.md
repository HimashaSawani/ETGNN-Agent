# ETGNN Research Log

## 2026-09-25 — Project setup
- Created project folders on D: drive.
- Recorded a preliminary chronological split: train 1–32, validation 33–40, test 41–49.
- Created Python 3.13.3 virtual environment.
- Installed CPU PyTorch, PyTorch Geometric, and initial research libraries.
- Created requirements.txt, .gitignore, and README.md.
- Initialized the Git repository.

## Decisions to verify during dataset inspection
- Confirm the available time steps and class labels.
- Confirm how transaction IDs map to feature rows and graph edges.
- Define how each model avoids using future graph information.
- Keep unknown labels out of supervised training and evaluation.

## Dataset verification — 2026-09-25
- Features: 203,769 rows and 167 columns, without a header row.
- Edges: 234,355; classes: 203,769.
- Labels: 4,545 illicit (1), 42,019 licit (2), 157,205 unknown.
- Time steps: 1–49.
- No duplicate transaction IDs or missing edge endpoints.
- All 234,355 edges connect transactions within the same time step; there are no edges between time steps.
- The proposed train/validation/test periods each contain both labelled classes.
- Temporal model design must account for the absence of cross-time-step edges.

## Exploratory data analysis — 2026-09-25
- The feature file contains 165 numeric feature columns, plus transaction ID and time step.
- No missing or infinite feature values were found.
- Illicit proportion varies substantially by time step among labelled transactions.
- For example: time step 13 = 35.97% illicit; time step 46 = 0.28% illicit.
- These percentages exclude transactions labelled unknown.
- This variation makes chronological evaluation and per-time-step reporting important.

## Data preparation — 2026-09-25
- Mapped original transaction IDs to feature row indices.
- Mapped classes: illicit=1, licit=0, unknown=-1.
- Created non-overlapping chronological supervision masks.
- Prepared stage-specific edge lists so training cannot see future edges.

## Graph preparation — 2026-09-25
- Fitted feature scaler on time steps 1–32 only (130,628 nodes).
- Built separate train, validation, and test graph snapshots.
- Verified graph indexes, time cutoffs, and non-overlapping masks.
- No model trained or test result inspected at this stage.

## Traditional ML baselines — 2026-09-25
- Trained Random Forest on labelled training-period nodes only.
- Random Forest validation: F1=0.8750, AUPRC=0.9102 at threshold 0.5.
- Trained XGBoost with validation-based early stopping.
- These are validation results; the test period has not been evaluated.

## GCN baseline — 2026-09-25
- Two-layer GCN trained using training-period graph and labelled training nodes.
- Best checkpoint selected by validation AUPRC at epoch 5.
- Validation at threshold 0.5: F1=0.2674, AUPRC=0.2655.
- This large gap from Random Forest requires analysis; no final conclusion yet.

## Baseline model comparison - 2026-09-25
- Evaluated 5 baseline models on the validation set (threshold 0.5):
  - Random Forest: F1=0.8750, AUPRC=0.9102
  - XGBoost: F1=0.6578, AUPRC=0.8425
  - GCN: F1=0.2674, AUPRC=0.2655
  - GraphSAGE: F1=0.3665, AUPRC=0.5069
  - GAT: F1=0.2531, AUPRC=0.3722
- All reported scores are from validation time steps 33–40. The test period (41–49) has not been evaluated.

## ETGNN temporal model evaluation - 2026-09-25
- Implemented fixed sinusoidal time encoding (dimension 16) and combined it with 165 node features before graph attention layers.
- ETGNN trained using the same protocol as GAT on training period (time steps 1–32).
- Validation results comparison (time steps 33–40, threshold 0.5):
  - GAT (static baseline): best_epoch=3, precision=0.1492, recall=0.8336, F1=0.2531, ROC-AUC=0.8181, AUPRC=0.3722
  - ETGNN (with time encoding): best_epoch=40, precision=0.3910, recall=0.7706, F1=0.5188, ROC-AUC=0.9203, AUPRC=0.5488
- Adding explicit sinusoidal time-step encoding improved GAT performance in this validation run (AUPRC increased from 0.3722 to 0.5488, F1 from 0.2531 to 0.5188, Precision from 0.1492 to 0.3910).
- All reported scores are from validation time steps 33–40. The test period (41–49) has not been evaluated.


## Matched ablation — 2026-09-25
- Validation AUPRC: GAT=0.3722, matched zero-time GAT=0.4836,
  sinusoidal-time ETGNN=0.5488 in one run.
- Zero input columns do not add usable information and their connected
  weights receive no learning signal.
- Differences between single runs may reflect initialization and training
  variation. Repeat the matched comparison across seeds before attributing
  the improvement to temporal information.

## Repeated matched ablation — 2026-09-25
- Validation AUPRC, seeds 7/42/123:
  - ETGNN: 0.5096 / 0.5488 / 0.5293; mean 0.5292.
  - Matched zero-time GAT: 0.5572 / 0.4836 / 0.3934; mean 0.4781.
- ETGNN was higher for two of three paired seeds; matched GAT was higher for seed 7.
- This is indicative validation evidence, not a definitive temporal-effect
  or statistical-significance claim.

## Feature provenance correction — 2026-09-25
- Elliptic's 166 published features comprise 94 local features (including time step) and 72 one-hop neighborhood aggregates.
- After separating time step, our 165-column matrix contains 93 local numeric columns followed by 72 graph-aggregated columns.
- The existing Random Forest baseline uses precomputed graph information. It must not be described as a graph-information-free baseline.
- Run local-only feature ablations to assess the contribution of precomputed neighborhood aggregates.

## Local-feature edge ablation — 2026-09-25
- Validation, seed 42, 93 local numeric features plus time encoding:
  - ETGNN with edges: AUPRC=0.5791, F1@0.5=0.4759.
  - Same architecture with empty edges: AUPRC=0.8282, F1@0.5=0.4139.
  - Local-only Random Forest: AUPRC=0.9145, F1@0.5=0.8309.
- The no-edge control has higher AUPRC in this run. The with-edge
  model has higher F1 at the fixed 0.5 threshold.
- These results do not establish why edge message passing underperforms.
  Oversmoothing and noisy neighbors remain hypotheses, not findings.
- Limit further validation-driven model changes before final testing.

## One-time held-out test evaluation — 2026-09-25
- Evaluated frozen checkpoints on labelled nodes from time steps 41–49.
- Primary metric AUPRC: RF=0.6323, XGBoost=0.5472,
  GCN=0.1306, GraphSAGE=0.2427, GAT=0.1155,
  matched GAT=0.2099, ETGNN=0.2250.
- Ablations: RF local=0.6301, RF+time=0.6295,
  ETGNN local=0.2344, ETGNN local without edges=0.4812.
- ETGNN did not outperform the strongest Random Forest baseline.
- Test scores were lower than validation scores for several models.
  The cause of this drop has not been established.
- No model will be tuned using held-out test results.

## Demo integration — 2026-09-26
- FastAPI and Streamlit demo connected successfully.
- TP, FP, FN, and TN saved explanation examples passed API checks.
- Unknown transaction IDs return HTTP 404.
- Demo is restricted to existing Elliptic transactions.

## Explanation and research closeout - 2026-09-26
- Verified full center-feature masks contain 165 values and full edge masks
  match the four validation subgraph edge counts.
- Made top-five edge deletion and thresholded mask sparsity reproducible.
- Added bottom-edge and seeded-random deletion controls on validation only.
- Only the FP top-edge deletion flipped class; effects were mixed across cases,
  so no general explanation-faithfulness claim is made.
- Finalized evidence-bounded research questions, literature review,
  methodology, results, discussion, limitations, and conclusion.

## Integration of Main C1 and C2 Results — 2026-10-21
- Preserved original archive `ETGNN_Main_C1_C2_Results.zip` (SHA256: 550092c5f77bcb51702986006c48c8fec29ee98ff564f833ad17f780a43872fc).
- Extracted untouched outputs to `results/main_c1_c2_results/` and generated SHA256 manifest.
- Resolved "Unknown" metadata in `category` column by joining `node_index` to frozen configuration `configs/main_c1_c2_run_config.json`.
- Saved metadata-corrected files in `results/main_c1_c2_results/derived/` with separate `etgnn_category` and `matched_gat_category` columns.
- Verified 1-to-1 node mapping: 377/377 unique execution nodes mapped, 0 missing. No probabilities, drops, or pair records modified.
- Verified integration: 377 unique completed transactions, 240 C1 transactions, 420 C2 pair records across 8 registers.
- Recomputed summary statistics directly from raw/detailed intervention records with 1e-10 precision against original CSVs.
- Generated 4 publication-grade figures (200 DPI PNG + vector PDF) and 4 publication tables in `results/figures/` and `results/tables/`:
  1. C1 adjusted and raw deletion response distributions (`c1_response_distributions`).
  2. C2 paired differences and 95% bootstrap CIs across all 8 registers (`c2_paired_differences_ci`).
  3. Primary vs Degree-Sensitivity comparative analysis (`c2_primary_vs_degree_sensitivity`).
  4. Candidate-edge distribution and deletion fraction inflation (`candidate_edges_deletion_budget`).
- Updated `reports/results_discussion_draft.md` and `reports/explainability_findings.md` explaining C1 mean diff (~0.0029), ETGNN TP-FP persistence in degree sensitivity, GAT sensitivity to matching specification, FN imprecision, 50% vs 25% deletion budget discreteness, and complete epistemic limitations.

