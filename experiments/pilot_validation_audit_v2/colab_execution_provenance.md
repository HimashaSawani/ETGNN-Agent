# Colab Execution & Audit Provenance Documentation

## 1. Executive Statement of Provenance
To maintain complete scientific integrity, this document explicitly distinguishes between:
1. **The original 700 perturbation trials**: Recorded on 2026-10-19 20:55:51 UTC (documented in `config.json`, SHA256: `64c4b252aa13c8608408ba9496d5648d4ee6eafa1fd271263e84babd0f08b69e`).
2. **The hosted Google Colab session**: Executed on 2026-10-20.

---

## 2. What Occurred in Hosted Google Colab
The notebook `ETGNN_Validation_Audit_Colab.ipynb` was uploaded and run in Google Colab with `FORCE_RERUN = False` (the default setting to avoid redundant heavy compute):

1. **Extraction & Environment Verification (Cells 1–4)**:
   - Mounted Google Drive, extracted `ETGNN_Colab_Bundle.zip` into `/content/etgnn_workspace`.
   - Inspected Colab hardware: Linux environment, 2 CPU threads allocated, RAM ~12.7 GB.
   - Installed `torch_geometric` without replacing native PyTorch.
2. **Cryptographic Checkpoint & Graph Integrity Checks (Cell 5)**:
   - Validated streaming SHA256 of `etgnn_local.pt` (`2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082`).
   - Validated streaming SHA256 of `validation_graph.pt` (`261b7ee5951d386bee9843adfc2471726ec9f433341d7c4359c7f66e44cbfc71`).
   - Verified that `test_graph.pt` was absent, enforcing leak-free isolation.
3. **Live Model-Inference Equivalence Audit (Cell 6)**:
   - Loaded `ETGNNClassifier` checkpoint on Colab CPU.
   - Evaluated 2-hop computation subgraphs across the 20 pilot nodes.
   - Proved numerical equivalence between full validation graph inference and localized 2-hop subgraph inference (maximum observed discrepancy = $1.49 \times 10^{-8}$).
4. **Post-Processing & Reporting Corrections (Cell 7)**:
   - Recomputed hierarchical per-node macro averages alongside pooled trial-level means directly from `raw_interventions.csv`.
   - Generated `node_summary_corrected.csv`, `aggregate_summary_corrected.csv`, and `quadrant_summary_corrected.csv`.
5. **Regeneration of Publication Figures & Packaging (Cells 9–11)**:
   - Rendered updated high-DPI bar plots (`pilot_paired_advantage_bars.png`, `pilot_paired_advantage_bars.pdf`).
   - Synced results to versioned directory and packaged outputs into `ETGNN_Audit_Corrected_Results.zip`.

---

## 3. What Did NOT Run from Scratch in Colab
- The 60 GNNExplainer optimization runs (50 epochs each), 600 random draws, and exhaustive single-edge occlusion sweeps were **not** re-trained or re-optimized from scratch in Colab during this session.
- The notebook detected that the saved 700 perturbation trials passed mathematical consistency and model equivalence checks, reusing them rather than repeating 105+ seconds of CPU computation.
- If an empirical rerun is ever required, toggling `FORCE_RERUN = True` in Cell 8 executes the live runner on Colab CPU.

---

## 4. Test Data Access Status
- **No new test inference or retraining was performed.**
- Test periods (time steps 41–49) were previously evaluated during initial baseline research and are completely frozen.
- The numerical verification script (`verify_results_and_figures.py`) reads frozen test probability arrays and test labels solely for read-only mathematical consistency checks against reported metric tables. Test graph tensors were excluded from the Colab execution bundle.

---

## 5. Cryptographic and Runtime Artifact Hashes

| Artifact | Location | File Size | SHA256 Hash |
| :--- | :--- | :--- | :--- |
| **Model Checkpoint** (`etgnn_local.pt`) | `results/checkpoints/` | 32,901 bytes | `2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082` |
| **Validation Graph** (`validation_graph.pt`) | `data/processed/` | 114,002,507 bytes | `261b7ee5951d386bee9843adfc2471726ec9f433341d7c4359c7f66e44cbfc71` |
| **Pilot Config** (`config.json`) | `experiments/pilot_validation_audit_v2/` | 935 bytes | `64c4b252aa13c8608408ba9496d5648d4ee6eafa1fd271263e84babd0f08b69e` |
| **Colab Execution Bundle** | Root | 45,833,268 bytes | Verified intact |
| **Colab Results ZIP** | Root | 732,663 bytes | Verified intact |
