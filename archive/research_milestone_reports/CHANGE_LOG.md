# Change Log: Final ETGNN C1 & C2 Integration & Corrections

Date: 2026-10-21
Repository: ETGNN-Fraud-Detection
Status: Audited, Corrected, and Invariance-Verified

---

## Summary of Corrections

1. **C2 Contrast Direction Alignment:**
   - **Formula Verified:** $\Delta_{pair} = A(\text{incorrect}) - A(\text{correct}) = A(\text{FP}) - A(\text{TP})$.
   - **Empirical Meaning:** Mean paired difference for ETGNN Primary TP-FP is $+0.039343$ (95% CI $[+0.020234, +0.057410]$). Positive sign indicates:
     *"Within the matched sample, the explanation-minus-random deletion response was larger for false positives than true positives."*
   - Earlier text inadvertently inverted this phrasing. All reports, captions, and table interpretations have been updated to reflect the exact calculation direction without asserting claims of error correction or explanation faithfulness.

2. **Ceiling Budget Formula Clarification:**
   - **Frozen Protocol:** $k = \max(1, \lceil 0.25 \times E_{cand} \rceil)$.
   - **Verification:** All 377 transactions ($100\%$) and 39,585 detailed rows in saved records strictly match the ceiling rule (e.g., $E_{cand}=5, 6, 7 \implies k = 2$). Zero mismatches.
   - Summaries and reproduction documentation were corrected from `floor` to `ceil`.
   - Clarified that the actual deletion fraction is $k / E_{cand}$ (e.g., $1/2 = 50\%$ for $84.1\%$ of nodes) and defined $E_{cand}$ as "eligible edges in the defined computation neighbourhood" rather than incident edges.

3. **Subgraph Inference Equivalence Separation:**
   - Clearly separated:
     - **Main-Run Assertion Coverage:** All 377 completed nodes verified at $\le 10^{-5}$ tolerance on forward pass probabilities during benchmark execution.
     - **Independent Stratified Verification Sample:** $N=26$ nodes across $E_{cand} \in \{2..7\}$ observed maximum discrepancy of $3.58 \times 10^{-7}$ for ETGNN and $8.94 \times 10^{-8}$ for Matched GAT.
   - Clarified that the sample maximum is descriptive and not claimed as an upper bound for all 377 nodes. Both checks evaluated on probabilities (not logits).

4. **Restrained C1 Interpretation:**
   - Replaced claims of model equivalence and noise removal with:
     *"The observed mean random-adjusted response difference was small in this selected sample of frozen checkpoints"* ($\Delta_{adj} = 0.002903$).
   - Explicitly identified $\pm 0.096180$ as the sample standard deviation across C1 transactions, not a confidence interval or standard error.
   - Clarified that random deletion serves as an empirical comparator rather than an automatic pure noise floor.

5. **Sensitivity Matching and Graph Dependence Caveats:**
   - Documented that Primary and Degree-Sensitivity registers comprise different, overlapping matched samples ($N=95$ vs $N=57$ for ETGNN TP-FP; $N=94$ vs $N=82$ for GAT TP-FP).
   - Changes cannot be attributed solely to degree adjustment.
   - Clarified that matched-pair bootstrap resamples pairs as clusters, which accounts for within-pair correlation but does not resolve spatial autocorrelation between different pairs sharing graph context.
   - All 8 confidence intervals remain explicitly labelled as marginal and unadjusted.

6. **Historical Data-Access and Novelty Boundaries:**
   - Explicitly disclosed partition timeline: development on steps 33–36, main follow-up on steps 37–40 within previously used validation data, and historical test steps 41–49 previously inspected.
   - Avoided describing test split access as "one-time" or "untouched".
   - Maintained literature novelty claims as provisional.

7. **Maintained Reproduction Scripts in `src/evaluation/`:**
   - Moved all correction, verification, publication generation, and demo test scripts out of `scratch/` into `src/evaluation/`.
   - Created automated end-to-end HTTP verification test (`src/evaluation/test_demo_http_server.py`) asserting real API startup, `/health`, `/examples`, `/transactions/{id}`, and 404 responses.

8. **Follow-Up Audit Pass Defect Corrections (2026-10-21):**
   - **Protocol Parameters Derived from Config:** Corrected `reproduction_documentation.md` to derive frozen GNNExplainer parameters directly from `configs/main_c1_c2_run_config.json` (seeds `[42, 43, 44]`, 50 epochs, seed formula `20261020 + node_index`), replacing incorrect older values `[42, 123, 999]` and 200 epochs.
   - **Methodology Matching & Threshold Corrections:** Updated `methodology_draft.md` to specify exact out-degree matching for degree-sensitivity registers (`treat_out_degree == ctrl_out_degree` for 100% of pairs) with margin and candidate-edge calipers, removed the unsupported development "threshold tuning" claim, stated the frozen decision threshold $\tau = 0.50$, and distinguished raw dataset classes (Class 1 illicit, Class 2 licit) from model target labels ($y=1, y=0$).
   - **Full C1 Histogram Bin Range:** Expanded histogram bins in `generate_c1_c2_publication_artifacts.py` from `[-0.35, 0.45]` to `[-0.60, 0.65]` (51 bins), covering all 240 transactions without truncation; added automated assertion verifying `counts.sum() == 240`.
   - **Figure Layout & Clean Legend Placements:** Set `sharey=False` in Figure 3 with independent categorical model-only labels ("ETGNN", "Matched GAT") to eliminate cross-panel label conflicts; moved C2 forest plot (Figure 2) legend above plot to completely clear all data points and text annotations.
   - **Strengthened Integration Verification:** Upgraded `verify_full_integration_evidence.py` to assert duplicate key uniqueness, exact configured ID and register membership, model-specific target classes, shared C1 deleted-edge sets across all seeds, full reconstruction of all 420 C2 pair differences from detailed records with explicit tolerance $< 10^{-10}$ (observed max $1.77 \times 10^{-16}$), and accounted for structurally absent runtime fields.
   - **Reproduction Evidence & Status:** Replaced "Fully Reproducible & Verified" with evidence-bounded status "Evidence-Bounded Invariance-Verified (Clean-environment reproduction test pending)"; removed guessed Python 3.10/3.11 claim (marked historical Colab Python/CUDA versions as "Not recorded"); documented original executed runner snapshot hash (`6081f2ba...`) vs. metadata-fixed runner hash (`ef67f742...`).
   - **Literature Review Expansion:** Updated `literature_review.md` around both core empirical questions (C1 shared interventions, C2 margin/degree-matched error contrasts), citing primary benchmarking literature (Hooker et al., 2019; Luo et al., 2020; Amara et al., 2022) and keeping novelty provisional.

9. **Final Thesis Pre-Assembly Corrections (2026-10-21):**
   - **Margin Matching Definition & Calipers:** Explicitly defined within-node prediction margin as $\text{margin}(v) = |p_{\text{illicit}}(v) - 0.50|$ and the pair eligibility condition as $|\text{margin}(\text{incorrect}) - \text{margin}(\text{correct})| \le 0.050$ across `methodology_draft.md` and `literature_review.md` (strictly distinguishing this pair condition from $|p_{\text{illicit}} - 0.50| \le 0.050$). Retained the source-confirmed protocol candidate-edge caliper $\le 5$ and distinguished this configured limit from smaller differences observed in selected pairs (e.g., $\le 2$ in ETGNN TP–FP).
   - **Official GraphFramEx BibTeX Import:** Imported the official publication record directly from PMLR (`https://proceedings.mlr.press/v198/amara22a.html`): Kenza Amara, Zhitao Ying, Zitao Zhang, Zhichao Han, Yang Zhao, Yinan Shan, Ulrik Brandes, Sebastian Schemm, and Ce Zhang (2022), *Proceedings of the First Learning on Graphs Conference*, PMLR volume 198, pages 44:1–44:23. Replaced all inaccurate author lists and page numbers from previous manual reconstructions.
   - **Official PGExplainer Reference:** Corrected official NeurIPS 2020 citation URL to `https://proceedings.nips.cc/paper_files/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html`.
   - **Literature Interpretation Restrained:** Replaced "did not induce a substantial differential response" with: *"The observed mean adjusted difference was small; equivalence was not established."*
   - **Reproduction Evidence & Manifest:** Added `requirements.txt`, standalone executed runner snapshot (`run_main_c1_c2_benchmark_executed_snapshot.py`, SHA-256 `6081f2ba...`), and full automated test suite logs (`results/verification_outputs.log`) to the review archive. Explicitly documented external supporting bundle locations (`ETGNN_Main_C1_C2_Execution_Bundle.zip`, $41.6\text{ MB}$, and `ETGNN_Main_C1_C2_Results.zip`, $0.49\text{ MB}$).
   - **Numerical Results & Protocol Preserved:** Maintained 100% invariance across all completed nodes, raw records, detailed interventions, 420 C2 pairs, and 8 bootstrap registers.

---

## Numerical Invariance Confirmation
- Total completed nodes: 377 (100% match)
- Total C1 nodes: 240 (100% match)
- Total C2 pairs: 420 (100% match across 8 registers)
- Numerical values modified: **ZERO (0)**
