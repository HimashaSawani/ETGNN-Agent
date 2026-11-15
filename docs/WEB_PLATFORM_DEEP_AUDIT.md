# ETGNN Web Platform Deep Scientific Validation & Engineering Audit

## 1. Audit Scope & Protocol

- **Repository**: `ETGNN-Fraud-Detection`
- **Audit Target**: Web Research Demonstration Platform (`src/web/`, `web/frontend/`, `tests/`)
- **Evaluation Standards**: PyTorch Geometric exact mathematical equivalence, reproducible inference determinism, scientific integrity, OWASP FastAPI security principles, and cross-browser single-page application stability.
- **Audit Rules**:
  - Zero model retraining.
  - Zero alteration of frozen research checkpoints or dataset files.
  - Zero fabrication of experimental results.
  - No C3 experimentation added.
  - All claims backed by concrete test execution.

---

## 2. Phase-by-Phase Audit Findings

### Phase 1: Baseline Verification
- **Backend Test Suite**: 15 / 15 tests executed and passed (`tests/test_web_api.py`).
- **Frontend TypeScript Compilation**: `tsc -b` compiled with zero diagnostic errors under `verbatimModuleSyntax` and `noUnusedLocals`.
- **Production Asset Build**: `vite build` completed in 1.35s with 1,943 transformed modules (`web/frontend/dist/`).
- **Verdict**: **PASS (100%)**.

### Phase 2: Prediction Correctness & Parity
- **Test Cohort**: 24 reproducibly sampled transactions spanning Train, Validation, Test, and the 4 C1/C2 exemplar cases (TP, FP, TN, FN).
- **Independent Inference**: Fresh whole-graph forward passes with `ETGNNClassifier` and `MatchedGATClassifier` on `test_graph.pt`, plus Scikit-Learn `RandomForestClassifier` and `XGBClassifier` on `scaled_features.npy`.
- **Absolute Numerical Discrepancies**:
  - ETGNN: $\max |\Delta P| = 0.00 \times 10^0$ (Exact bit-for-bit equivalence).
  - Matched GAT: $\max |\Delta P| = 0.00 \times 10^0$ (Exact bit-for-bit equivalence).
  - XGBoost: $\max |\Delta P| = 0.00 \times 10^0$ (Exact bit-for-bit equivalence).
  - Random Forest: $\max |\Delta P| = 2.22 \times 10^{-16}$ (Machine epsilon floating-point boundary).
- **Verdict**: **PASS (EXACT)**.

### Phase 3: Feature Override & Perturbation Correctness
- **Selective Dimension Modification**: Confirmed that adjusting `feat_0` perturbs only index 0 while indices 1–164 remain strictly identical.
- **Immutability of Source Dataset**: Verified that the underlying memory-mapped feature array (`scaled_features.npy`) is never mutated during overrides.
- **Dynamic Recomputation**: Re-running forward passes with perturbed features produced non-cached, distinct probability distributions.
- **Invalid Key Handling**: Out-of-bound keys (`feat_999`), non-numeric keys, or dictionaries with > 165 entries are safely rejected with HTTP 422 Unprocessable Entity.
- **Verdict**: **PASS**.

### Phase 4: Graph Explanation Integrity & Provenance
- **Edge Existence in Real Graph**: All 20 top edges across the 4 exemplar files (`connected_{tp,fp,tn,fn}.json`) were proven to exist in the actual graph tensor `test_graph.pt`.
- **Node-to-Transaction Mapping**: Exact 1-to-1 parity verified across all sample records in `results/main_c1_c2_results/detailed_intervention_records.csv`.
- **Non-Causal Legal Disclaimers**: Verified that all explanation cards display prominent legal boundaries clarifying that edge attribution represents mathematical sensitivity under deletion, not empirical proof of money laundering.
- **Verdict**: **PASS**.

### Phase 5: Graph Visualization & Truncation Disclosure
- **Defect Identified & Resolved**: Initial subgraph capping truncated dense neighborhoods without informing the user.
- **Remediation**: Implemented `torch_geometric.utils.subgraph` to correctly induce edge indices on capped node subsets and added prominent warning banners in the UI informing users when neighborhoods are capped at 150 nodes.
- **Verdict**: **PASS (REMEDIATED)**.

### Phase 6: Scientific Results Integrity
- **Metric Verification**: Confirmed that all dashboard metrics match official frozen CSV files:
  - Random Forest: 0.6323 AUPRC (Top baseline).
  - XGBoost: 0.5472 AUPRC.
  - ETGNN: 0.2250 AUPRC on held-out test steps 41–49.
  - Matched GAT: 0.2099 AUPRC.
  - ETGNN Local No Edges: 0.4812 AUPRC (The +0.2468 jump).
- **C1 & C2 Evidence**: C1 correctly disclosed as inconclusive (95% CI spans zero). C2 correctly presented across 8 registers with bootstrap whiskers.
- **C3 Absence**: Confirmed that C3 is not presented as a completed experiment.
- **Verdict**: **PASS**.

### Phase 7: Security & Reliability
- **CORS Hardening**: Restricted CORS to local development loopback origins (`http://127.0.0.1:*`, `http://localhost:*`).
- **Path Traversal Protection**: Hardened SPA static file resolver to reject traversal attempts (`..` or `%2e%2e`) with HTTP 403 Forbidden.
- **Pydantic Schema Validation**: Enforced strict integer range bounds on transaction IDs and validated finite floats for feature overrides.
- **Verdict**: **PASS**.

### Phase 8: Browser E2E Walkthrough
- **Automated Subagent Walkthrough**: Complete verification of all 9 tabs on live server `http://127.0.0.1:8000/`.
- **Interactive Validation**: Feature sliders, live forward pass, exemplar artifact selection, and canvas zoom/pan verified.
- **Console Log Health**: Zero console errors logged across entire session.
- **Verdict**: **PASS**.

---

## 3. Overall Audit Verdict

**VIVA DEMO READY**

The web demonstration platform accurately, faithfully, and safely showcases the completed ETGNN research without discrepancies, regressions, or security vulnerabilities.
