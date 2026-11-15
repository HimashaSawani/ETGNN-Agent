# Reproduction Documentation: ETGNN C1 & C2 Benchmark

**Date:** 2026-10-21  
**Project:** Explainable Time-Encoded GAT (ETGNN) for Elliptic Illicit Transaction Detection  
**Authoritative Protocol Status:** Evidence-Bounded Invariance-Verified (Clean-environment reproduction test pending)  

---

## 1. Environment Specification

### A. Historical Benchmark Execution Environment (Google Colab)
The main benchmark experiment ($N = 377$ nodes, 39,585 intervention passes) was executed on hosted Google Colab to support isolated batch execution and Drive checkpointing:
- **Platform:** Linux (hosted VM container, Colab environment)
- **Kernel & Python Version:** Not recorded (Colab dynamic container allocation)
- **Acceleration Device:** GPU / CUDA Runtime (exact hardware GPU model and CUDA patch version not recorded; PyTorch synchronized execution)
- **Execution Timestamp:** Epoch timestamp `1792517776.818479` (recorded in `main_resume_state.json`)
- **Execution Mode:** `MAIN_RUN`

### B. Local Verification & Artifact Generation Environment
All verification checks, metadata joins, full-graph equivalence assertions, and publication artifact generation were performed locally in the project repository:
- **Operating System:** Windows 11 Pro (10.0.26300, 64-bit AMD64)
- **Processor:** Intel64 Family 6 Model 140 Stepping 2, GenuineIntel
- **System Memory:** 7.65 GB RAM
- **Python Version:** 3.13.3 (`tags/v3.13.3:6280bb5`, Apr 8 2025, MSC v.1943 64-bit)
- **PyTorch:** `2.14.0+cpu`
- **PyTorch Geometric (PyG):** `2.8.0.post1`
- **NumPy:** `2.5.3`
- **CUDA Available:** False (CPU inference)

---

## 2. Dependency Installation

To set up the reproduction environment from scratch:

```bash
# 1. Initialize virtual environment with Python 3.13
python -m venv .venv

# 2. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install pinned dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 3. Inputs, Checkpoints, and Hash Manifest

All inputs and model weights are verified against `manifest.sha256` and `results/main_c1_c2_results/manifest.sha256`:

| Artifact | Local Relative Path | SHA-256 Checksum |
|---|---|---|
| **Run Config** | `configs/main_c1_c2_run_config.json` | `32ec1fff99a50817a1355bc765ecb691c081022cb28396ff93d987d7f21f827b` |
| **Executed Runner Snapshot** | `src/evaluation/run_main_c1_c2_benchmark_executed_snapshot.py` | `6081f2baa80ad91e1fad88181f6da43c5c475a3008779c7308eeb9e9fe75682a` |
| **Metadata-Fixed Runner** | `src/evaluation/run_main_c1_c2_benchmark.py` | `ef67f7427a290eec54bfb990fba3e7c6b384a8fe2e4983af384912a17f938006` |
| **Environment Dependencies** | `requirements.txt` | `17d15f9d3b7c6c46b7912d9a3152ce0d866309b44acb9482206171071cb4a363` |
| **Verification Outputs Log** | `results/verification_outputs.log` | `0b94032335d99053ba111f5b7d340b1c0aef1b2e0a062b5a4bda84fb99de9dfa` |
| **Validation Graph** | `data/processed/validation_graph.pt` | `261b7ee5951d386bee9843adfc2471726ec9f433341d7c4359c7f66e44cbfc71` |
| **Node Mapping** | `data/processed/node_mapping.csv` | `90ae469bb1449e51f0741faa9836075ec8577451124d180fb65dc17a3d68c175` |
| **Historical Exclusions** | `experiments/pilot_validation_audit/pilot_nodes.csv` | `ed733950dfcfe21b61341357363e5041ca86c74487a3df60bc062d8d0a91433a` |
| **ETGNN Checkpoint** | `results/checkpoints/etgnn.pt` | `b2f5535f3b6b1675109dae205934e761d2ad1b80b5409564bbe5889edb79dadc` |
| **Matched GAT Checkpoint**| `results/checkpoints/matched_gat.pt` | `1e9b43ea1cc25a26dd8bbe953c49a8782a0d9c27b5e0da17afe726fd21405074` |
| **Original Benchmark ZIP**| `ETGNN_Main_C1_C2_Results.zip` | `550092c5f77bcb51702986006c48c8fec29ee98ff564f833ad17f780a43872fc` |

*Provenance Note on Runner Modification:* The original executed runner (`6081f2ba...`, preserved in `src/evaluation/run_main_c1_c2_benchmark_executed_snapshot.py`) attempted to read `item.get("etgnn_prediction_category", "Unknown")`, which resulted in `"Unknown"` being written to the output category column because the frozen configuration key was `etgnn_category`. The metadata-fixed runner (`ef67f742...`) changes this single line to `item.get("etgnn_category", item.get("etgnn_prediction_category", "Unknown"))`. All numerical code, loops, seeds, and deletion logic are 100% identical.

### External Supporting Bundles & Locations
For full end-to-end retraining, full graph reconstruction, or external verification, the following supplementary bundles are maintained in the repository root:
1. **Full Execution Bundle (`ETGNN_Main_C1_C2_Execution_Bundle.zip`):** Size $41.6\text{ MB}$, SHA-256 `1a48608440287556bd5bf867902f07956def2c5a037ba075607a0d98ff016d3d`. Contains complete standalone codebase, model checkpoints, node mappings, and processed PyTorch Geometric validation graph (`data/processed/validation_graph.pt`).
2. **Original Output Archive (`ETGNN_Main_C1_C2_Results.zip`):** Size $0.49\text{ MB}$, SHA-256 `550092c5f77bcb51702986006c48c8fec29ee98ff564f833ad17f780a43872fc`. Contains pristine historical Colab run outputs (summary, raw, detailed records, resume state).

---

## 4. Random Seeds and Protocol Parameters (Frozen in Config)

Directly derived from `configs/main_c1_c2_run_config.json`:
- **GNNExplainer Optimization Epochs:** 50 epochs (`protocol_parameters.gnnexplainer_epochs`).
- **Explainer Optimization Seeds ($S = 3$):** `[42, 43, 44]` (`protocol_parameters.explainer_seeds`).
- **Random Baseline Comparator Draws ($D = 30$):** 30 independent draws per node per model under identical budget $k$ (`protocol_parameters.random_comparator_draws`).
- **Random Seed Formula:** `20261020 + node_index` (`protocol_parameters.random_seed_formula`).
- **Budget Rule:** Nominal $\alpha = 0.25$, with discrete budget $k = \max(1, \lceil \alpha \cdot E_{cand} \rceil)$ (`protocol_parameters.budget_formula`), where $E_{cand}$ denotes eligible edges in the defined 2-hop computation neighbourhood. The actual deletion fraction is $k / E_{cand}$ (e.g., $1/2 = 50.0\%$ for $E_{cand}=2$, representing $84.1\%$ of nodes).
- **C2 Cluster Bootstrap Resampling:**
  - Resampling unit: Matched pairs (clusters). Preserving pairs accounts for within-pair correlation, but does not resolve spatial dependence between pairs sharing graph context.
  - Iterations: $B = 2,000$.
  - Fixed Seed: `20261020` (`numpy.random.default_rng(20261020)`).
  - Re-seeded for each register independently in benchmark code.
  - All eight resulting confidence intervals are marginal and unadjusted.

---

## 5. Exact Execution & Reproduction Commands

The full reproduction workflow is maintained in `src/evaluation/` and executes via the following ordered sequence:

### Step 1: Subgraph Inference Equivalence Verification
Separates main-run assertion coverage from additional stratified sample verification:
- **Main Run:** Runtime assertions verified full-graph vs. 2-hop computation subgraph forward pass probabilities within $10^{-5}$ tolerance across all 377 completed nodes.
- **Stratified Sample:** An independent stratified sample ($N=26$ across $E_{cand} \in \{2..7\}$) evaluates numerical equivalence on probabilities:
```bash
.\.venv\Scripts\python.exe src/evaluation/verify_subgraph_equivalence.py
```
*Empirical Result:* Maximum observed discrepancy in the sample was $3.58 \times 10^{-7}$ for ETGNN and $8.94 \times 10^{-8}$ for Matched GAT. Zero tolerance violations. Note: The sample maximum is descriptive and is not claimed as an upper bound for all 377 nodes.

### Step 2: Comprehensive Integration Verification
Verifies 377 unique completed transactions, 240 C1 transactions, 420 C2 pairs across 8 registers, frozen ceiling budget rule for all nodes, and recomputes all statistics from raw records:
```bash
.\.venv\Scripts\python.exe src/evaluation/verify_full_integration_evidence.py
```
*Empirical Result:* All 12 automated verification passes succeed with floating-point difference $< 10^{-10}$.

### Step 3: Metadata Correction & Derived File Generation
Joins `node_index` with `execution_node_union["nodes"]` to populate `etgnn_category` and `matched_gat_category` without altering probabilities, deletion responses, or statistical results:
```bash
.\.venv\Scripts\python.exe src/evaluation/create_derived_metadata.py
```

### Step 4: Publication Artifact Generation
Generates high-resolution figures (200 DPI PNG and vector PDF) and CSV tables with corrected contrast direction and budget ceiling formula:
```bash
.\.venv\Scripts\python.exe src/evaluation/generate_c1_c2_publication_artifacts.py
```

### Step 5: Research Demo Verification (Real HTTP Server)
Spawns the demo API server locally via Uvicorn and tests real HTTP network requests (`/health`, `/examples`, `/transactions/{id}`, and 404 handler):
```bash
.\.venv\Scripts\python.exe src/evaluation/test_demo_http_server.py
```

---

## 6. Output Paths & Directory Structure

```
results/
├── checkpoints/
│   ├── etgnn.pt
│   └── matched_gat.pt
├── figures/
│   ├── c1_response_distributions.png / .pdf
│   ├── c2_paired_differences_ci.png / .pdf
│   ├── c2_primary_vs_degree_sensitivity.png / .pdf
│   └── candidate_edges_deletion_budget.png / .pdf
├── tables/
│   ├── c1_response_summary.csv
│   ├── c2_registers_bootstrap_summary.csv
│   ├── c2_sensitivity_comparison.csv
│   └── candidate_edge_budget_summary.csv
└── main_c1_c2_results/
    ├── manifest.sha256
    ├── c1_statistical_results.csv              (original, preserved)
    ├── raw_benchmark_records.csv               (original, preserved)
    ├── main_benchmark_summary.csv              (original, preserved)
    ├── detailed_intervention_records.csv       (original, preserved)
    ├── c2_individual_pair_differences.csv      (original, preserved)
    ├── c2_bootstrap_results.csv                (original, preserved)
    ├── main_resume_state.json                  (original, preserved)
    └── derived/
        ├── c1_statistical_results_derived.csv
        ├── raw_benchmark_records_derived.csv
        ├── main_benchmark_summary_derived.csv
        └── detailed_intervention_records_derived.csv
```
