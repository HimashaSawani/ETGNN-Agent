# Complete Operations, Engineering, and Reproducibility Guide

**Repository:** `D:\ETGNN-Fraud-Detection`  
**Research Title:** *Explainable Temporal Graph Neural Networks for Adaptive Financial Fraud Detection*

---

## 1. Installation & Windows Environment Setup

### 1.1 System Requirements
- **Operating System**: Windows 10/11 (x64)
- **Python**: `3.13.3` (virtual environment located at `D:\ETGNN-Fraud-Detection\.venv`)
- **Node.js / npm**: Node.js `v20+` and `npm` (for building `web/frontend`)
- **Hardware**: CPU-only execution supported (`torch==2.11.0+cpu`, `torch_geometric==2.7.0`); `torch.set_num_threads(2)` enforced across harnesses to prevent memory contention on 8 GB RAM machines.

### 1.2 Environment Initialization
```powershell
cd D:\ETGNN-Fraud-Detection
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 2. Dataset Setup & Provenance Verification

### 2.1 Primary Dataset: Elliptic1 Bitcoin Transaction Graph
Raw files in `data/raw/elliptic/`:
- `elliptic_txs_features.csv` ($203,769$ rows, $167$ columns: `txId`, `time_step`, $93$ local features, $72$ 1-hop aggregate features)
- `elliptic_txs_classes.csv` (`1` = illicit [$4,545$], `2` = licit [$42,019$], `unknown` [$157,205$])
- `elliptic_txs_edgelist.csv` ($234,355$ directed payment edges)

To rebuild the processed expanding-window PyTorch Geometric stage graphs (`data/processed/train_graph.pt`, `validation_graph.pt`, `test_graph.pt`) from raw CSVs:
```powershell
.\.venv\Scripts\python.exe src\data\create_id_map.py
.\.venv\Scripts\python.exe src\data\map_edges.py
.\.venv\Scripts\python.exe src\data\map_labels.py
.\.venv\Scripts\python.exe src\data\create_masks.py
.\.venv\Scripts\python.exe src\data\scale_features.py
.\.venv\Scripts\python.exe src\data\build_stage_graphs.py
```

### 2.2 Secondary Cross-Representation Dataset: `Elliptic++ Actors`
When authorized to download `wallets_features.csv`, `wallets_classes.csv`, and `AddrAddr_edgelist.csv` from `https://github.com/git-disl/EllipticPlusPlus`, place them in `data/external/elliptic_plusplus/` and run the chunked, memory-safe preprocessing pipeline:
```powershell
.\.venv\Scripts\python.exe -c "from src.data.elliptic_plusplus_pipeline import EllipticPlusPlusPipeline; print(EllipticPlusPlusPipeline().process_and_save())"
```

---

## 3. Model Training & Evaluation Guide

### 3.1 Original Baselines & ETGNN (165 Features)
```powershell
.\.venv\Scripts\python.exe -m src.training.train_random_forest
.\.venv\Scripts\python.exe -m src.training.train_xgboost
.\.venv\Scripts\python.exe -m src.training.train_gcn
.\.venv\Scripts\python.exe -m src.training.train_graphsage
.\.venv\Scripts\python.exe -m src.training.train_gat
.\.venv\Scripts\python.exe -m src.training.train_etgnn
.\.venv\Scripts\python.exe -m src.training.train_matched_gat
```

### 3.2 Phase 4 Controlled 40-Trial Factorial Benchmark ($93\text{f}$ vs. $165\text{f}$)
All 40 trials (`Cond_A`, `Cond_B`, `Cond_C`, `Cond_D` $\times$ `{93f, 165f}` $\times$ seeds `{42..46}`) are completed and archived in `results/phase4_experiments/` and `research_extension/PHASE4_EXPERIMENT_MATRIX.csv`. To regenerate all Phase 4 summary tables and verify $0/40$ transcription discrepancies:
```powershell
.\.venv\Scripts\python.exe scratch\regenerate_phase4_reports.py
```

### 3.3 Phase 5 External Evaluation Harness (`src/training/run_phase5_external_eval.py`)
```powershell
# Stage 1 (Synthetic 4-condition dry run) + Stage 2 (Hardware preflight)
.\.venv\Scripts\python.exe src\training\run_phase5_external_eval.py --stage stage1_dryrun

# Stage 3 (Single-seed smoke pilot on real Elliptic++ Actors once downloaded)
.\.venv\Scripts\python.exe src\training\run_phase5_external_eval.py --stage stage3_pilot

# Stage 4 (Full 5-seed x 4-condition validation matrix = 20 runs)
.\.venv\Scripts\python.exe src\training\run_phase5_external_eval.py --stage stage4_validation

# Stage 5 (Single-shot test evaluation - requires explicit flag)
.\.venv\Scripts\python.exe src\training\run_phase5_external_eval.py --stage stage5_test --authorize-test
```

---

## 4. Explainability (XAI) Execution Guide

### 4.1 Phase H Validation on Frozen 93f `ResidualETGNNClassifier` ($N=30$ Multi-Edge Cohort)
```powershell
.\.venv\Scripts\python.exe scratch\run_phase_g_and_h_validation.py
```
Outputs:
- `results/phase5_xai/residual_etgnn_93f_xai_fidelity.csv`
- `results/phase5_xai/residual_etgnn_93f_xai_stability.csv`
- `results/phase5_xai/xai_validation_summary.json`
- `results/figures/phase5_xai_fidelity_and_stability.png`

---

## 5. Backend API & React Frontend Guide

### 5.1 REST API Endpoints (`src/web/app.py`)
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/health` | System health check, loaded models, and epistemic notice |
| `GET` | `/api/models` | Architectural metadata & SHA-256 hashes for all 9 model families |
| `GET` | `/api/metrics` | Held-out chronological test metrics (`N=9,973`, `5.25%` illicit) |
| `GET` | `/api/transactions` | Paginated node explorer with `split`, `label`, `time_step`, and `search` filters |
| `GET` | `/api/transactions/{tx_id}` | Full node degree, feature summary, and 4-model predictions |
| `GET` | `/api/transactions/{tx_id}/subgraph` | Local $k$-hop neighborhood (`hops=1..2`, truncation disclosure) |
| `POST` | `/api/predict` | Live inference across all loaded models with optional `features_override` |
| `GET` | `/api/explanations/{tx_id}` | Frozen C1/C2 exemplar explanation or live 2-hop `GNNExplainer` mask |
| `GET` | `/api/research/c1` | Frozen C1 temporal explanation-sensitivity summary |
| `GET` | `/api/research/c2` | Frozen C2 8-register matched-pair bootstrap summary |
| `GET` | `/api/research/ablations` | Multi-seed equivalence, $+0.2468$ edge-removal anomaly, and step volatility |
| `GET` | `/api/research/phase4` | Authoritative 40-trial Phase 4 factorial summary & Holm-adjusted contrasts |
| `GET` | `/api/research/xai_validation` | Phase H XAI validation summary (`Exact LOO Oracle` vs. `GNNExplainer`) |

### 5.2 Frontend Build & Verification
```powershell
npm --prefix web\frontend run build
.\.venv\Scripts\python.exe -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

---

## 6. Reproducibility & Checksum Verification

Run the automated test suite to verify SHA-256 checksums in `configs/phase5_frozen_manifest.json`, Phase 4 CSV-to-markdown parity, degree-preserving rewiring invariants, `EllipticPlusPlusPipeline` chunked processing, and all FastAPI endpoints:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 7. Troubleshooting Guide

1. **`ModuleNotFoundError: No module named 'torch_geometric'`**:
   Ensure you invoke the project virtual environment `.\.venv\Scripts\python.exe` rather than system Python.
2. **Matplotlib Cache Warning on Windows**:
   Set `$env:MPLCONFIGDIR = "D:\ETGNN-Fraud-Detection\scratch\.mpl_cache"` before running plotting scripts.
3. **Low Free RAM on 8 GB Windows Host**:
   Close browser tabs before loading `test_graph.pt` ($203,769$ nodes); `ResearchDataLoader` and `EllipticPlusPlusPipeline` automatically cap PyTorch threads at `2` and process CSVs in chunks.

---

## 8. Intended Use & Epistemic Safety Statement

This codebase and web platform are designed **exclusively for academic research, peer review, and thesis demonstration**. Predictions and post-hoc subgraph explanations reflect statistical correlations and gradient/occlusion sensitivity on historical anonymized blockchain snapshots; they must **never** be used for automated account freezing, regulatory compliance enforcement, or criminal attribution.

