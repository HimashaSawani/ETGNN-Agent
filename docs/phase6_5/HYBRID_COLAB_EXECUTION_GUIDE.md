# ETGNN-Agent — Phase 6.5 Hybrid Colab Pro Execution Guide

## Overview

This guide provides step-by-step instructions for executing the resource-intensive machine learning benchmarks, multi-model evaluation, synchronized GPU latency timing, and temporal concept drift calculations of ETGNN-Agent on **Google Colab Pro**, while keeping lightweight application components (FastAPI backend, React 19 frontend, and Playwright integration) on your **local Windows PC**.

---

## 1. Why a Hybrid Architecture?

### Resource Comparison

| Resource / Metric | Local Windows PC | Google Colab Pro | Optimization Benefit |
| :--- | :--- | :--- | :--- |
| **System RAM** | 7.65 GB total (~0.56 GB free) | 25.0 GB to 53.0 GB (High-RAM) | Eliminates Python `MemoryError` & swap thrashing |
| **Primary Storage** | C: drive (6.56 GB free), D: drive (9.60 GB free) | 150 GB+ High-Speed NVMe | Protects local C: drive from filling up |
| **Compute Device** | Intel Core CPU (slow multi-thread) | NVIDIA T4 / A100 Tensor Core GPU | Accelerates full-graph inference from minutes to seconds |
| **PyTorch Geometric** | CPU JIT compilation bottleneck | Pre-accelerated CUDA PyG | Instant sparse graph convolutions |
| **Primary Role** | UI, API, Browser E2E, Local Demo | Heavy ML Inference, Drift Census, Macro Evals | Optimal workload separation |

---

## 2. Preparing Files for Google Drive (Zero Local C: Drive Impact)

Upload the lean project bundle directly from `D:\ETGNN-Fraud-Detection` to Google Drive:

1. Open your web browser and navigate to **[Google Drive](https://drive.google.com)**.
2. In your Google Drive root (**My Drive**), create or use the dedicated folder named:
   ```text
   ETGNN
   ```
   *(Note: The notebook also supports `ETGNN-Fraud-Detection` via automated path detection).*
3. Upload the following folders and files from `D:\ETGNN-Fraud-Detection`:
   - `artifacts/` (`baseline_manifest.json`)
   - `data/processed/` (`test_graph.pt`, `nodes_with_masks.csv`)
   - `results/checkpoints/` (`etgnn.pt`, `matched_gat.pt`, `gat.pt`, `gcn.pt`, `graphsage.pt`, `random_forest.joblib`, `xgboost.json`)
   - `src/models/` (`etgnn.py`, `matched_gat.py`, `gat.py`, `gcn.py`, `graphsage.py`, `time_encoding.py`)
   - `notebooks/phase6_5/colab_verification.ipynb`

> [!TIP]
> You only need to upload **156.41 MB** of files. Do **not** upload `.venv`, `node_modules`, or `.git`. This ensures your local `C:` drive remains untouched.

---

## 3. Opening the Notebook in Google Colab Pro

1. Go to **[Google Colaboratory](https://colab.research.google.com)**.
2. Click **File** -> **Open notebook**.
3. Select the **Google Drive** tab and open:
   ```text
   ETGNN/notebooks/phase6_5/colab_verification.ipynb
   ```
   *(Alternatively, select the **Upload** tab and upload `D:\ETGNN-Fraud-Detection\notebooks\phase6_5\colab_verification.ipynb` directly).*

---

## 4. Selecting Hardware Accelerator in Colab Pro

1. In the top-right menu of Colab, click the **RAM/Disk** status dropdown, or click **Runtime** -> **Change runtime type**.
2. Select:
   - **Hardware accelerator**: `T4 GPU` (or `A100 GPU` if compute units permit).
   - **Runtime shape**: `High-RAM`.
3. Click **Save** and then click **Connect** in the top right.

---

## 5. Step-by-Step Notebook Execution & Verification Gates

### Cell 1: Hardware & Compute Environment Inspection
- Run this cell to verify GPU assignment (`NVIDIA T4` or `A100`) and high-RAM allocation (>25 GB RAM).
- Automatically detects CUDA capabilities and falls back to CPU if no GPU units are allocated.

### Cell 2: Google Drive Mounting & Path Configuration
- Authorize Google Drive access when prompted.
- Automatically selects `/content/drive/MyDrive/ETGNN` (or falls back to `ETGNN-Fraud-Detection`).

### Cell 3: Fast Workspace Setup
- Copies lightweight Python modules to the ephemeral NVMe disk at `/content/etgnn_workspace`.
- Configures output directories under `results/colab_benchmarks/`.

### Cell 4: Selective Dependency Installation
- Installs `torch_geometric`, `xgboost`, and `scikit-learn` in less than 30 seconds.

### Cell 5: Checkpoint & Dataset Cryptographic SHA-256 Verification Gate
- Automatically verifies all 11 model checkpoints against `artifacts/baseline_manifest.json`.
- Guarantees 100% bitwise integrity across Drive transfer before inference runs.

### Cell 6: Graph Alignment & Test Split Validation Gate
- Validates the structural alignment between `test_graph.pt` and `nodes_with_masks.csv`.
- Confirms test split strictly spans **Time Steps 41–49** with **9,973 labeled test transactions** (524 illicit, 9,449 licit).

### Cell 7: Multi-Model Benchmark & Synchronized GPU Latency
- Evaluates ETGNN, Matched GAT, GAT, GCN, GraphSAGE, Random Forest, and XGBoost on test steps 41–49.
- Performs **10 warm-up passes** and uses **`torch.cuda.synchronize()`** across 30 timed iterations.
- Eliminates inaccurate asynchronous timing and reports genuine per-pass and per-transaction latency.

### Cell 8: 49-Step Temporal Concept Drift, Top Feature KS-Test & Snapshot 43 Topology
- Computes transaction distribution drift across all 49 time steps.
- Runs Kolmogorov-Smirnov (KS) two-sample feature drift tests across train (steps 1–32) vs test (steps 41–49).
- Executes directed degree census on Snapshot 43 (in-degree, out-degree, zero-in-degree nodes).
- Enforces strict non-causal epistemic framing regarding darknet market disruptions.

### Cell 9: Fail-Closed Verification Certificate
- Implements strict gate logic:
  - Requires `checkpoint_hash_integrity == PASS`
  - Requires `graph_alignment_and_test_split == PASS`
  - Requires `etgnn_model_inference == PASS`
  - Itemizes baseline model statuses (`PASS`, `PARTIAL`, or `SKIPPED`)
- Issues `COLAB_VERIFIED_COMPLETE` or `COLAB_PARTIAL_VERIFIED` accordingly.
- Exports `colab_execution_certificate.json` to Google Drive.

---

## 6. Retrieving Verified Artifacts Back to Windows

Once the Colab run completes, verified results are stored in your Google Drive at:
```text
/content/drive/MyDrive/ETGNN/results/colab_benchmarks/
```

To sync them back to your local Windows repository:
1. Download the lightweight `colab_benchmarks/` folder from Google Drive (~50 KB JSON files).
2. Place them into:
   ```text
   D:\ETGNN-Fraud-Detection\results\colab_benchmarks\
   ```
3. Your local Windows PC remains fast, with 0 additional `C:` drive consumption, and fully ready for local portfolio demonstration.
