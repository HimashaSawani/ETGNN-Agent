import os
import json
from pathlib import Path


def create_colab_notebook():
    EXPECTED_SIZE = 43534835
    EXPECTED_SHA256 = "b689a9f24ba8f95c0dc2d94cf3a0c5c64c784c0422c1ea3ad069c9b139db92e2"  # to be updated

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ETGNN + Matched GAT Development Technical Pilot Benchmark\n",
                    "\n",
                    "**Protocol:** Dated Frozen Protocol (v2.1)  \n",
                    "**Target Hardware:** Google Colab GPU (T4 / V100 / A100)  \n",
                    "**Objective:** Measure actual hardware execution latency (ms) per seed, CPU peak RSS, and GPU peak allocated VRAM for GNNExplainer (50 epochs, seeds 42, 43, 44), 30 random deletions, degree ranking, and single-edge occlusion on the **4 saved development validation nodes (steps 33–36)**.  \n",
                    "\n",
                    "> [!IMPORTANT]\n",
                    "> **Scope Disclosure:** This benchmark is executed strictly on 4 development-period nodes to profile hardware latency and memory overhead. **It is NOT a matched-pair effect-variance pilot**, and development-period outcomes will not be used to estimate effect sizes or stopping rules."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Inspect Hardware Environment\n",
                    "Verify that a GPU runtime is active."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!nvidia-smi\n",
                    "import torch\n",
                    "print('PyTorch Version:', torch.__version__)\n",
                    "print('CUDA Available:', torch.cuda.is_available())\n",
                    "if torch.cuda.is_available():\n",
                    "    print('GPU Device Name:', torch.cuda.get_device_name(0))\n",
                    "    print('VRAM Total (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))\n",
                    "else:\n",
                    "    print('WARNING: Running on CPU. Please switch to GPU in Runtime -> Change runtime type.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Load the Execution Bundle\n",
                    "\n",
                    "The bundle archive `ETGNN_Technical_Pilot_Bundle.zip` is **43.5 MB** (43,534,835 bytes).\n",
                    "\n",
                    "> [!WARNING]\n",
                    "> **Avoid Drag-and-Drop Truncation:** In Google Colab, dragging and dropping large files into the left sidebar file browser often truncates the upload at 10–12 MB due to browser WebSocket timeouts, producing a corrupted `BadZipFile`.\n",
                    ">\n",
                    "> **Use one of the two reliable methods below:**\n",
                    "> - **Option A (Recommended & Fastest):** Mount Google Drive (upload `ETGNN_Technical_Pilot_Bundle.zip` to your Google Drive `MyDrive/` first, which never truncates).\n",
                    "> - **Option B:** Use the Colab `files.upload()` widget below, which streams with an active progress bar."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Option A: Mount Google Drive (Recommended)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, shutil\n",
                    "\n",
                    "bundle_path = None\n",
                    "# Check if already uploaded directly to /content/\n",
                    "if os.path.exists('/content/ETGNN_Technical_Pilot_Bundle.zip'):\n",
                    "    sz = os.path.getsize('/content/ETGNN_Technical_Pilot_Bundle.zip')\n",
                    "    if sz >= 40000000:\n",
                    "        bundle_path = '/content/ETGNN_Technical_Pilot_Bundle.zip'\n",
                    "        print(f'Found valid complete bundle in /content/ ({sz:,} bytes).')\n",
                    "    else:\n",
                    "        print(f'Detected incomplete upload in /content/ ({sz:,} / 43,534,835 bytes).')\n",
                    "\n",
                    "if bundle_path is None:\n",
                    "    print('Mounting Google Drive to look for bundle...')\n",
                    "    try:\n",
                    "        from google.colab import drive\n",
                    "        drive.mount('/content/drive')\n",
                    "        drive_candidates = [\n",
                    "            '/content/drive/MyDrive/ETGNN_Technical_Pilot_Bundle.zip',\n",
                    "            '/content/drive/MyDrive/Colab Notebooks/ETGNN_Technical_Pilot_Bundle.zip',\n",
                    "        ]\n",
                    "        for c in drive_candidates:\n",
                    "            if os.path.exists(c):\n",
                    "                sz = os.path.getsize(c)\n",
                    "                if sz >= 40000000:\n",
                    "                    bundle_path = c\n",
                    "                    print(f'Found complete bundle in Google Drive: {c} ({sz:,} bytes)')\n",
                    "                    break\n",
                    "    except Exception as e:\n",
                    "        print(f'Drive mount skipped or failed: {e}')\n",
                    "\n",
                    "if bundle_path:\n",
                    "    print('SUCCESS: Bundle located at:', bundle_path)\n",
                    "else:\n",
                    "    print('Bundle not found in Drive. Please run Option B below to upload via progress widget.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Option B: Direct Upload via Colab Widget (with Progress Bar)\n",
                    "Run this cell only if Option A was not used."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if bundle_path is None:\n",
                    "    from google.colab import files\n",
                    "    print('Please select ETGNN_Technical_Pilot_Bundle.zip from your computer:')\n",
                    "    uploaded = files.upload()\n",
                    "    for fn in uploaded.keys():\n",
                    "        if fn.endswith('.zip'):\n",
                    "            bundle_path = os.path.abspath(fn)\n",
                    "            print(f'Uploaded: {bundle_path} ({os.path.getsize(bundle_path):,} bytes)')\n",
                    "            break"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Verify Size & Extract Archive"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, zipfile\n",
                    "\n",
                    "EXPECTED_BYTES = 43534835\n",
                    "if not bundle_path or not os.path.exists(bundle_path):\n",
                    "    raise FileNotFoundError('ETGNN_Technical_Pilot_Bundle.zip has not been uploaded yet.')\n",
                    "\n",
                    "actual_bytes = os.path.getsize(bundle_path)\n",
                    "print(f'Archive location: {bundle_path}')\n",
                    "print(f'Archive size:     {actual_bytes:,} bytes (Expected: {EXPECTED_BYTES:,} bytes)')\n",
                    "\n",
                    "if actual_bytes < 40000000:\n",
                    "    raise RuntimeError(\n",
                    "        f'TRUNCATED FILE ERROR: Archive is only {actual_bytes:,} bytes, which is missing data! '\n",
                    "        f'The complete file must be {EXPECTED_BYTES:,} bytes. '\n",
                    "        'Please upload the bundle to your Google Drive (MyDrive) and mount it via Option A, '\n",
                    "        'or use Option B files.upload() to re-upload completely.'\n",
                    "    )\n",
                    "\n",
                    "work_dir = '/content/pilot_workdir'\n",
                    "if os.path.exists(work_dir):\n",
                    "    import shutil\n",
                    "    shutil.rmtree(work_dir)\n",
                    "os.makedirs(work_dir, exist_ok=True)\n",
                    "\n",
                    "print(f'Extracting to {work_dir}...')\n",
                    "with zipfile.ZipFile(bundle_path, 'r') as z:\n",
                    "    z.extractall(work_dir)\n",
                    "\n",
                    "os.chdir(work_dir)\n",
                    "print('SUCCESS: Extraction complete.')\n",
                    "!ls -lh"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Cryptographic SHA-256 Manifest Verification"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import hashlib\n",
                    "\n",
                    "print('Verifying extracted artifacts against manifest.sha256...')\n",
                    "with open('manifest.sha256', 'r') as f:\n",
                    "    lines = f.readlines()\n",
                    "\n",
                    "all_passed = True\n",
                    "for line in lines:\n",
                    "    line = line.strip()\n",
                    "    if not line:\n",
                    "        continue\n",
                    "    expected_hash, sz, rel_path = line.split(maxsplit=2)\n",
                    "    if not os.path.exists(rel_path):\n",
                    "        print(f'FAIL: {rel_path} missing!')\n",
                    "        all_passed = False\n",
                    "        continue\n",
                    "    h = hashlib.sha256(open(rel_path, 'rb').read()).hexdigest()\n",
                    "    if h != expected_hash:\n",
                    "        print(f'FAIL: Hash mismatch for {rel_path}!')\n",
                    "        all_passed = False\n",
                    "    else:\n",
                    "        print(f'OK: {rel_path:<45} ({int(sz):>10,} bytes)')\n",
                    "\n",
                    "assert all_passed, 'Artifact manifest verification failed!'\n",
                    "print('\\nAll 13 artifacts cryptographically verified.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Install Dependencies"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install -q torch-geometric psutil"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Execute Technical Pilot Benchmark"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "if '/content/pilot_workdir' not in sys.path:\n",
                    "    sys.path.insert(0, '/content/pilot_workdir')\n",
                    "\n",
                    "from src.evaluation.run_technical_pilot_benchmark import run_benchmark\n",
                    "\n",
                    "summary_df, summary_json = run_benchmark(\n",
                    "    sample_csv_path='experiments/matched_c1_c2_audit/technical_pilot_sample.csv',\n",
                    "    val_graph_path='data/processed/validation_graph.pt',\n",
                    "    node_mapping_path='data/processed/node_mapping.csv',\n",
                    "    etgnn_ckpt_path='results/checkpoints/etgnn.pt',\n",
                    "    matched_gat_ckpt_path='results/checkpoints/matched_gat.pt',\n",
                    "    output_dir='experiments/technical_pilot_results',\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Display Benchmark Measurements & Runtime Projections"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "\n",
                    "sum_path = 'experiments/technical_pilot_results/technical_pilot_benchmark_summary.csv'\n",
                    "df_res = pd.read_csv(sum_path)\n",
                    "print('=== MEASURED TECHNICAL PILOT BENCHMARK RESULTS (4 NODES) ===')\n",
                    "cols = [\n",
                    "    'category', 'node_index', 'E_cand', 'k_budget', 'actual_k_over_E',\n",
                    "    'extract_runtime_ms', 'gnnexplainer_mean_runtime_ms', 'random_30_total_runtime_ms',\n",
                    "    'peak_rss_mb', 'peak_vram_mb', 'advantage_A_etgnn', 'advantage_A_gat'\n",
                    "]\n",
                    "display(df_res[cols])\n",
                    "\n",
                    "mean_ge_s = df_res['gnnexplainer_mean_runtime_ms'].mean() / 1000.0\n",
                    "mean_rnd_s = df_res['random_30_total_runtime_ms'].mean() / 1000.0\n",
                    "mean_ext_s = df_res['extract_runtime_ms'].mean() / 1000.0\n",
                    "total_per_node_s = (mean_ge_s * 3.0) + mean_rnd_s + mean_ext_s\n",
                    "\n",
                    "print(f'\\nMeasured average runtime per node (3 seeds + 30 random): {total_per_node_s:.2f} seconds')\n",
                    "print('\\n=== EMPIRICAL RUNTIME PROJECTIONS FOR MAIN EVALUATION (STEPS 37–40) ===')\n",
                    "for n_pairs, label in [\n",
                    "    (57, 'Degree-Balanced Sensitivity Match (57 pairs = 114 nodes)'),\n",
                    "    (95, 'Primary ETGNN_full TP-FP Match (95 pairs = 190 nodes)'),\n",
                    "    (120, 'All Achievable Main Follow-Up Pairs (120 pairs = 240 nodes)'),\n",
                    "]:\n",
                    "    n_nodes = n_pairs * 2\n",
                    "    proj_sec = n_nodes * total_per_node_s\n",
                    "    proj_min = proj_sec / 60.0\n",
                    "    print(f'{label:<60} | {proj_min:6.1f} min ({proj_sec:7.1f} s)')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Package Results for Download"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!zip -r /content/technical_pilot_results.zip experiments/technical_pilot_results\n",
                    "print('SUCCESS: Results packaged into /content/technical_pilot_results.zip.')\n",
                    "print('Please download /content/technical_pilot_results.zip and provide it for synthesis.')"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "ETGNN_Technical_Pilot_Colab.ipynb",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_file = Path("ETGNN_Technical_Pilot_Colab.ipynb")
    out_file.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Generated updated {out_file} ({os.path.getsize(out_file)} bytes).")


if __name__ == "__main__":
    create_colab_notebook()
