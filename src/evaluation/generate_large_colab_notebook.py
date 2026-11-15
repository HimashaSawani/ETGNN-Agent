"""
Generates the ETGNN_Large_Validation_Audit_Colab.ipynb notebook for the
80-node pre-registered validation evaluation on Google Colab.
"""

import json
from pathlib import Path

def create_large_notebook():
    cells = []
    
    # Cell 0: Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ETGNN Large-Scale Validation Evaluation (N=80 Stratified Nodes)\n",
            "## Pre-Registered Edge-Attribution Perturbation Audit on Google Colab\n",
            "\n",
            "**Target Sample**: $N = 80$ validation nodes (20 TP, 20 FP, 20 FN, 20 TN) sampled randomly from time steps 33–40 with $E_{\\mathrm{cand}} \\ge 2$.\n",
            "**Pre-Registered Protocol**: `reports/larger_evaluation_protocol.md`\n",
            "**Evaluated Checkpoint**: `results/checkpoints/etgnn_local.pt` (SHA256: `2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082`)\n",
            "**Target Data**: Validation snapshot `data/processed/validation_graph.pt`\n",
            "**Strict Test Isolation**: Test steps 41–49 are **strictly unaccessed** and excluded from this bundle.\n",
            "\n",
            "---\n",
            "### Evaluation Workflow\n",
            "1. **Drive Mount & Bundle Location**: Mounts Drive and locates `ETGNN_Large_Colab_Bundle.zip`.\n",
            "2. **Extraction under `/content`**: Unpacks project structure into `/content/etgnn_large_workspace`.\n",
            "3. **Hardware Inspection**: Verifies CPU/RAM (~12.7 GB in Colab) and sets 2 threads for safety.\n",
            "4. **PyG Setup**: Cleanly installs `torch_geometric` without replacing native Colab PyTorch.\n",
            "5. **Hash & Artifact Verification**: Asserts zero corruption and verifies sample integrity.\n",
            "6. **Live Execution Engine**: Evaluates 80 nodes across GNNExplainer (3 seeds), Uniform-Random (30 draws), and Degree ranking.\n",
            "7. **Macro Statistical Aggregation**: Computes node-level paired advantages, 95% Confidence Intervals, FP/FN correction rates, and TN induction rates.\n",
            "8. **Publication Plots & Drive Sync**: Renders figures, updates versioned Drive folder, and downloads results ZIP."
        ]
    })

    # Cell 1: Mount Drive and Locate Bundle
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 1] Mount Google Drive and Locate ETGNN_Large_Colab_Bundle.zip\n",
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    print(\"\\n[OK] Google Drive mounted successfully.\")\n",
            "except Exception as e:\n",
            "    print(\"[INFO] google.colab.drive not available or already mounted:\", e)\n",
            "\n",
            "candidate_paths = [\n",
            "    Path(\"/content/drive/MyDrive/ETGNN/ETGNN_Large_Colab_Bundle.zip\"),\n",
            "    Path(\"/content/drive/MyDrive/ETGNN_Large_Colab_Bundle.zip\"),\n",
            "    Path(\"/content/ETGNN_Large_Colab_Bundle.zip\"),\n",
            "]\n",
            "\n",
            "bundle_path = None\n",
            "for p in candidate_paths:\n",
            "    if p.exists():\n",
            "        bundle_path = p\n",
            "        break\n",
            "\n",
            "if bundle_path is None and os.path.exists(\"/content/drive/MyDrive\"):\n",
            "    print(\"[SEARCH] Searching for ETGNN_Large_Colab_Bundle.zip in /content/drive/MyDrive...\")\n",
            "    for root, _, files in os.walk(\"/content/drive/MyDrive\"):\n",
            "        if \"ETGNN_Large_Colab_Bundle.zip\" in files:\n",
            "            bundle_path = Path(root) / \"ETGNN_Large_Colab_Bundle.zip\"\n",
            "            break\n",
            "\n",
            "if bundle_path and bundle_path.exists():\n",
            "    size_mb = bundle_path.stat().st_size / (1024 * 1024)\n",
            "    print(f\"[FOUND] Large bundle located: {bundle_path} ({size_mb:.2f} MB)\")\n",
            "else:\n",
            "    raise FileNotFoundError(\"ERROR: 'ETGNN_Large_Colab_Bundle.zip' was not found in Drive or /content!\")"
        ]
    })

    # Cell 2: Copy and Extract
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 2] Extract Bundle under /content/etgnn_large_workspace\n",
            "import shutil\n",
            "import zipfile\n",
            "\n",
            "work_dir = Path(\"/content/etgnn_large_workspace\")\n",
            "work_dir.mkdir(parents=True, exist_ok=True)\n",
            "local_zip = work_dir / \"ETGNN_Large_Colab_Bundle.zip\"\n",
            "\n",
            "if bundle_path.resolve() != local_zip.resolve():\n",
            "    shutil.copy2(bundle_path, local_zip)\n",
            "\n",
            "with zipfile.ZipFile(local_zip, 'r') as zf:\n",
            "    zf.extractall(work_dir)\n",
            "\n",
            "os.chdir(work_dir)\n",
            "if str(work_dir) not in sys.path:\n",
            "    sys.path.insert(0, str(work_dir))\n",
            "\n",
            "print(f\"[OK] Workspace set: {os.getcwd()}\")\n",
            "print(\"[OK] Directories present:\", [p.name for p in work_dir.iterdir() if p.is_dir()])"
        ]
    })

    # Cell 3: Hardware Diagnostics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 3] Hardware & Runtime Diagnostics\n",
            "import platform\n",
            "import psutil\n",
            "import torch\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(f\"Platform:          {platform.platform()}\")\n",
            "print(f\"Python Version:    {platform.python_version()}\")\n",
            "print(f\"PyTorch Version:   {torch.__version__}\")\n",
            "ram = psutil.virtual_memory()\n",
            "print(f\"RAM Available:     {ram.available / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB\")\n",
            "print(f\"CPU Cores:         {psutil.cpu_count(logical=True)}\")\n",
            "torch.set_num_threads(2)\n",
            "print(f\"PyTorch Threads:   {torch.get_num_threads()}\")\n",
            "print(\"=\" * 60)"
        ]
    })

    # Cell 4: Install PyG Cleanly
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 4] Install PyTorch Geometric Cleanly\n",
            "import subprocess\n",
            "try:\n",
            "    import torch_geometric\n",
            "    print(f\"[OK] torch_geometric already installed: {torch_geometric.__version__}\")\n",
            "except ImportError:\n",
            "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"--no-cache-dir\", \"torch_geometric\"])\n",
            "    import torch_geometric\n",
            "    print(f\"[OK] Successfully installed torch_geometric: {torch_geometric.__version__}\")\n",
            "\n",
            "from torch_geometric.explain import Explainer, GNNExplainer\n",
            "from torch_geometric.nn import GATConv\n",
            "from torch_geometric.utils import degree, k_hop_subgraph\n",
            "print(\"[OK] PyG explanation modules verified.\")"
        ]
    })

    # Cell 5: Checkpoint & Graph Integrity
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 5] Cryptographic Hash Verification\n",
            "import hashlib\n",
            "\n",
            "def check_hash(path, expected):\n",
            "    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()\n",
            "    match = (h.lower() == expected.lower())\n",
            "    print(f\"{path}: {'MATCH' if match else 'MISMATCH'} ({h})\")\n",
            "    assert match, f\"Corrupted artifact: {path}\"\n",
            "\n",
            "check_hash(\"results/checkpoints/etgnn_local.pt\", \"2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082\")\n",
            "check_hash(\"data/processed/validation_graph.pt\", \"261b7ee5951d386bee9843adfc2471726ec9f433341d7c4359c7f66e44cbfc71\")\n",
            "assert not Path(\"data/processed/test_graph.pt\").exists(), \"Strict isolation violated: test_graph.pt found!\"\n",
            "print(\"[PASSED] Artifact hashes verified; test steps 41-49 strictly unaccessed.\")"
        ]
    })

    # Cell 6: Verify Pre-Registered 80-Node Sample
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 6] Verify Pre-Registered 80-Node Sample\n",
            "import pandas as pd\n",
            "\n",
            "sample_csv = Path(\"experiments/large_validation_audit/sampled_nodes_80.csv\")\n",
            "edges_csv = Path(\"experiments/large_validation_audit/candidate_edges_80.csv\")\n",
            "\n",
            "sample_df = pd.read_csv(sample_csv)\n",
            "edges_df = pd.read_csv(edges_csv)\n",
            "\n",
            "print(f\"Sample size: {len(sample_df)} nodes ({sample_df['case'].value_counts().to_dict()})\")\n",
            "print(f\"Total candidate edges: {len(edges_df)}\")\n",
            "assert len(sample_df) == 80, \"Expected exactly 80 nodes!\"\n",
            "assert (sample_df['case'].value_counts() == 20).all(), \"Expected exactly 20 nodes per quadrant!\"\n",
            "print(\"[PASSED] Stratified 80-node sample pre-registration verified.\")"
        ]
    })

    # Cell 7: Live Large Validation Execution Runner
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 7] Execute Large-Scale Validation Evaluation (N=80 Nodes, CPU)\n",
            "# Processes 1 node at a time with atomic progress saving and deduplicated resume support.\n",
            "from src.evaluation.run_large_validation_audit import main as run_large_audit\n",
            "\n",
            "print(\"[START] Launching 80-node evaluation on Colab CPU...\")\n",
            "run_large_audit(\"experiments/large_validation_audit\")\n",
            "print(\"[DONE] Large-scale evaluation successfully completed.\")"
        ]
    })

    # Cell 8: Display Summary Tables & CIs
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 8] Display Results, 95% Confidence Intervals, and Quadrant Breakdown\n",
            "import pandas as pd\n",
            "\n",
            "exp_dir = Path(\"experiments/large_validation_audit\")\n",
            "quad_df = pd.read_csv(exp_dir / \"quadrant_summary_80.csv\")\n",
            "fp_df = pd.read_csv(exp_dir / \"fp_seed_outcomes_80.csv\")\n",
            "tn_df = pd.read_csv(exp_dir / \"tn_induction_outcomes_80.csv\")\n",
            "\n",
            "print(\"=\" * 75)\n",
            "print(\"QUADRANT SUMMARY WITH 95% CONFIDENCE INTERVALS (N=80 Nodes):\")\n",
            "print(\"=\" * 75)\n",
            "print(quad_df[[\"case\", \"n_nodes\", \"paired_advantage_mean\", \"ci95_low\", \"ci95_high\", \"ge_correction_rate_macro\", \"ge_induction_rate_macro\"]].to_string(index=False))\n",
            "\n",
            "print(\"\\n\" + \"=\" * 75)\n",
            "print(\"FALSE POSITIVE CORRECTION BREAKDOWN (N=20 FP Transactions):\")\n",
            "print(\"=\" * 75)\n",
            "print(f\"Any-seed (>=1/3):   {fp_df['any_seed'].sum()}/{len(fp_df)} ({fp_df['any_seed'].mean():.1%})\")\n",
            "print(f\"Majority-seed (>=2): {fp_df['majority_seed'].sum()}/{len(fp_df)} ({fp_df['majority_seed'].mean():.1%})\")\n",
            "print(f\"All-seed (3/3):     {fp_df['all_seed'].sum()}/{len(fp_df)} ({fp_df['all_seed'].mean():.1%})\")\n",
            "\n",
            "print(\"\\n\" + \"=\" * 75)\n",
            "print(\"TRUE NEGATIVE ERROR INDUCTION BREAKDOWN (N=20 TN Transactions):\")\n",
            "print(\"=\" * 75)\n",
            "print(f\"Any-seed (>=1/3):   {tn_df['any_seed'].sum()}/{len(tn_df)} ({tn_df['any_seed'].mean():.1%})\")\n",
            "print(f\"Majority-seed (>=2): {tn_df['majority_seed'].sum()}/{len(tn_df)} ({tn_df['majority_seed'].mean():.1%})\")\n",
            "print(f\"All-seed (3/3):     {tn_df['all_seed'].sum()}/{len(tn_df)} ({tn_df['all_seed'].mean():.1%})\")"
        ]
    })

    # Cell 9: Render Publication Figures Inline
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 9] Display Publication Figure Inline\n",
            "from IPython.display import Image, display\n",
            "plot_p = Path(\"experiments/large_validation_audit/plots/large_paired_advantage_bars.png\")\n",
            "if plot_p.exists():\n",
            "    display(Image(str(plot_p)))\n",
            "else:\n",
            "    print(\"Plot not found:\", plot_p)"
        ]
    })

    # Cell 10: Drive Sync
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 10] Save Results to Versioned Google Drive Folder\n",
            "from datetime import datetime\n",
            "import shutil\n",
            "\n",
            "t_str = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
            "drive_out = Path(f\"/content/drive/MyDrive/ETGNN_Large_Audit_Results_{t_str}\")\n",
            "if os.path.exists(\"/content/drive/MyDrive\"):\n",
            "    drive_out.mkdir(parents=True, exist_ok=True)\n",
            "    for item in Path(\"experiments/large_validation_audit\").glob(\"*\"):\n",
            "        if item.is_file():\n",
            "            shutil.copy2(item, drive_out / item.name)\n",
            "        elif item.is_dir():\n",
            "            shutil.copytree(item, drive_out / item.name, dirs_exist_ok=True)\n",
            "    print(f\"[SUCCESS] Large validation audit synced to Drive:\\n  {drive_out}\")\n",
            "else:\n",
            "    print(\"[NOTE] /content/drive/MyDrive not mounted. Results remain local under /content.\")"
        ]
    })

    # Cell 11: Package and Download
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [CELL 11] Package and Download Result Archive\n",
            "import zipfile\n",
            "out_zip = Path(\"/content/ETGNN_Large_Audit_Results.zip\")\n",
            "src_dir = Path(\"experiments/large_validation_audit\")\n",
            "\n",
            "with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
            "    for root, _, files in os.walk(src_dir):\n",
            "        for f in files:\n",
            "            p = Path(root) / f\n",
            "            arc = p.relative_to(src_dir)\n",
            "            zf.write(p, arcname=f\"large_audit_results/{arc}\")\n",
            "\n",
            "print(f\"[SUCCESS] Archive created: {out_zip} ({out_zip.stat().st_size / 1024:.1f} KB)\")\n",
            "try:\n",
            "    from google.colab import files\n",
            "    files.download(str(out_zip))\n",
            "except Exception:\n",
            "    pass"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "NONE",
            "colab": {"name": "ETGNN_Large_Validation_Audit_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = Path("ETGNN_Large_Validation_Audit_Colab.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"[OK] Large Colab Notebook generated: {out_file.resolve()} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    create_large_notebook()
