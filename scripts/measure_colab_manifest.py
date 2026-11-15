import os
import json
from pathlib import Path

root = Path(".")

with open("artifacts/baseline_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"{'#':2s} | {'Component':32s} | {'Windows Source Path':58s} | {'Size (Bytes)':14s} | {'Size (MB)':10s} | {'Type':10s}")
print("-" * 136)

idx = 1
total_mand = 0
total_rec = 0
total_opt = 0

# 1. Mandatory Core Execution Files
mand_files = [
    ("Notebook", "notebooks/phase6_5/colab_verification.ipynb"),
    ("Baseline Manifest", "artifacts/baseline_manifest.json"),
    ("Processed Test Graph", "data/processed/test_graph.pt"),
    ("Processed Node Metadata", "data/processed/nodes_with_masks.csv"),
    ("ETGNN Checkpoint", "results/checkpoints/etgnn.pt"),
    ("ETGNN Model Architecture", "src/models/etgnn.py"),
    ("Time Encoding Architecture", "src/models/time_encoding.py"),
    ("Models Init Module", "src/models/__init__.py"),
    ("Src Root Init Module", "src/__init__.py"),
]

for label, rel in mand_files:
    p = root / rel
    size = p.stat().st_size if p.exists() else 0
    total_mand += size
    print(f"{idx:2d} | {label:32s} | {str(p):58s} | {size:14,d} | {size/(1024*1024):9.2f} MB | Mandatory")
    idx += 1

# 2. Recommended Full Checkpoints (All remaining 10 in baseline manifest)
rec_checkpoints = [
    ("Matched GAT Checkpoint", "results/checkpoints/matched_gat.pt"),
    ("GAT Baseline Checkpoint", "results/checkpoints/gat.pt"),
    ("GCN Baseline Checkpoint", "results/checkpoints/gcn.pt"),
    ("GraphSAGE Baseline Checkpoint", "results/checkpoints/graphsage.pt"),
    ("Random Forest Checkpoint", "results/checkpoints/random_forest.joblib"),
    ("RF Local Checkpoint", "results/checkpoints/rf_local.joblib"),
    ("RF With Time Checkpoint", "results/checkpoints/rf_with_time.joblib"),
    ("XGBoost Checkpoint", "results/checkpoints/xgboost.json"),
    ("ETGNN Local Checkpoint", "results/checkpoints/etgnn_local.pt"),
    ("ETGNN Local No Edges Checkpoint", "results/checkpoints/etgnn_local_no_edges.pt"),
    ("Matched GAT Model Arch", "src/models/matched_gat.py"),
    ("GAT Model Arch", "src/models/gat.py"),
    ("GCN Model Arch", "src/models/gcn.py"),
    ("GraphSAGE Model Arch", "src/models/graphsage.py"),
]

for label, rel in rec_checkpoints:
    p = root / rel
    size = p.stat().st_size if p.exists() else 0
    total_rec += size
    print(f"{idx:2d} | {label:32s} | {str(p):58s} | {size:14,d} | {size/(1024*1024):9.2f} MB | Recommended")
    idx += 1

# 3. Optional Raw Datasets
raw_files = [
    ("Elliptic Classes Raw CSV", "data/raw/elliptic/elliptic_txs_classes.csv"),
    ("Elliptic Edgelist Raw CSV", "data/raw/elliptic/elliptic_txs_edgelist.csv"),
    ("Elliptic Features Raw CSV", "data/raw/elliptic/elliptic_txs_features.csv"),
]

for label, rel in raw_files:
    p = root / rel
    size = p.stat().st_size if p.exists() else 0
    total_opt += size
    print(f"{idx:2d} | {label:32s} | {str(p):58s} | {size:14,d} | {size/(1024*1024):9.2f} MB | Optional")
    idx += 1

print("=" * 136)
print(f"Mandatory Core Upload Size         : {total_mand:,d} bytes ({total_mand/(1024*1024):.2f} MB)")
print(f"Recommended Checkpoints Upload Size: {total_rec:,d} bytes ({total_rec/(1024*1024):.2f} MB)")
print(f"Total Lean Benchmark Upload (1+2)  : {(total_mand + total_rec):,d} bytes ({(total_mand + total_rec)/(1024*1024):.2f} MB)")
print(f"Optional Raw Datasets Size         : {total_opt:,d} bytes ({total_opt/(1024*1024):.2f} MB)")
