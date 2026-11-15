"""
Creates the self-contained ETGNN_Large_Colab_Bundle.zip for the 80-node
validation evaluation on Google Colab.
Excludes .venv, raw elliptic datasets, training graph, and test graph.
"""

import os
import zipfile
from pathlib import Path

def create_large_bundle(output_zip: str = "ETGNN_Large_Colab_Bundle.zip"):
    project_root = Path(".")
    zip_path = project_root / output_zip
    
    files_to_include = []
    
    # 1. Source code and package initializers
    for src_dir in ["src/models", "src/evaluation", "src/data", "src/explain"]:
        for root, _, files in os.walk(src_dir):
            if "__pycache__" in root:
                continue
            for f in files:
                if not f.endswith(".pyc"):
                    files_to_include.append(os.path.join(root, f))
    
    if os.path.exists("src/__init__.py"):
        files_to_include.append("src/__init__.py")
        
    # 2. Checkpoint
    ckpt = "results/checkpoints/etgnn_local.pt"
    if os.path.exists(ckpt):
        files_to_include.append(ckpt)
    else:
        raise FileNotFoundError(f"Missing required checkpoint: {ckpt}")
        
    # 3. Validation graph & mappings
    for vf in ["data/processed/validation_graph.pt", "data/processed/node_mapping.csv", "data/processed/nodes_with_masks.csv"]:
        if os.path.exists(vf):
            files_to_include.append(vf)
        else:
            raise FileNotFoundError(f"Missing data artifact: {vf}")
            
    # 4. Large evaluation pre-registered artifacts
    large_exp_dir = "experiments/large_validation_audit"
    for lf in ["sampled_nodes_80.csv", "candidate_edges_80.csv", "protocol.md"]:
        p = os.path.join(large_exp_dir, lf)
        if os.path.exists(p):
            files_to_include.append(p)
            
    # 5. Protocol report & configs
    if os.path.exists("reports/larger_evaluation_protocol.md"):
        files_to_include.append("reports/larger_evaluation_protocol.md")
    if os.path.exists("configs/experiment_config.yaml"):
        files_to_include.append("configs/experiment_config.yaml")

    unique_files = sorted(list(set(os.path.normpath(p).replace("\\", "/") for p in files_to_include)))
    
    print(f"Packaging {len(unique_files)} files into {zip_path.resolve()}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in unique_files:
            zf.write(f, arcname=f)
            
    bundle_size_bytes = zip_path.stat().st_size
    bundle_size_mb = bundle_size_bytes / (1024 * 1024)
    print(f"Large evaluation bundle successfully created: {zip_path.name}")
    print(f"Total size: {bundle_size_bytes:,} bytes ({bundle_size_mb:.2f} MB)")
    return zip_path, bundle_size_bytes

if __name__ == "__main__":
    create_large_bundle()
