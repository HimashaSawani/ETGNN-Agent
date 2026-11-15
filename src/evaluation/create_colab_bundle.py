"""
Script to create ETGNN_Colab_Bundle.zip containing all required source modules,
checkpoints, validation graph, mapping files, and pilot records.
Excludes .venv, raw datasets, test graphs (preserving test isolation),
and unrelated model checkpoints.
"""

import os
import zipfile
from pathlib import Path

def create_bundle(output_zip: str = "ETGNN_Colab_Bundle.zip"):
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
    
    # Explicitly include src/__init__.py
    if os.path.exists("src/__init__.py"):
        files_to_include.append("src/__init__.py")
        
    # 2. Specifically required checkpoint (etgnn_local.pt only)
    ckpt = "results/checkpoints/etgnn_local.pt"
    if os.path.exists(ckpt):
        files_to_include.append(ckpt)
    else:
        raise FileNotFoundError(f"Missing required checkpoint: {ckpt}")
        
    # 3. Validation graph and mappings (strictly excluding train_graph and test_graph)
    val_files = [
        "data/processed/validation_graph.pt",
        "data/processed/node_mapping.csv",
        "data/processed/nodes_with_masks.csv"
    ]
    for vf in val_files:
        if os.path.exists(vf):
            files_to_include.append(vf)
        else:
            raise FileNotFoundError(f"Missing required data artifact: {vf}")
            
    # 4. Saved pilot records and explanations
    pilot_dir = "experiments/pilot_validation_audit"
    if os.path.exists(pilot_dir):
        for root, _, files in os.walk(pilot_dir):
            for f in files:
                files_to_include.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(f"Missing pilot experiment directory: {pilot_dir}")

    # 5. Connected explanation case JSONs and summary tables
    exp_dir = "results/explanations"
    if os.path.exists(exp_dir):
        for root, _, files in os.walk(exp_dir):
            for f in files:
                files_to_include.append(os.path.join(root, f))
                
    # 6. Configuration & key reports
    if os.path.exists("configs/experiment_config.yaml"):
        files_to_include.append("configs/experiment_config.yaml")
        
    for report_file in ["reports/explainability_findings.md", "reports/research_audit.md", "reports/etgnn_design.md"]:
        if os.path.exists(report_file):
            files_to_include.append(report_file)

    # Remove duplicates and normalize paths
    unique_files = sorted(list(set(os.path.normpath(p).replace("\\", "/") for p in files_to_include)))
    
    print(f"Packaging {len(unique_files)} files into {zip_path.resolve()}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in unique_files:
            zf.write(f, arcname=f)
            
    bundle_size_bytes = zip_path.stat().st_size
    bundle_size_mb = bundle_size_bytes / (1024 * 1024)
    print(f"Bundle successfully created: {zip_path.name}")
    print(f"Total size: {bundle_size_bytes:,} bytes ({bundle_size_mb:.2f} MB)")
    return zip_path, bundle_size_bytes

if __name__ == "__main__":
    create_bundle()
