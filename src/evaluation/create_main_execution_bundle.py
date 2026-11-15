#!/usr/bin/env python3
"""
create_main_execution_bundle.py
===============================
Packages the complete, authoritative execution bundle for the Main C1 + C2
benchmark on the 377-node execution union in steps 37–40.

Includes:
- All source code and model definitions
- Authoritative main benchmark runner (run_main_c1_c2_benchmark.py)
- Fixed technical pilot benchmark runner (run_technical_pilot_benchmark.py)
- Frozen configuration (main_c1_c2_run_config.json)
- All 8 primary and sensitivity pair registers
- Dev pilot sample (technical_pilot_sample.csv)
- Checkpoints (etgnn.pt, matched_gat.pt)
- Processed computation graph (validation_graph.pt)
- Node mapping (node_mapping.csv)
- Cryptographic manifest (manifest.sha256)
"""

import zipfile
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_bundle():
    bundle_path = PROJECT_ROOT / "ETGNN_Main_C1_C2_Execution_Bundle.zip"

    files_to_pack = [
        # Source code
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/etgnn.py",
        "src/models/matched_gat.py",
        "src/models/time_encoding.py",
        "src/evaluation/__init__.py",
        "src/evaluation/run_main_c1_c2_benchmark.py",
        "src/evaluation/run_technical_pilot_benchmark.py",

        # Configuration
        "configs/main_c1_c2_run_config.json",

        # Pair registers
        "experiments/matched_c1_c2_audit/caliper_pairs_etgnn_full_tp_fp.csv",
        "experiments/matched_c1_c2_audit/caliper_pairs_etgnn_full_tn_fn.csv",
        "experiments/matched_c1_c2_audit/caliper_pairs_matched_gat_tp_fp.csv",
        "experiments/matched_c1_c2_audit/caliper_pairs_matched_gat_tn_fn.csv",
        "experiments/matched_c1_c2_audit/degree_sensitivity_pairs_etgnn_full_tp_fp.csv",
        "experiments/matched_c1_c2_audit/degree_sensitivity_pairs_etgnn_full_tn_fn.csv",
        "experiments/matched_c1_c2_audit/degree_sensitivity_pairs_matched_gat_tp_fp.csv",
        "experiments/matched_c1_c2_audit/degree_sensitivity_pairs_matched_gat_tn_fn.csv",
        "experiments/matched_c1_c2_audit/technical_pilot_sample.csv",
        "experiments/pilot_validation_audit/pilot_nodes.csv",
        "src/evaluation/test_c2_all_registers_synthetic.py",

        # Checkpoints
        "results/checkpoints/etgnn.pt",
        "results/checkpoints/matched_gat.pt",

        # Data artifacts
        "data/processed/validation_graph.pt",
        "data/processed/node_mapping.csv",

        # Manifest
        "manifest.sha256",
    ]

    print(f"Packaging {len(files_to_pack)} artifacts into {bundle_path.name}...")

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel_path in files_to_pack:
            src_file = PROJECT_ROOT / rel_path
            assert src_file.exists(), f"Missing required file: {src_file}"
            zf.write(src_file, arcname=rel_path)
            print(f"  Packed: {rel_path:60s} ({src_file.stat().st_size:>12,d} bytes)")

    bundle_size = bundle_path.stat().st_size
    bundle_hash = compute_sha256(bundle_path)

    print(f"\nSuccessfully created {bundle_path.name}:")
    print(f"  Size:   {bundle_size:,d} bytes")
    print(f"  SHA256: {bundle_hash}")
    print(f"  Files:  {len(files_to_pack)}")

    return bundle_path, bundle_size, bundle_hash


if __name__ == "__main__":
    build_bundle()
