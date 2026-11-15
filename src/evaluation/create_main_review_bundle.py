import os
import zipfile
import hashlib
from pathlib import Path


def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    bundle_zip = Path("ETGNN_C1_C2_Main_Review_Bundle.zip")
    if bundle_zip.exists():
        bundle_zip.unlink()

    files_to_pack = [
        # 1. Corrected benchmark runner and model definitions
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/etgnn.py",
        "src/models/matched_gat.py",
        "src/models/time_encoding.py",
        "src/evaluation/__init__.py",
        "src/evaluation/run_technical_pilot_benchmark.py",

        # 2. Corrected development raw/detailed records, summary, and execution environment
        "experiments/technical_pilot_results/raw_benchmark_records.csv",
        "experiments/technical_pilot_results/detailed_intervention_records.csv",
        "experiments/technical_pilot_results/technical_pilot_benchmark_summary.csv",
        "experiments/technical_pilot_results/technical_pilot_summary.json",
        "experiments/technical_pilot_results/input_manifest.json",
        "experiments/technical_pilot_results/pilot_resume_state.json",

        # Preserved technical pilot version 1 artifacts
        "experiments/technical_pilot_results_v1/raw_benchmark_records.csv",
        "experiments/technical_pilot_results_v1/technical_pilot_benchmark_summary.csv",
        "experiments/technical_pilot_results_v1/technical_pilot_summary.json",
        "experiments/technical_pilot_results_v1/input_manifest.json",
        "experiments/technical_pilot_results_v1/pilot_resume_state.json",

        # 3. Main C1+C2 Run Configuration
        "configs/main_c1_c2_run_config.json",

        # 4. Main Execution Plan
        "reports/main_c1_c2_execution_plan.md",

        # 5. Every primary and degree-sensitivity pair register
        "experiments/matched_c1_c2_audit/caliper_pairs_etgnn_full_tp_fp.csv",
        "experiments/matched_c1_c2_audit/caliper_pairs_etgnn_full_tn_fn.csv",
        "experiments/matched_c1_c2_audit/caliper_pairs_matched_gat_tp_fp.csv",
        "experiments/matched_c1_c2_audit/caliper_pairs_matched_gat_tn_fn.csv",
        "experiments/matched_c1_c2_audit/degree_sensitivity_pairs_etgnn_full_tp_fp.csv",
        "experiments/matched_c1_c2_audit/degree_sensitivity_pairs_etgnn_full_tn_fn.csv",
        "experiments/matched_c1_c2_audit/degree_sensitivity_pairs_matched_gat_tp_fp.csv",
        "experiments/matched_c1_c2_audit/degree_sensitivity_pairs_matched_gat_tn_fn.csv",
        "experiments/matched_c1_c2_audit/technical_pilot_sample.csv",

        # 6. Input provenance reconciliation and current manifest
        "experiments/technical_pilot_results/input_provenance_reconciliation.md",
        "manifest.sha256",
    ]

    print(f"Packaging {len(files_to_pack)} review artifacts into {bundle_zip}...")
    manifest_records = []
    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files_to_pack:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing required review artifact: {f}")
            h = get_sha256(f)
            sz = os.path.getsize(f)
            manifest_records.append(f"{h}  {sz:>10}  {f}")
            z.write(f, f)
            print(f"  Packed: {f:<75} ({sz:>10,} bytes)")

    bundle_sha = get_sha256(str(bundle_zip))
    bundle_size = os.path.getsize(bundle_zip)
    print(f"\nSuccessfully created {bundle_zip}:")
    print(f"  Size:   {bundle_size:,} bytes")
    print(f"  SHA256: {bundle_sha}")
    print(f"  Files:  {len(files_to_pack)}")


if __name__ == "__main__":
    main()
