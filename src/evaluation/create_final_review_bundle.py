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
    bundle_zip = Path("ETGNN_C1_C2_Final_Review_Bundle.zip")
    if bundle_zip.exists():
        bundle_zip.unlink()

    files_to_pack = [
        # 1. Maintained reproduction and verification scripts
        "src/evaluation/create_derived_metadata.py",
        "src/evaluation/verify_subgraph_equivalence.py",
        "src/evaluation/verify_full_integration_evidence.py",
        "src/evaluation/generate_c1_c2_publication_artifacts.py",
        "src/evaluation/test_demo_http_server.py",
        "src/evaluation/update_manifest.py",
        "src/evaluation/run_main_c1_c2_benchmark.py",
        "src/evaluation/run_main_c1_c2_benchmark_executed_snapshot.py",

        # 2. Frozen configuration, environment dependencies, and provenance
        "configs/main_c1_c2_run_config.json",
        "results/main_c1_c2_results/main_resume_state.json",
        "requirements.txt",
        "results/verification_outputs.log",

        # 3. Derived file manifests and correction log
        "results/main_c1_c2_results/manifest.sha256",
        "reports/metadata_correction_log.md",
        "CHANGE_LOG.md",

        # 4. Publication figures (PNG + Vector PDF)
        "results/figures/c1_response_distributions.png",
        "results/figures/c1_response_distributions.pdf",
        "results/figures/c2_paired_differences_ci.png",
        "results/figures/c2_paired_differences_ci.pdf",
        "results/figures/c2_primary_vs_degree_sensitivity.png",
        "results/figures/c2_primary_vs_degree_sensitivity.pdf",
        "results/figures/candidate_edges_deletion_budget.png",
        "results/figures/candidate_edges_deletion_budget.pdf",

        # 5. Publication summary tables
        "results/tables/c1_response_summary.csv",
        "results/tables/c2_registers_bootstrap_summary.csv",
        "results/tables/c2_sensitivity_comparison.csv",
        "results/tables/candidate_edge_budget_summary.csv",

        # 6. Corrected derived data files
        "results/main_c1_c2_results/derived/c1_statistical_results_derived.csv",
        "results/main_c1_c2_results/derived/raw_benchmark_records_derived.csv",
        "results/main_c1_c2_results/derived/main_benchmark_summary_derived.csv",
        "results/main_c1_c2_results/derived/detailed_intervention_records_derived.csv",

        # 7. Updated research reports
        "reports/results_discussion_draft.md",
        "reports/methodology_draft.md",
        "reports/explainability_findings.md",
        "reports/reproduction_documentation.md",
        "reports/literature_review.md",
        "reports/literature_matrix.md",
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
            print(f"  Packed: {f:<70} ({sz:>10,} bytes)")

    bundle_sha = get_sha256(str(bundle_zip))
    bundle_size = os.path.getsize(bundle_zip)
    print(f"\nSuccessfully created {bundle_zip}:")
    print(f"  Size:   {bundle_size:,} bytes ({bundle_size / (1024*1024):.2f} MB)")
    print(f"  SHA256: {bundle_sha}")
    print(f"  Files:  {len(files_to_pack)}")


if __name__ == "__main__":
    main()
