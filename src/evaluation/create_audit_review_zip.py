import os
import zipfile
from pathlib import Path


def main():
    zip_path = Path("experiments/matched_c1_c2_audit/C1_C2_Audit_Review_Bundle.zip")
    if zip_path.exists():
        zip_path.unlink()

    files_to_zip = [
        ("src/evaluation/run_c1_c2_audit_v2.py", "scripts/run_c1_c2_audit_v2.py"),
        ("src/evaluation/generate_verified_report_tables.py", "scripts/generate_verified_report_tables.py"),
        ("src/evaluation/technical_pilot_plan.py", "scripts/technical_pilot_plan.py"),
        ("reports/c1_c2_protocol_amendment.md", "reports/c1_c2_protocol_amendment.md"),
        ("experiments/matched_c1_c2_audit/technical_pilot_sample.csv", "data/technical_pilot_sample.csv"),
        ("experiments/matched_c1_c2_audit/assertion_verification_results.csv", "tables/assertion_verification_results.csv"),
        ("experiments/matched_c1_c2_audit/verified_caliper_balance_diagnostics.csv", "tables/verified_caliper_balance_diagnostics.csv"),
        ("experiments/matched_c1_c2_audit/verified_caliper_within_pair_differences.csv", "tables/verified_caliper_within_pair_differences.csv"),
        ("experiments/matched_c1_c2_audit/degree_sensitivity_balance_summary.csv", "tables/degree_sensitivity_balance_summary.csv"),
        ("experiments/matched_c1_c2_audit/automated_report_tables.md", "tables/automated_report_tables.md"),
        ("experiments/matched_c1_c2_audit/balance_diagnostics_summary.csv", "tables/balance_diagnostics_summary.csv"),
    ]

    for m in ["etgnn_full", "matched_gat", "etgnn_local"]:
        files_to_zip.append((f"experiments/matched_c1_c2_audit/feasibility_{m}.csv", f"feasibility/feasibility_{m}.csv"))

    audit_dir = Path("experiments/matched_c1_c2_audit")
    for f in sorted(os.listdir(audit_dir)):
        if f.startswith("caliper_pairs_") and f.endswith(".csv"):
            files_to_zip.append((str(audit_dir / f), f"pairs/{f}"))
        if f.startswith("degree_sensitivity_pairs_") and f.endswith(".csv"):
            files_to_zip.append((str(audit_dir / f), f"degree_sensitivity_pairs/{f}"))
        if f.startswith("order_sensitivity_") and f.endswith(".csv"):
            files_to_zip.append((str(audit_dir / f), f"order_sensitivity/{f}"))

    plot_dir = audit_dir / "plots"
    for f in sorted(os.listdir(plot_dir)):
        if f.endswith(".png"):
            files_to_zip.append((str(plot_dir / f), f"plots/{f}"))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, dst in files_to_zip:
            if os.path.exists(src):
                z.write(src, dst)
            else:
                print(f"Warning: {src} missing!")

    print(f"SUCCESS: Built {zip_path} ({os.path.getsize(zip_path)} bytes, {len(files_to_zip)} files).")


if __name__ == "__main__":
    main()
