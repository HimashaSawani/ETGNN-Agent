"""
Packages the complete ETGNN Validation Pilot Evidence Audit Bundle.
Excludes large graph datasets, raw data, credentials, and virtual environments.
"""

import os
import zipfile
from pathlib import Path

def package_audit():
    out_zip = Path("ETGNN_Pilot_Evidence_Audit_Bundle.zip")
    
    files_to_pack = [
        # 1. Pilot runner & verification scripts
        "src/evaluation/run_validation_pilot.py",
        "src/evaluation/audit_pilot_evidence.py",
        "src/evaluation/verify_results_and_figures.py",
        "src/evaluation/metrics.py",
        "src/__init__.py",
        "src/evaluation/__init__.py",
        
        # 2. Model dependencies
        "src/models/__init__.py",
        "src/models/etgnn.py",
        "src/models/time_encoding.py",
        
        # 3. Configuration
        "configs/experiment_config.yaml",
        "experiments/pilot_validation_audit_v2/config.json",
        
        # 4. Verification output
        "experiments/pilot_validation_audit_v2/verification_output.txt",
        
        # 5. Raw CSVs
        "experiments/pilot_validation_audit_v2/raw_interventions.csv",
        "experiments/pilot_validation_audit_v2/pilot_nodes.csv",
        "experiments/pilot_validation_audit_v2/candidate_edges.csv",
        "experiments/pilot_validation_audit_v2/reconciliation_table.csv",
        "experiments/pilot_validation_audit_v2/fp_seed_outcomes.csv",
        "experiments/pilot_validation_audit_v2/tn_induction_outcomes.csv",
        "experiments/pilot_validation_audit_v2/quadrant_characteristics.csv",
        "experiments/pilot_validation_audit_v2/jaccard_stability_comparison.csv",
        
        # 6. Corrected summaries
        "experiments/pilot_validation_audit_v2/aggregate_summary_corrected.csv",
        "experiments/pilot_validation_audit_v2/node_summary_corrected.csv",
        "experiments/pilot_validation_audit_v2/quadrant_summary_corrected.csv",
        
        # 7. Masks
        "experiments/pilot_validation_audit_v2/explanations/raw_edge_masks.json",
        
        # 8. Discrepancy logs & provenance
        "experiments/pilot_validation_audit_v2/candidate_edge_discrepancy_log.md",
        "experiments/pilot_validation_audit_v2/colab_execution_provenance.md",
        
        # 9. Corrected report
        "experiments/pilot_validation_audit_v2/pilot_report_v2.md",
        
        # 10. Reproduction instructions
        "experiments/pilot_validation_audit_v2/reproduction_instructions.md",
    ]
    
    # 11. Plots
    plots_dir = Path("experiments/pilot_validation_audit_v2/plots")
    if plots_dir.exists():
        for p in plots_dir.glob("*"):
            files_to_pack.append(str(p).replace("\\", "/"))
            
    print(f"Creating audit package {out_zip.resolve()}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files_to_pack:
            p = Path(f)
            if p.exists():
                # Store relative to project root
                zf.write(p, arcname=f)
                print(f"  Added: {f} ({p.stat().st_size:,} bytes)")
            else:
                raise FileNotFoundError(f"Missing required artifact: {f}")
                
    size_bytes = out_zip.stat().st_size
    size_kb = size_bytes / 1024
    print(f"\nAudit package successfully built: {out_zip.name}")
    print(f"Total size: {size_bytes:,} bytes ({size_kb:.1f} KB)")
    return out_zip

if __name__ == "__main__":
    package_audit()
