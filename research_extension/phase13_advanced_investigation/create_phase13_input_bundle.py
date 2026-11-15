import zipfile
import hashlib
from pathlib import Path

ROOT = Path(".")
out_zip = ROOT / "ETGNN_Phase13_Input_Bundle.zip"

files_to_pack = [
    ROOT / "data" / "processed" / "train_graph.pt",
    ROOT / "data" / "processed" / "validation_graph.pt",
    ROOT / "data" / "processed" / "test_graph.pt",
    ROOT / "data" / "processed" / "nodes_with_masks.csv",
    ROOT / "results" / "checkpoints" / "etgnn.pt",
    ROOT / "results" / "checkpoints" / "matched_gat.pt",
    ROOT / "results" / "main_c1_c2_results" / "c2_individual_pair_differences.csv",
    ROOT / "results" / "tables" / "c1_response_summary.csv",
    ROOT / "results" / "tables" / "c2_registers_bootstrap_summary.csv"
]

print("Packing ETGNN_Phase13_Input_Bundle.zip...")
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in files_to_pack:
        if f.exists():
            arcname = str(f.relative_to(ROOT)).replace("\\", "/")
            z.write(f, arcname=arcname)
            print(f"  Added: {arcname} ({f.stat().st_size:,} bytes)")
        else:
            print(f"  WARNING: Missing {f}")

size_mb = out_zip.stat().st_size / (1024 * 1024)
sha = hashlib.sha256(out_zip.read_bytes()).hexdigest()
print(f"[SUCCESS] Created {out_zip.name} ({size_mb:.1f} MB) | SHA-256: {sha}")

