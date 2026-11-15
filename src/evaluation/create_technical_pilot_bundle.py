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
    bundle_zip = Path("ETGNN_Technical_Pilot_Bundle.zip")
    if bundle_zip.exists():
        bundle_zip.unlink()

    files_to_pack = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/etgnn.py",
        "src/models/matched_gat.py",
        "src/models/time_encoding.py",
        "src/evaluation/__init__.py",
        "src/evaluation/run_technical_pilot_benchmark.py",
        "data/processed/node_mapping.csv",
        "data/processed/validation_graph.pt",
        "results/checkpoints/etgnn.pt",
        "results/checkpoints/matched_gat.pt",
        "experiments/matched_c1_c2_audit/technical_pilot_sample.csv",
    ]

    manifest_lines = []
    for f in files_to_pack:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing bundle file: {f}")
        h = get_sha256(f)
        sz = os.path.getsize(f)
        manifest_lines.append(f"{h}  {sz:>10}  {f}")

    manifest_path = Path("manifest.sha256")
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    files_to_pack.append("manifest.sha256")

    print("Building ETGNN_Technical_Pilot_Bundle.zip...")
    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files_to_pack:
            z.write(f, f)
            print(f"  Packed: {f} ({os.path.getsize(f)} bytes)")

    print(f"\nSuccessfully created {bundle_zip} ({os.path.getsize(bundle_zip)} bytes, {len(files_to_pack)} files).")


if __name__ == "__main__":
    main()
