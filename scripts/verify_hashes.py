import json
import hashlib
from pathlib import Path

def main():
    with open("artifacts/baseline_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)["scientific_inventory"]

    checkpoints = manifest["checkpoints"]
    datasets = manifest["datasets"]

    c_matches = {}
    for name, info in checkpoints.items():
        p = Path(info["path"])
        if not p.exists():
            c_matches[name] = False
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        c_matches[name] = (h == info["sha256"])

    d_matches = {}
    for name, info in datasets.items():
        p = Path(info["path"])
        if not p.exists():
            d_matches[name] = False
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        d_matches[name] = (h == info["sha256"])

    c_all = all(c_matches.values()) and len(c_matches) == 11
    d_all = all(d_matches.values()) and len(d_matches) == 4

    print(f"Checkpoints: {sum(c_matches.values())}/{len(c_matches)} match (All OK: {c_all})")
    print(f"Datasets: {sum(d_matches.values())}/{len(d_matches)} match (All OK: {d_all})")
    return c_all and d_all

if __name__ == "__main__":
    main()
