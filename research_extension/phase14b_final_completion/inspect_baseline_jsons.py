import json
from pathlib import Path

for json_file in [
    Path("results/tables/rf_validation.json"),
    Path("results/tables/rf_local_validation.json"),
    Path("results/tables/rf_with_time_validation.json"),
    Path("results/tables/xgb_validation.json")
]:
    if json_file.exists():
        print(f"=== {json_file.name} ===")
        print(json_file.read_text())

