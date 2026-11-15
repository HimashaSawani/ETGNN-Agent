from pathlib import Path

root = Path(".")
print("Searching for trained models and benchmark results in results/ ...")
for ext in ["*.joblib", "*.pkl", "*.model", "*.bin", "*.json", "*.csv"]:
    for f in (root / "results").rglob(ext):
        if any(term in f.name.lower() for term in ["rf", "random_forest", "xgb", "xgboost", "baseline"]):
            print(f"  {str(f.relative_to(root)):60s} | {f.stat().st_size:,} bytes")

