"""
Summary verification script for Google Colab and Google Drive integration
"""
from pathlib import Path

def print_summary():
    staging = Path("google_drive_staging/ETGNN-Fraud-Detection")
    files = [
        staging / "notebooks/ETGNN_End_to_End_Training_and_Benchmark.ipynb",
        staging / "data/processed/train_graph.pt",
        staging / "data/processed/validation_graph.pt",
        staging / "data/processed/test_graph.pt",
        staging / "data/processed/scaled_features.npy",
        staging / "data/processed/nodes_with_masks.csv",
    ]
    
    print("Staged Files for Google Drive Upload:")
    total_size = 0
    for f in files:
        assert f.exists(), f"Missing {f}"
        sz = f.stat().st_size
        total_size += sz
        print(f"  - {f.relative_to(staging)}: {sz / (1024**2):.2f} MB")
    print(f"Total Upload Size: {total_size / (1024**2):.2f} MB")

if __name__ == "__main__":
    print_summary()

