import os
import zipfile
from pathlib import Path

zip_path = Path("results/etgnn_figures_and_results_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    # 1. Full source code (data, models, training, evaluation, explain)
    for src_dir in ["src/data", "src/models", "src/training", "src/evaluation", "src/explain"]:
        if os.path.exists(src_dir):
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if not f.endswith(".pyc") and "__pycache__" not in root:
                        p = os.path.join(root, f)
                        z.write(p, arcname=p)
            
    # 2. Result tables & ablations
    for root, _, files in os.walk("results/tables"):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, arcname=p)

    # 3. Explanations (json subgraphs and summary CSVs)
    for root, _, files in os.walk("results/explanations"):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, arcname=p)

    # 4. Figures (PNG and vector PDF)
    for root, _, files in os.walk("results/figures"):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, arcname=p)

    # 5. Saved test probabilities & mapping
    prob_path = "data/processed/final_test_probabilities.npz"
    if os.path.exists(prob_path):
        z.write(prob_path, arcname=prob_path)

    map_path = "data/processed/node_mapping.csv"
    if os.path.exists(map_path):
        z.write(map_path, arcname=map_path)

    # 6. Reports & Presentations
    for root, _, files in os.walk("reports"):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, arcname=p)

print(f"Archive successfully updated at: {zip_path.resolve()}")
print(f"Archive size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
