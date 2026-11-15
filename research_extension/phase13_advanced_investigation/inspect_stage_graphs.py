from pathlib import Path

p_scale = Path("src/data/scale_features.py")
p_graphs = Path("src/data/build_stage_graphs.py")

print("=== scale_features.py ===")
if p_scale.exists():
    print(p_scale.read_text(encoding="utf-8")[:1000])

print("\n=== build_stage_graphs.py ===")
if p_graphs.exists():
    print(p_graphs.read_text(encoding="utf-8")[:1000])

