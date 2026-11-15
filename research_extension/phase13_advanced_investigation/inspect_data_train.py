from pathlib import Path

p_masks = Path("src/data/create_masks.py")
p_train = Path("src/training/train_etgnn.py")

print("=== create_masks.py ===")
if p_masks.exists():
    print(p_masks.read_text(encoding="utf-8")[:1000])

print("\n=== train_etgnn.py ===")
if p_train.exists():
    print(p_train.read_text(encoding="utf-8")[:1000])

