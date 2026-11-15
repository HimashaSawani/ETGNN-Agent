import os
from pathlib import Path

ROOT = Path(".")
src_files = []
for root, dirs, files in os.walk("src"):
    for f in files:
        src_files.append(os.path.join(root, f))

print("=== SRC FILES ===")
for sf in sorted(src_files):
    print(sf)

