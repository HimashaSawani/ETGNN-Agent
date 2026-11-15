import zipfile
from pathlib import Path

zips = list(Path(".").glob("*.zip"))
for z in zips:
    size_mb = z.stat().st_size / (1024 * 1024)
    print(f"{z.name:35s}: {size_mb:6.1f} MB")

