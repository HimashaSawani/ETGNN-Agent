import zipfile
from pathlib import Path

zpath = Path("ETGNN_Phase13_Results_Bundle.zip")
with zipfile.ZipFile(zpath, "r") as z:
    z.extractall(Path("."))
print("[SUCCESS] Extracted ETGNN_Phase13_Results_Bundle.zip into workspace.")

