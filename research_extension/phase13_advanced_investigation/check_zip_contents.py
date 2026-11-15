import zipfile
from pathlib import Path

z_path = Path("ETGNN_Verification_Data_Bundle.zip")
if z_path.exists():
    with zipfile.ZipFile(z_path, "r") as z:
        print(f"Total files in {z_path.name}: {len(z.namelist())}")
        for name in z.namelist()[:15]:
            print("  ", name)
else:
    print("ETGNN_Verification_Data_Bundle.zip not found at root.")

