import zipfile
from pathlib import Path

zpath = Path("ETGNN_Phase13_Results_Bundle.zip")
with zipfile.ZipFile(zpath, "r") as z:
    infolist = z.infolist()
    print(f"Total entries in {zpath.name}: {len(infolist)}")
    for info in infolist:
        print(f"  {info.filename:70s} | {info.file_size:10,d} bytes")

