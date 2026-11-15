from pathlib import Path
import os

p = Path("C:/Users/himas/Downloads")
print("Exists:", p.exists())
try:
    entries = os.listdir(p)
    print(f"Total entries: {len(entries)}")
    for e in entries[:10]:
        print(" ", e)
except Exception as ex:
    print("Exception:", ex)
