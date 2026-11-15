from pathlib import Path
import os

p = Path("C:/Users/himas/Downloads")
try:
    files = list(p.glob("*Phase13*")) + list(p.glob("*Results*")) + list(p.glob("*bundle*"))
    print("Found in Downloads:", files)
except Exception as e:
    print("Error accessing Downloads:", e)

