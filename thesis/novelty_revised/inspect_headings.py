import re

with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "add_h1(" in l or "add_h2(" in l:
        print(f"{i+1:4d}: {l.strip()}")

