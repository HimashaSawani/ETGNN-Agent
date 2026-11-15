with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(375, min(410, len(lines))):
    print(f"{i+1:4d}: {repr(lines[i])}")

