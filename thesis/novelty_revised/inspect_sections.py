with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("=== ABSTRACT ===")
for i in range(2065, 2115):
    print(lines[i], end="")

print("\n=== CHAPTER 5 (Lines 1030-1065) ===")
for i in range(1030, 1065):
    print(lines[i], end="")

