with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    text = f.read()

terms = [
    "first-ever", "first ever", "proven superiority", "only viable mitigation", 
    "guaranteed convergence", "provably superior", "provably optimal", "unprecedented",
    "completely novel", "revolutionary", "outperforms all"
]

print("=== UNSUPPORTED CLAIMS CHECK ===")
for t in terms:
    cnt = text.lower().count(t)
    print(f"'{t}': {cnt}")
    if cnt > 0:
        lines = text.split("\n")
        for i, l in enumerate(lines):
            if t in l.lower():
                print(f"   Line {i+1}: {l.strip()[:100]}")

