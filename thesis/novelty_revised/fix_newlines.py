with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    text = f.read()

# Fix unescaped newlines in string literals
text = text.replace('Specifically:\n"\n        "  (a)', 'Specifically:\\n"\n        "  (a)')
text = text.replace('probability drop?\n"\n        "  (b)', 'probability drop?\\n"\n        "  (b)')
text = text.replace('(FP, FN)?\n"\n        "  (c)', '(FP, FN)?\\n"\n        "  (c)')

with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "w", encoding="utf-8") as f:
    f.write(text)

print("[SUCCESS] Fixed newline escaping in string literals.")

