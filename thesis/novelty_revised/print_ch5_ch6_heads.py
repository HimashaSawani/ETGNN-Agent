with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    text = f.read()

import re
print("=== CHAPTER 5 HEADINGS ===")
m5 = re.search(r'def build_chapter_5\(doc\):.*?(?=def build_chapter_6)', text, re.DOTALL)
if m5:
    for line in m5.group(0).split('\n'):
        if 'add_h2(' in line:
            print(line.strip())

print("\n=== CHAPTER 6 HEADINGS ===")
m6 = re.search(r'def build_chapter_6\(doc\):.*?(?=def build_references)', text, re.DOTALL)
if m6:
    for line in m6.group(0).split('\n'):
        if 'add_h2(' in line:
            print(line.strip())

