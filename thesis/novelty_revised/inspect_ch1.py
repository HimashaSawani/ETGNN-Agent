with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    text = f.read()

import re
m = re.search(r'def build_chapter_1\(doc\):.*?(?=def build_chapter_2)', text, re.DOTALL)
if m:
    print(f"build_chapter_1 length: {len(m.group(0))} chars")

