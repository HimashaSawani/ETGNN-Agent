with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    text = f.read()

import re
m6 = re.search(r'def build_chapter_6\(doc\):.*?(?=def build_references)', text, re.DOTALL)
if m6:
    for l in m6.group(0).split('\n'):
        if 'add_h2(' in l or 'bold_prefix=' in l:
            print(l.strip())

