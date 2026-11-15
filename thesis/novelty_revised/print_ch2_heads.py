with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    text = f.read()

import re
m = re.search(r'def build_chapter_2\(doc\):.*?(?=def build_chapter_3)', text, re.DOTALL)
if m:
    ch2 = m.group(0)
    for line in ch2.split('\n'):
        if 'add_h2(' in line:
            print(line.strip())

