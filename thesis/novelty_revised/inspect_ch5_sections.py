with open("thesis/novelty_revised/build_novelty_revised_thesis.py", "r", encoding="utf-8") as f:
    text = f.read()

import re
m5_2 = re.search(r'add_h2\(doc, "5\.2.*?(?=add_h2\(doc, "5\.3)', text, re.DOTALL)
if m5_2:
    print("=== Section 5.2 ===")
    print(m5_2.group(0)[:500])

m5_9 = re.search(r'add_h2\(doc, "5\.9.*?(?=add_h2\(doc, "5\.10)', text, re.DOTALL)
if m5_9:
    print("=== Section 5.9 ===")
    print(m5_9.group(0)[:500])

m5_10 = re.search(r'add_h2\(doc, "5\.10.*?(?=add_h2\(doc, "5\.11)', text, re.DOTALL)
if m5_10:
    print("=== Section 5.10 ===")
    print(m5_10.group(0)[:500])

m5_11 = re.search(r'add_h2\(doc, "5\.11.*?(?=def build_chapter_6)', text, re.DOTALL)
if m5_11:
    print("=== Section 5.11 ===")
    print(m5_11.group(0)[:500])

