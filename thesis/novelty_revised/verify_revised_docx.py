"""
verify_revised_docx.py
Inspects ETGNN_Final_Thesis_Novelty_Revised.docx for:
- Heading structure
- Table counts and captions
- Figure counts and captions
- Paragraph count and text integrity
- Verification of 24 references
- Verification of unsupported words
"""

import docx
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DOCX_PATH = ROOT / "thesis" / "novelty_revised" / "ETGNN_Final_Thesis_Novelty_Revised.docx"

doc = docx.Document(DOCX_PATH)

print(f"Total paragraphs: {len(doc.paragraphs)}")
print(f"Total tables: {len(doc.tables)}")

headings_h1 = []
headings_h2 = []
tables = []
figures = []

for p in doc.paragraphs:
    txt = p.text.strip()
    if p.style.name == "Heading 1":
        headings_h1.append(txt)
    elif p.style.name == "Heading 2":
        headings_h2.append(txt)
    elif txt.startswith("Figure 5."):
        figures.append(txt[:60])
    elif re.match(r"^Table \d+\.\d+", txt):
        tables.append(txt[:60])

print("\n--- H1 Headings ---")
for h in headings_h1:
    print(f"  {h}")

print(f"\nTotal H2 Headings: {len(headings_h2)}")
for h in headings_h2:
    print(f"  {h}")

print(f"\nTotal Figures detected in text: {len(figures)}")
for f in figures:
    print(f"  {f}")

print(f"\nTotal Tables detected in text: {len(tables)}")
for t in tables:
    print(f"  {t}")

# Check for forbidden claims in the entire document text
full_text = "\n".join([p.text for p in doc.paragraphs] + [c.text for t in doc.tables for row in t.rows for c in row.cells])

forbidden = [
    "first-ever", "proven superiority", "only viable mitigation", 
    "guaranteed convergence", "provably superior", "provably optimal"
]

print("\n--- Forbidden Terms Check ---")
for term in forbidden:
    cnt = full_text.lower().count(term)
    print(f"  '{term}': {cnt}")

# Check references count
ref_count = 0
in_refs = False
for p in doc.paragraphs:
    txt = p.text.strip()
    if txt == "References":
        in_refs = True
        continue
    if in_refs and txt.startswith("Appendix:"):
        in_refs = False
        break
    if in_refs and len(txt) > 20 and ("(" in txt and ")" in txt):
        ref_count += 1

print(f"\nTotal References counted in References section: {ref_count}")

