"""
Script to construct thesis/novelty_revised/build_novelty_revised_thesis.py
Integrating verified novelty and related-work findings into the thesis.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_SCRIPT = ROOT / "thesis" / "phase12_final_integration" / "build_phase12_integrated_thesis.py"
TARGET_SCRIPT = ROOT / "thesis" / "novelty_revised" / "build_novelty_revised_thesis.py"

with open(BASE_SCRIPT, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update output directories
code = code.replace(
    'OUT_DIR = ROOT / "thesis" / "phase12_final_integration"',
    'OUT_DIR = ROOT / "thesis" / "novelty_revised"'
)

# 2. Update docx and pdf output file names
code = code.replace(
    'docx_out = OUT_DIR / "ETGNN_Phase12_Final_Corrected_Thesis.docx"',
    'docx_out = OUT_DIR / "ETGNN_Final_Thesis_Novelty_Revised.docx"'
)
code = code.replace(
    'pdf_out = OUT_DIR / "ETGNN_Phase12_Final_Corrected_Thesis.pdf"',
    'pdf_out = OUT_DIR / "ETGNN_Final_Thesis_Novelty_Revised.pdf"'
)

# Also ensure copies are created at ROOT in main
old_main = """if __name__ == "__main__":
    docx_file = generate_master_docx()
    pdf_file = generate_master_pdf(docx_file)
    for f in [docx_file, pdf_file]:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"SHA256({f.name}) = {h}")"""

new_main = """if __name__ == "__main__":
    import shutil
    docx_file = generate_master_docx()
    pdf_file = generate_master_pdf(docx_file)
    
    # Also copy to ROOT for deliverable accessibility
    root_docx = ROOT / "ETGNN_Final_Thesis_Novelty_Revised.docx"
    root_pdf = ROOT / "ETGNN_Final_Thesis_Novelty_Revised.pdf"
    shutil.copy2(docx_file, root_docx)
    shutil.copy2(pdf_file, root_pdf)
    print(f"[COPIED] Deliverable to ROOT: {root_docx} ({root_docx.stat().st_size:,} bytes)")
    print(f"[COPIED] Deliverable to ROOT: {root_pdf} ({root_pdf.stat().st_size:,} bytes)")

    for f in [docx_file, pdf_file, root_docx, root_pdf]:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"SHA256({f.name}) = {h}")"""

code = code.replace(old_main, new_main)

# 3. Update Title Page Subtitle and Meta
code = code.replace(
    'A Final Year Research Thesis Submitted in Fulfilment of the Honours Degree Requirements\\n'
    'Department of Computer Science & Engineering ? Academic Year 2025/2026',
    'A Final Year Research Thesis (Academic Novelty & Related-Work Revised Edition)\\n'
    'Submitted in Fulfilment of the Honours Degree Requirements\\n'
    'Department of Computer Science & Engineering ? Academic Year 2025/2026'
)

with open(TARGET_SCRIPT, "w", encoding="utf-8") as f:
    f.write(code)

print(f"[SUCCESS] Base code copied and output paths adjusted in {TARGET_SCRIPT}")

