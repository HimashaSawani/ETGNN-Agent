import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def markdown_to_docx(md_path, docx_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)

    i = 0
    in_table = False
    table_lines = []

    def flush_table(tbl_lines):
        if not tbl_lines:
            return
        rows_data = []
        for line in tbl_lines:
            line_str = line.strip()
            if not line_str.startswith('|'):
                continue
            cells = [c.strip() for c in line_str.split('|')[1:-1]]
            if all(re.match(r'^:?-+:?$', c) for c in cells):
                continue
            rows_data.append(cells)

        if not rows_data:
            return

        num_cols = max(len(r) for r in rows_data)
        tbl = doc.add_table(rows=len(rows_data), cols=num_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = True

        for r_idx, row in enumerate(rows_data):
            is_header = (r_idx == 0)
            for c_idx in range(num_cols):
                cell_text = row[c_idx] if c_idx < len(row) else ""
                cell = tbl.cell(r_idx, c_idx)
                cell.text = cell_text
                set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                for run in p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(9.5)
                    if is_header:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(15, 23, 42)
                    else:
                        run.font.color.rgb = RGBColor(30, 41, 59)
                if is_header:
                    set_cell_background(cell, 'F1F5F9')
                elif r_idx % 2 == 1:
                    set_cell_background(cell, 'F8FAFC')
        doc.add_paragraph()

    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        if line.startswith('|') and '|' in line[1:]:
            in_table = True
            table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                flush_table(table_lines)
                table_lines = []
                in_table = False

        if not line:
            i += 1
            continue

        if line in ['---', '***', '___']:
            i += 1
            continue

        # Check for major headers
        if line.startswith('CHAPTER ') or line in ['ABSTRACT', 'TABLE OF CONTENTS', 'LIST OF TABLES', 'LIST OF FIGURES', 'REFERENCES', 'APPENDIX']:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(line)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(15, 23, 42)
        elif re.match(r'^\d+\.\d+\s', line) or line.startswith('Chapter Introduction') or line.startswith('Chapter overview'):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(line)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = RGBColor(30, 58, 138) if line.startswith('Chapter') else RGBColor(15, 23, 42)
        elif line.startswith('• ') or line.startswith('- ') or line.startswith('* '):
            text = line[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(text)
            r.font.name = 'Times New Roman'
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(line)
            run.font.name = 'Times New Roman'

        i += 1

    if in_table and table_lines:
        flush_table(table_lines)

    doc.save(docx_path)
    print(f"Generated docx: {docx_path}")

if __name__ == '__main__':
    src_md = r"d:\ETGNN-Fraud-Detection\NIBM_FINAL_INDIVIDUAL_PROJECT_REPORT_SUBMISSION.md"
    dst_docx = r"d:\ETGNN-Fraud-Detection\NIBM_FINAL_INDIVIDUAL_PROJECT_REPORT_SUBMISSION.docx"
    markdown_to_docx(src_md, dst_docx)

