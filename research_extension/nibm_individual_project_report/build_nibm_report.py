import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from pathlib import Path

out_dir = Path("research_extension/nibm_individual_project_report")
docx_path = out_dir / "NIBM_BSc_AI_Individual_Project_Report.docx"

doc = docx.Document()

# Page setup: Standard Letter, 1 inch margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Institutional styling palette (NIBM Academic Theme)
COLOR_PRIMARY = RGBColor(0x1F, 0x38, 0x64)    # Deep Institutional Navy
COLOR_SECONDARY = RGBColor(0x2B, 0x54, 0x7E)  # Slate Blue
COLOR_DARK = RGBColor(0x22, 0x22, 0x22)       # Off-black / Body text
COLOR_MUTED = RGBColor(0x55, 0x55, 0x55)      # Charcoal Gray

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_header(text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(14 if level==1 else (10 if level==2 else 8))
    h.paragraph_format.space_after = Pt(4)
    run = h.runs[0]
    run.font.name = "Times New Roman"
    if level == 1:
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
    elif level == 2:
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY
    else:
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = COLOR_DARK
    return h

def add_p(text, bold_prefix=None, italic=False, space_after=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.alignment = align
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.font.name = "Times New Roman"
        r_b.font.size = Pt(11)
        r_b.font.bold = True
        r_b.font.color.rgb = COLOR_DARK
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.italic = italic
    r.font.color.rgb = COLOR_DARK
    return p

def add_figure(image_path, caption_text, width=Inches(5.8)):
    p = Path(image_path)
    if p.exists():
        p_img = doc.add_paragraph()
        p_img.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(str(p), width=width)
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = COLOR_MUTED

def add_styled_table(headers, rows, col_widths=None, alignment=WD_TABLE_ALIGNMENT.CENTER):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = alignment
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "1F3864") # Navy
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data Rows
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F2F4F8" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=70, bottom=70, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(9.0)
                r.font.color.rgb = COLOR_DARK

    # Column Widths
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(6)
    return table

# ==============================================================================
# COVER PAGE
# ==============================================================================
p_inst = doc.add_paragraph()
p_inst.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_inst.paragraph_format.space_before = Pt(36)
p_inst.paragraph_format.space_after = Pt(6)
r = p_inst.add_run("KANDY INNOVATION CENTER\nNATIONAL INSTITUTE OF BUSINESS MANAGEMENT")
r.font.name = "Times New Roman"
r.font.size = Pt(15)
r.font.bold = True
r.font.color.rgb = COLOR_PRIMARY

p_prog = doc.add_paragraph()
p_prog.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_prog.paragraph_format.space_after = Pt(36)
r = p_prog.add_run("BACHELOR OF SCIENCE IN COMPUTER SCIENCE WITH AI\n\nINDIVIDUAL PROJECT REPORT")
r.font.name = "Times New Roman"
r.font.size = Pt(13)
r.font.bold = True
r.font.color.rgb = COLOR_SECONDARY

p_title = doc.add_paragraph()
p_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(24)
p_title.paragraph_format.space_after = Pt(36)
r = p_title.add_run("Explainable Temporal Graph Neural Networks for Adaptive Financial Fraud Detection")
r.font.name = "Times New Roman"
r.font.size = Pt(17)
r.font.bold = True
r.font.color.rgb = COLOR_PRIMARY

p_sub = doc.add_paragraph()
p_sub.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_before = Pt(40)
p_sub.paragraph_format.space_after = Pt(12)
r = p_sub.add_run("Submitted by\n")
r.font.name = "Times New Roman"
r.font.size = Pt(11)
r.font.color.rgb = COLOR_MUTED

r_name = p_sub.add_run("P.V.H.S. Vitharama\nRegistration No: [STUDENT-ID-REDACTED]\n")
r_name.font.name = "Times New Roman"
r_name.font.size = Pt(12)
r_name.font.bold = True
r_name.font.color.rgb = COLOR_DARK

p_meta = doc.add_paragraph()
p_meta.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_meta.paragraph_format.space_before = Pt(36)
r = p_meta.add_run("Date of Submission: November 5, 2026")
r.font.name = "Times New Roman"
r.font.size = Pt(11)
r.font.italic = True
r.font.color.rgb = COLOR_MUTED

doc.add_page_break()

# ==============================================================================
# ABSTRACT
# ==============================================================================
add_header("ABSTRACT", level=1)
add_p("Explainable Temporal Graph Neural Networks for Adaptive Financial Fraud Detection", bold_prefix="TITLE: ", space_after=2)
add_p("P.V.H.S. Vitharama | National Institute of Business Management, Kandy Innovation Center", bold_prefix="AUTHOR & AFFILIATION: ", space_after=10)

add_p(
    "Bitcoin illicit-transaction classification is challenging because labels are sparse, illicit cases are uncommon, and transaction patterns change over time. This study evaluates offline time-encoded graph attention, conventional baselines, graph ablations, and post-hoc explanations using the Elliptic Bitcoin benchmark (203,769 nodes, 234,355 directed edges, 165 features and 49 time steps). Chronological partitions are training steps 1–32, validation 33–40, and test 41–49. The model uses 16-dimensional sinusoidal snapshot encoding; it is neither a continuous-time memory network nor an online adaptive system.",
    space_after=8
)
add_p(
    "On the chronological test set, Random Forest achieved AUPRC 0.6323 versus 0.2250 for the canonical ETGNN. In a separately retrained 93-feature ablation, removing edges increased AUPRC from 0.2344 to 0.4812. Earlier five-seed validation experiments reported 0.7758 ± 0.0111 for a residual true graph versus 0.7252 ± 0.0420 for rewired edges; the Holm-adjusted p = 0.1223 was not significant. In a 30-node explanation pilot, GNNExplainer Top-1 edge deletion yielded a mean shift of 0.0150 versus 0.0170 for random deletion (p = 0.6827). Elliptic++ analysis raised concerns about globally aggregated future-information features.",
    space_after=8
)
add_p(
    "Phase 13 tested five neural configurations over seeds 42–46. Test AUPRC mean ± sample SD was 0.312623 ± 0.055945 for residual ETGNN (M1), 0.338011 ± 0.038850 for deep MLP (M2), 0.241915 ± 0.046583 for reliability-gated CE (M3), 0.227318 ± 0.035741 for gated focal M3, and 0.312666 ± 0.050831 for no-gate M4. M1 and M4 were effectively tied. The canonical M0 used interleaved time encoding (AUPRC 0.225030), whereas the Colab evaluation used block encoding (0.255970); these are not matched architecture comparisons. The C2 evaluation included 275 matched TP/FP pairs and reported OR 2.99 for classification flips following edge deletion. These findings concern model sensitivity, not causal fraud attribution.",
    space_after=8
)
add_p(
    "financial fraud detection; Bitcoin; Elliptic; graph attention networks; temporal encoding; explainable artificial intelligence; AUPRC; graph ablation.",
    bold_prefix="Keywords: ", space_after=12
)

doc.add_page_break()

# ==============================================================================
# TABLE OF CONTENTS, LIST OF TABLES, LIST OF FIGURES (PAGINATED EXACTLY)
# ==============================================================================
add_header("TABLE OF CONTENTS", level=1)
toc_items = [
    ("Abstract", "2"),
    ("Table of Contents", "3"),
    ("List of Tables", "4"),
    ("List of Figures", "4"),
    ("Chapter 01: Introduction", "5"),
    ("  1.1 Background of the Study", "5"),
    ("  1.2 Justification of the Problem", "5"),
    ("  1.3 Objectives of the Study", "5"),
    ("Chapter 02: Literature Review", "6"),
    ("  2.1 Financial Crime and Transaction Classification", "6"),
    ("  2.2 Graph and Temporal Learning", "6"),
    ("  2.3 Explainability and Research Gap", "6"),
    ("Chapter 03: Methodology", "7"),
    ("  3.1 Data Collection", "7"),
    ("  3.2 Data Preprocessing", "7"),
    ("  3.3 Model Development and Experimental Design", "8"),
    ("  3.4 Explainability and Statistical Protocol", "10"),
    ("Chapter 04: Results", "11"),
    ("  4.1 Data Preprocessing", "11"),
    ("  4.2 Exploratory Data Analysis", "11"),
    ("  4.3 Project Outcomes and Classification Results", "12"),
    ("  4.4 Graph Ablations, Statistical Comparisons and Explainability", "13"),
    ("  4.5 Elliptic++ Cross-Representation Evaluation", "15"),
    ("Chapter 05: Discussion", "16"),
    ("  5.1 Interpretation of Predictive Performance", "16"),
    ("  5.2 Interpretation of Temporal Modelling", "16"),
    ("  5.3 Interpretation of Explainability", "17"),
    ("  5.4 Research Contributions", "17"),
    ("  5.5 Limitations and Threats to Validity", "17"),
    ("Chapter 06: Conclusion", "18"),
    ("References", "19"),
    ("Appendix", "20"),
    ("  Appendix A: Experimental Reproducibility and Software Verification", "20"),
    ("  Appendix B: Experimental Provenance and Baseline Reconciliation", "20"),
    ("  Appendix C: Five-Seed Neural Model Performance", "20"),
    ("  Appendix D: Matched C2 Edge-Intervention Results", "21"),
]

for title, page in toc_items:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.05
    p.paragraph_format.space_after = Pt(2)
    r1 = p.add_run(title)
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(10)
    if "Chapter" in title or "References" in title or "Appendix" in title or "Abstract" in title:
        r1.font.bold = True
    r2 = p.add_run(f" {' .' * (42 - len(title)//2)} {page}")
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = COLOR_MUTED

doc.add_page_break()

add_header("LIST OF TABLES", level=1)
lot_items = [
    ("Table 3.1: Dataset and experimental scope", "7"),
    ("Table 3.2: Model and graph control conditions", "8"),
    ("Table 4.1: Reported main Elliptic held-out test average precision (steps 41–49)", "12"),
    ("Table 4.2: Ablation and Phase 4 results; note the different model families and data splits", "13"),
    ("Table 4.3: Phase H explanation sensitivity summary (30 multi-edge validation nodes)", "14"),
    ("Table 4.4: Selected Elliptic++ cross-representation findings", "15"),
    ("Table C.1: Five-seed held-out test AUPRC by neural configuration", "20"),
    ("Table D.1: Matched C2 one-hop edge-deletion classification outcomes", "21"),
]
for title, page in lot_items:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.add_run(title).font.name = "Times New Roman"
    p.runs[0].font.size = Pt(10)
    r = p.add_run(f" {' .' * (38 - len(title)//2)} {page}")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.5)
    r.font.color.rgb = COLOR_MUTED

p_gap = doc.add_paragraph()
p_gap.paragraph_format.space_before = Pt(14)

add_header("LIST OF FIGURES", level=1)
lof_items = [
    ("Figure 3.1: Research evaluation pipeline (conceptual overview)", "9"),
    ("Figure 4.1: Reported held-out test AUPRC for primary Elliptic benchmark models", "12"),
    ("Figure 4.2: Residual model validation AUPRC (Holm-adjusted p = 0.1223)", "14"),
]
for title, page in lof_items:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.add_run(title).font.name = "Times New Roman"
    p.runs[0].font.size = Pt(10)
    r = p.add_run(f" {' .' * (36 - len(title)//2)} {page}")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.5)
    r.font.color.rgb = COLOR_MUTED

doc.add_page_break()

# ==============================================================================
# CHAPTER 01: INTRODUCTION
# ==============================================================================
add_header("Chapter 01: Introduction", level=1)

add_header("1.1 Background of the Study", level=2)
add_p(
    "Financial crime monitoring seeks to identify suspicious transactions without generating an unmanageable volume of false alerts. However, the presence of a graph does not guarantee that relational aggregation improves classification: the usefulness of edges depends on graph construction, label patterns, feature engineering and the evaluation protocol (Weber et al., 2019).",
    space_after=6
)
add_p(
    "The Elliptic Bitcoin benchmark is particularly relevant because it provides transaction attributes, graph relationships, licit and illicit labels for a subset of nodes, and 49 ordered time steps. This is more realistic than random train/test partitioning for systems intended to evaluate future activity.",
    space_after=6
)
add_p(
    "This project evaluates a time-encoded graph-attention classifier and several baseline models. The implemented ETGNN uses sinusoidal time-step information concatenated to transaction features before graph-attention processing.",
    space_after=8
)

add_header("1.2 Justification of the Problem", level=2)
add_p(
    "Three connected problems motivate the work: severe class imbalance, temporal distribution changes, and uncertain benefits of graph aggregation. These factors complicate reliable detection of illicit Bitcoin transactions.",
    space_after=6
)
add_p(
    "Research questions address chronological baseline performance (RQ1), graph and feature ablations (RQ2), explanation sensitivity (RQ3), statistical robustness (RQ4), and Elliptic++ transfer limitations (RQ5).",
    space_after=8
)

add_header("1.3 Objectives of the Study", level=2)
add_p("Implement and evaluate a snapshot time-encoded GAT classifier using a chronological train/validation/test protocol.", bold_prefix="• ")
add_p("Compare the evaluated ETGNN with Random Forest, XGBoost, GCN, GraphSAGE, GAT and controlled feature-only baselines.", bold_prefix="• ")
add_p("Quantify effects of true, empty and degree-preserving rewired graphs under matched experimental conditions.", bold_prefix="• ")
add_p("Evaluate post-hoc explanation sensitivity using GNNExplainer, random edge deletion and exhaustive single-edge deletion within bounded candidate sets.", bold_prefix="• ")
add_p("Report uncertainty, multiplicity corrections, data-provenance limitations and reproducibility evidence transparently.", bold_prefix="• ")
add_p(
    "Scope: the study concerns offline supervised transaction classification on benchmark data and a demonstration interface. It does not establish real-time fraud prevention, online adaptation, causal responsibility of actors or deployment-grade regulatory compliance.",
    italic=True, space_after=12
)

doc.add_page_break()

# ==============================================================================
# CHAPTER 02: LITERATURE REVIEW
# ==============================================================================
add_header("Chapter 02: Literature Review", level=1)

add_header("2.1 Financial Crime and Transaction Classification", level=2)
add_p(
    "Weber et al. (2019) introduced a benchmark-oriented approach to anti-money laundering in Bitcoin and evaluated graph convolutional models using transaction relationships.",
    space_after=6
)
add_p(
    "Fraud classification is not equivalent to ordinary balanced binary classification. Precision, recall, F1, ROC-AUC and especially average precision are therefore relevant, with AUPRC requiring careful interpretation when illicit prevalence changes across time.",
    space_after=8
)

add_header("2.2 Graph and Temporal Learning", level=2)
add_p(
    "GCN (Kipf & Welling, 2017) aggregates graph-local information through normalized convolution. GraphSAGE (Hamilton et al., 2017) learns neighbourhood aggregation functions, and GAT (Veličković et al., 2018) uses attention coefficients to weight neighbouring messages.",
    space_after=6
)
add_p(
    "Temporal graph research includes EvolveGCN (Pareja et al., 2020), which updates model parameters over graph snapshots, and Temporal Graph Networks (Rossi et al., 2020), which use temporal memory for event streams. Time2Vec (Kazemi et al., 2019) demonstrates learned time representations.",
    space_after=8
)

add_header("2.3 Explainability and Research Gap", level=2)
add_p(
    "GNNExplainer (Ying et al., 2019) optimizes a mask to identify graph structures and features associated with a prediction. GraphFramEx (Amara et al., 2022) highlights the need for systematic explanation evaluation.",
    space_after=6
)
add_p(
    "The research gap addressed here is not the absence of graph attention, time encoding or edge-deletion explanations. This is a context-specific methodological contribution; a claim of global first priority would require further exhaustive literature verification.",
    space_after=12
)

doc.add_page_break()

# ==============================================================================
# CHAPTER 03: METHODOLOGY
# ==============================================================================
add_header("Chapter 03: Methodology", level=1)

add_header("3.1 Data Collection", level=2)
add_p(
    "The primary dataset is the Elliptic Bitcoin transaction benchmark. Nodes correspond to transactions rather than people or wallets, and directed edges represent transaction-to-transaction payment-flow relationships. The graph contains 203,769 nodes and 234,355 edges over 49 time steps. Each node has 165 provided features, commonly distinguished as 93 local transaction features and 72 aggregated features. Labels are licit, illicit or unknown; only labelled nodes are used for supervised classification targets, while unlabelled nodes may still be relevant to graph topology depending on the experiment.",
    space_after=4
)

headers_31 = ["Property", "Primary Elliptic benchmark", "Methodological relevance"]
rows_31 = [
    ["Nodes", "203,769", "Transaction-level classification"],
    ["Edges", "234,355", "Directed payment-flow relations"],
    ["Time steps", "49", "Chronological evaluation"],
    ["Features", "165 = 93 local + 72 aggregate", "Feature redundancy controls"],
    ["Labels", "Licit / illicit / unknown", "Partial supervision"],
    ["Split", "Train 1–32; val 33–40; test 41–49", "Future-period generalization"]
]
add_styled_table(headers_31, rows_31, [1.5, 2.5, 2.5])
add_p("Table 3.1: Dataset and experimental scope.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)

add_p(
    "A secondary Elliptic++ analysis used a wallet/actor-oriented representation related to the same underlying Bitcoin activity. Some wallet lifetime features summarize future activity and are therefore unsuitable for strong prediction-time leakage-free claims without reconstruction from past-only information.",
    space_after=8
)

add_header("3.2 Data Preprocessing", level=2)
add_p(
    "The evaluation maintains chronological separation between development and held-out periods. Feature preparation and any normalization should be fitted using training-period data only. Model variants using 93 local features are separated explicitly from those using all 165 features.",
    space_after=6
)
add_p(
    "The use of precomputed aggregate features introduces an important interpretive caveat: a model trained with 165 features may already receive neighbourhood summaries before GNN aggregation. The experiments observe performance differences, while the underlying mechanisms remain hypotheses requiring separate isolation.",
    space_after=8
)

add_header("3.3 Model Development and Experimental Design", level=2)
add_p(
    "Phase 13 evaluated five neural configurations on test steps 41–49 with seeds 42–46: M1 residual ETGNN, M2 deep MLP, M3 reliability-gated ETGNN under cross-entropy and focal losses, and M4 no-gate control. Their test results are distinct from earlier Phase 4 validation comparisons. The original ETGNN remains a two-layer graph-attention network with 16-dimensional sinusoidal time-step features. Training was offline, with frozen checkpoints for evaluation.",
    space_after=6
)
add_p(
    "Residual ETGNN provides an ego-feature projection path alongside graph-neighbour aggregation. The no-gate M4 comparison tests the incremental contribution of a particular learned gate within the corresponding implementation. Nearly identical M1 and M4 average precision does not prove the sole cause of any gain, nor establish equivalence without an adequately powered paired equivalence design.",
    space_after=6
)
add_p(
    "The reliability-gated variant introduces a learned multiplier for relational messages. Its gate values were inspected after training to diagnose saturation and weak message transmission.",
    space_after=6
)
add_p(
    "Temporal encoding reproducibility. The frozen historical M0 checkpoint uses an interleaved sinusoidal representation created with torch.stack(...).flatten(1), while the Colab re-evaluation used a block-concatenated sine/cosine representation created with torch.cat(...). Both have sixteen temporal values but differ in feature order. The original frozen checkpoint scored AUPRC 0.225030 and ROC-AUC 0.806248 using its canonical interleaved input; its Colab block-encoded evaluation scored AUPRC 0.255970 and ROC-AUC 0.813184. Because this is an input-representation discrepancy, the historical checkpoint is not a clean matched control for attributing Phase 13 gains to architecture alone.",
    space_after=6
)
add_p(
    "The principal ETGNN model uses a two-layer Graph Attention Network with a 16-dimensional sinusoidal representation of the discrete time-step index concatenated to node features. Training is offline, and evaluation uses frozen weights.",
    space_after=6
)
add_p(
    "Baseline models include Random Forest, XGBoost, GCN, GraphSAGE, static GAT, a zero-time GAT control and MLP variants. Empty-edge retraining and post-training edge deletion answer different questions: the former measures the predictive capacity of a separately optimized model without message passing, while the latter measures the sensitivity of an already trained model to an intervention.",
    space_after=4
)

headers_32 = ["Condition", "Input/graph treatment", "Interpretation"]
rows_32 = [
    ["Original ETGNN", "Time-encoded two-layer GAT", "Primary proposed configuration"],
    ["Zero-time GAT", "Matched architecture without time signal", "Time-encoding control"],
    ["Residual true graph", "Ego-residual pathway, observed edges", "Controlled residual configuration"],
    ["Residual empty graph", "Same model family, no graph edges", "Topology ablation"],
    ["Residual rewired graph", "Degree-preserving rewired edges", "Graph-structure null control"],
    ["Deep MLP", "Node features only", "Feature-only comparator"]
]
add_styled_table(headers_32, rows_32, [1.8, 2.5, 2.2])
add_p("Table 3.2: Model and graph control conditions.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)

add_p(
    "Model evaluation uses precision, recall, F1, ROC-AUC and AUPRC where reported. Since the illicit class is relatively uncommon and its prevalence varies over time, AUPRC is emphasized.",
    space_after=8
)

# Insert Figure 3.1
add_figure(
    "research_extension/phase14b_final_completion/phase14b_complete_benchmark_comparison.png",
    "Figure 3.1: Research evaluation pipeline (conceptual overview)."
)

add_header("3.4 Explainability and Statistical Protocol", level=2)
add_p(
    "Extended C1 and C2 analyses. The final extension adds a feature-versus-edge perturbation comparison (C1) and a separate matched error-cohort analysis (C2). These should not be conflated with the earlier 30-node Phase H GNNExplainer-versus-random pilot, which tests a different explanation method and deletion budget. In C1, a frozen predictor is evaluated before and after a specified feature mask or edge intervention. The change in predicted illicit probability is a sensitivity measurement conditional on the perturbation, the retained feature aggregates, and the fixed model.",
    space_after=6
)
add_p(
    "For C2, 275 false-positive and 275 true-positive validation transactions were paired using the reported matching criteria, including timestep, prediction score and available in-degree. The reported selected cohort is concentrated at in-degree one. Each transaction was reevaluated after the defined one-hop edge deletion; a flip means its score crossed the classification threshold. The comparison estimates an association between cohort membership and model response under this intervention. It does not identify a fraudulent actor, show that removing a real payment would prevent fraud, or justify automatically dismissing a compliance alert.",
    space_after=6
)
add_p(
    "Five-seed sample SD describes run-to-run variation; SE is SD divided by √5. Confidence intervals are based on only five runs and require cautious interpretation. The historical M0 has one checkpoint and uses a different time-encoding arrangement from Phase 13, preventing a rigorously paired architecture comparison. Matched graph observations are not necessarily independent; multiplicity-adjusted results should be distinguished from exploratory p-values.",
    space_after=6
)
add_p(
    "GNNExplainer provides candidate subgraph attributions for selected predictions. In the C1 protocol, matched candidate-edge interventions are compared between ETGNN and a zero-time graph-attention control. In C2, explanations are contrasted across prediction-outcome cohorts using matching criteria such as prediction margin and edge-count availability.",
    space_after=6
)
add_p(
    "A further explanation benchmark evaluates GNNExplainer Top-1 deletion, random single-edge deletion and exhaustive Leave-One-Out (LOO) candidate search. It is therefore a useful intervention reference but is advantaged by its search objective; it should not be presented as an independent causal oracle or a universally superior explanation method.",
    space_after=6
)
add_p(
    "Where experiments use repeated seeds, paired contrasts and Holm-adjusted p-values are reported to control family-wise error across the specified comparison family. A non-significant result is not evidence of equality.",
    space_after=12
)

doc.add_page_break()

# ==============================================================================
# CHAPTER 04: RESULTS
# ==============================================================================
add_header("Chapter 04: Results", level=1)

add_header("4.1 Data Preprocessing", level=2)
add_p(
    "The project reports a reproducible chronological benchmark using frozen result artifacts and documented experiment matrices. The core distinction is between the full 165-feature setting and local-only 93-feature conditions. The later tests assess performance on time steps 41–49, while the Phase 4 factorial experiments summarize validation results from time steps 33–40.",
    space_after=8
)

add_header("4.2 Exploratory Data Analysis", level=2)
add_p(
    "Topology audit and limits of attribution. Repository audit reports cite TOPOLOGY_AWARE_EXPLANATION_REPORT.md and EXP_A_CANDIDATE_EDGE_SHIFT_CENSUS.csv for an approximately 84.08% decline in the measured average local clustering coefficient around snapshot 43, alongside reported mean-degree values of 4.82 and 1.18 and zero-edge shares of 13.52% and 76.92% across compared regimes. These figures describe the reported graph-census comparison, not a universal property of all Elliptic nodes. The underlying CSV, denominator definitions and timestamp mapping should be retained as traceable evidence for examination.",
    space_after=6
)
add_p(
    "A temporal coincidence with a reported May 2019 darknet-market law-enforcement event is not evidence that the event caused the graph shift or the model performance difference. The dataset period and snapshot-to-calendar mapping must be verified before associating snapshot 43 with a specific real-world event.",
    space_after=6
)
add_p(
    "The graph contains many unlabelled transactions and an illicit minority among labelled observations. This structure supports precision–recall-oriented evaluation. Temporal distribution changes are especially relevant in the secondary Elliptic++ analysis: the reported illicit prevalence decreased from 10.02% in validation to 5.45% in the test window, a relative decline of approximately 45.6%.",
    space_after=8
)

add_header("4.3 Project Outcomes and Classification Results", level=2)
add_p(
    "Expanded Phase 13 test benchmark (separate from the historical checkpoint and Phase 4 validation results). The five-seed mean test AUPRC was 0.338011 for M2 Deep MLP, 0.312666 for M4 NoGate, 0.312623 for M1 Residual ETGNN, 0.241915 for M3 Reliability-Gated ETGNN trained with cross-entropy, and 0.227318 for M3 trained with focal loss. The strongest tested tree ensemble remained Random Forest at AUPRC 0.632263, followed by XGBoost at 0.547215. These results do not support a claim that the new residual GNN exceeds the best tabular baseline.",
    space_after=6
)
add_p(
    "Five-seed uncertainty reporting. For M1 Residual ETGNN, mean test AUPRC is 0.312623, sample SD 0.055945, SE 0.025020 and the reported t-based 95% confidence interval [0.243158, 0.382089]. For M2 Deep MLP, mean test AUPRC is 0.338011 with sample SD 0.038850 (not 0.0024). M4 NoGate has sample SD 0.050831; M3 cross-entropy and focal-loss variants have sample SD 0.046583 and 0.035741 respectively. All are five-seed results (seeds 42–46), and SD must not be confused with SE or an uncertainty measure for a different metric.",
    space_after=6
)
add_p(
    "M1 exceeded the canonical historical M0 by 0.087593 AUPRC, but the one-sample comparison p ≈ 0.0249 is not evidence of paired architectural superiority: M0 has one checkpoint and a different time encoding. Against block-encoded M0 (0.255970), the reported one-sample p ≈ 0.0863. M1 and no-gate M4 were virtually tied. All comparisons must preserve the model, feature, split and encoding identifiers.",
    space_after=4
)

headers_41 = ["Model", "Test AUPRC", "Dataset / setting"]
rows_41 = [
    ["Random Forest", "0.6323", "Elliptic; 165 features"],
    ["XGBoost", "0.5472", "Elliptic; 165 features"],
    ["GraphSAGE", "0.2427", "Elliptic; reported baseline"],
    ["ETGNN", "0.2250", "Elliptic; 165 features"],
    ["Zero-time GAT", "0.2099", "Elliptic; matched control"],
    ["GCN", "0.1306", "Elliptic; baseline"],
    ["Static GAT", "0.1155", "Elliptic; baseline"]
]
add_styled_table(headers_41, rows_41, [2.0, 1.5, 3.0])
add_p("Table 4.1: Reported main Elliptic held-out test average precision (steps 41–49).", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)

add_p(
    "Random Forest was the strongest model in this reported comparison. This negative result is an important research finding because it motivates examination of graph inductive bias and feature composition rather than selective reporting of favourable model outcomes.",
    space_after=6
)

# Insert Figure 4.1
add_figure(
    "research_extension/phase14_results_analysis/phase14_model_auprc_comparison.png",
    "Figure 4.1: Reported held-out test AUPRC for primary Elliptic benchmark models."
)

add_header("4.4 Graph Ablations, Statistical Comparisons and Explainability", level=2)
add_p(
    "Phase 13 gate diagnostic. In the supplied final research narrative, reliability-gated M3 reached AUPRC 0.241915 under cross-entropy and 0.227318 under focal loss. Inspections of selected trained checkpoints reported layer-two gate values close to zero in several runs, while the no-gate M4 condition attained AUPRC 0.312666. This is a useful negative result for the tested configuration. However, gate collapse frequency and coefficient distributions should be reproduced from checkpoint inspection before quoting a universal numerical range, and the performance contrast alone does not identify a unique mechanism.",
    space_after=6
)
add_p(
    "Final C1 sensitivity result. The later C1 summary reports a mean illicit-probability change of approximately +0.1250 after the specified feature masking and a much smaller edge-deletion response of approximately 0.0091 in magnitude. This is consistent with the evaluated predictor being more sensitive to the tested feature perturbation than to the tested graph perturbation. It is not proof that relational information is intrinsically useless: the provided 165 features include 72 precomputed neighbourhood aggregates, and the masking/deletion interventions are not necessarily comparable in magnitude or distributional plausibility.",
    space_after=6
)
add_p(
    "C2 matched-cohort edge-intervention sensitivity. Among 275 selected false positives, 33 (12.00%) changed to a licit prediction following the specified one-hop edge ablation, compared with 12 of 275 true positives (4.36%). The reported odds ratio is 2.99 (95% CI [1.51, 5.92]); Fisher exact p = 0.0016 and Holm-adjusted p = 0.0032. This is a difference in model response within a matched validation cohort, not proof that graph contamination caused the false positives, and not a validated operational rule for dismissing fraud alerts. The selected cohort and intervention definition limit generalization.",
    space_after=6
)
add_p(
    "Relationship to earlier explanations. The earlier 30-node pilot found mean Top-1 GNNExplainer deletion response 0.0150 versus 0.0170 for random single-edge deletion (reported p = 0.6827). The later 275-pair C2 analysis tests an error-cohort contrast, not a direct replication of the pilot.",
    space_after=6
)
add_p(
    "For a local-feature non-residual ETGNN comparison, a model separately retrained with no edges achieved test AUPRC 0.4812, compared with 0.2344 for the reported graph-based counterpart. This +0.2468 difference suggests that the evaluated edge aggregation was not beneficial in that particular configuration.",
    space_after=4
)

headers_42 = ["Comparison", "Reported AUPRC", "Inference"]
rows_42 = [
    ["Local-feature ETGNN, graph", "0.2344 test", "Non-residual configuration"],
    ["Local-feature ETGNN, retrained without edges", "0.4812 test", "Separate training condition"],
    ["Residual true graph, 93f", "0.7758 ± 0.0111 validation", "Five-seed mean ± SD"],
    ["Residual rewired graph, 93f", "0.7252 ± 0.0420 validation", "Holm p = 0.1223 vs true graph"],
    ["Deep MLP, 165f", "0.7189 ± 0.0016 validation", "Feature-only comparator"],
    ["Residual true graph, 165f", "0.5812 ± 0.0639 validation", "MLP contrast Holm p = 0.0498"]
]
add_styled_table(headers_42, rows_42, [2.3, 1.9, 2.3])
add_p("Table 4.2: Ablation and Phase 4 results; note the different model families and data splits.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)

add_p(
    "The 93-feature residual true-graph mean exceeded the rewired-graph mean by 0.0506, with an unadjusted paired p-value of 0.0245. After Holm correction over the six reported primary contrasts, p = 0.1223, so this comparison did not meet a 0.05 family-wise significance threshold. The result is therefore exploratory rather than confirmation of topological superiority. On 165 features, the deep MLP performed better than the tested residual GNN in the reported validation comparison (Holm p = 0.0498); this establishes a condition-specific performance difference, not a uniquely demonstrated feature-redundancy mechanism.",
    space_after=6
)

# Insert Figure 4.2
add_figure(
    "research_extension/phase14_results_analysis/phase14_validation_vs_test_scatter.png",
    "Figure 4.2: Residual model validation AUPRC (Holm-adjusted p = 0.1223)."
)

headers_43 = ["Intervention", "Mean probability shift", "Measured runtime"]
rows_43 = [
    ["GNNExplainer Top-1 edge deletion", "0.0150", "260.47 ms"],
    ["Random-1 edge deletion", "0.0170", "1.10 ms"],
    ["LOO maximum single-edge deletion", "0.0547", "13.69 ms"],
    ["All 2-hop computation edges omitted", "0.0773", "Not compared here"]
]
add_styled_table(headers_43, rows_43, [2.8, 1.8, 1.9])
add_p("Table 4.3: Phase H explanation sensitivity summary (30 selected multi-edge validation nodes).", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)

add_p(
    "The GNNExplainer versus random single-edge deletion comparison yielded p = 0.6827, indicating that the evaluated test did not detect a significant difference; this is not proof of equivalence. LOO identified a larger mean response by searching candidate deletions for the maximum effect, so the numerical advantage is conditional on the selection procedure. The mean 0.0773 response under complete neighbourhood omission is an empirical observation, not a mathematical robustness bound.",
    space_after=8
)

add_header("4.5 Elliptic++ Cross-Representation Evaluation", level=2)
headers_44 = ["Condition", "Test AUPRC (mean ± SD)", "Interpretation"]
rows_44 = [
    ["Deep feature-only MLP", "0.0991 ± 0.0268", "17,730 parameters"],
    ["Residual ETGNN, true graph", "0.0968 ± 0.0214", "9,736 parameters"],
    ["Validation illicit prevalence", "10.02%", "Earlier window"],
    ["Test illicit prevalence", "5.45%", "Later window"]
]
add_styled_table(headers_44, rows_44, [2.3, 2.0, 2.2])
add_p("Table 4.4: Selected Elliptic++ cross-representation findings.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)

add_p(
    "The Elliptic++ test results were much lower than validation values across evaluated configurations. The MLP and residual GNN were not parameter-matched (17,730 versus 9,736 parameters). In addition, the reported wallet-level features last_block_appeared_in, lifetime_in_blocks and total_txs may incorporate activity occurring after the target prediction time. These features create a prediction-time leakage concern; evaluation of previously unseen actors does not eliminate that concern. The secondary benchmark is therefore informative for sensitivity and provenance analysis but should not be presented as an independent leakage-free external validation.",
    space_after=12
)

doc.add_page_break()

# ==============================================================================
# CHAPTER 05: DISCUSSION
# ==============================================================================
add_header("Chapter 05: Discussion", level=1)

add_header("5.1 Interpretation of Predictive Performance", level=2)
add_p(
    "Integrated interpretation of later test experiments. The strongest result on the specified held-out test split is the Random Forest baseline, and the feature-only Deep MLP also exceeds all Phase 13 GNN variants by mean AUPRC.",
    space_after=6
)
add_p(
    "The apparent tension between the 93-feature validation experiments, the 165-feature historical test, and the later 166-dimensional neural configurations is not a contradiction when the distinct experimental conditions are retained. Each result addresses a different question.",
    space_after=6
)
add_p(
    "The principal finding is that adding time-encoded graph attention did not improve average precision over the strongest conventional baseline on the main held-out Elliptic evaluation. The Random Forest result of 0.6323 versus ETGNN 0.2250 is a direct benchmark-specific comparison, not a universal statement about the value of GNNs.",
    space_after=6
)
add_p(
    "The contrast between non-residual edge removal and the residual factorial study further cautions against a single explanation of graph behaviour. They motivate future studies that isolate these factors.",
    space_after=8
)

add_header("5.2 Interpretation of Temporal Modelling", level=2)
add_p(
    "The time encoding introduces a discrete time-step signal but does not constitute continuous-time event modelling, online learning or automatic adaptation. The chronological split evaluates generalization across time, while adaptation to new data would require additional mechanisms and experiments beyond the current implementation.",
    space_after=8
)

add_header("5.3 Interpretation of Explainability", level=2)
add_p(
    "The final matched-pair analysis adds an error-diagnostic perspective beyond visual explanations: under the tested deletion, false positives were more likely than true positives to cross the classification threshold. The result is best presented as a research-stage sensitivity signal.",
    space_after=6
)
add_p(
    "Explanation methods are most useful when their outputs are compared with meaningful baselines and subjected to consistent perturbations. The C1 shared-edge comparisons and C2 matched error-cohort analysis provide a structured way to ask whether model explanations differ across architectures or prediction outcomes.",
    space_after=6
)
add_p(
    "The 30-node Phase H analysis demonstrates the value of explicit reference methods. Exhaustive candidate search produced a larger mean effect but is optimized to do so.",
    space_after=8
)

add_header("5.4 Research Contributions", level=2)
add_p("A reproducible, chronological comparison of time-encoded graph attention against conventional and graph-based fraud-classification baselines.", bold_prefix="• ")
add_p("Controlled graph and feature ablations, including retrained empty-graph and degree-preserving rewired-graph conditions, with multiplicity-aware statistical reporting.", bold_prefix="• ")
add_p("A matched explanation evaluation design combining between-model interventions, prediction-error cohorts and random/LOO deletion references.", bold_prefix="• ")
add_p("A documented investigation of Elliptic++ provenance, overlap and potential future-information leakage in global lifetime features.", bold_prefix="• ")
add_p("A FastAPI/React demonstration interface and preserved experimental artifacts, subject to final live-demo verification.", bold_prefix="• ")
add_p(
    "These contributions are primarily empirical and methodological. They do not establish the invention of GAT, sinusoidal encoding, GNNExplainer or LOO. Novelty priority relative to all worldwide prior work has not been independently established.",
    italic=True, space_after=8
)

add_header("5.5 Limitations and Threats to Validity", level=2)
add_p(
    "Further limitations include the mismatch between canonical and Colab M0 time encodings, five-seed uncertainty, non-independence of matched graph observations, restricted explanation candidate sets, and uncertainty about the provenance of secondary-benchmark features. Reported topology changes near snapshot 43 do not establish their external cause or prove why graph message passing underperformed.",
    space_after=6
)
add_p(
    "Conclusions are limited to these datasets, feature configurations and chronological splits. The study does not implement continuous-time memory, online adaptation, a verified production deployment or regulatory-grade fraud attribution. Explanation perturbations measure model response, not real-world causality. The Elliptic++ comparison also has model-capacity differences and potential future-information leakage.",
    space_after=6
)
add_p(
    "Sparse-neighbourhood deletion-budget limitation. In the reported explanation cohort, 317 of 377 transactions (84.1%) had exactly two eligible candidate edges. Under the nominal 25% deletion rule k(v) = max(1, ceil(0.25 × E_cand(v))), deleting one of two edges corresponds to an actual 50% deletion fraction. The reported mean actual deletion fraction was approximately 47.08%, not 25%. Consequently, explanation-guided and random edge-deletion responses should be interpreted as sensitivity under relatively high deletion fractions in small candidate neighbourhoods, rather than as uniform 25% perturbations. These cohort statistics should be reconciled with the final experiment artifacts before submission.",
    space_after=6
)
add_p(
    "Results are based on the recorded experimental artifacts and should remain traceable to their frozen source files.",
    space_after=12
)

doc.add_page_break()

# ==============================================================================
# CHAPTER 06: CONCLUSION
# ==============================================================================
add_header("Chapter 06: Conclusion", level=1)
add_p(
    "This project evaluated offline snapshot time-encoded graph attention for Bitcoin illicit-transaction classification using chronological testing, controlled graph conditions and explanation sensitivity analysis. Random Forest (test AUPRC 0.6323) outperformed canonical ETGNN (0.2250). Later five-seed experiments showed that residual GNN variants improved on some neural baselines but did not surpass the strongest tabular tree models. M1 and no-gate M4 were nearly tied; the available evidence does not isolate one improvement mechanism.",
    space_after=6
)
add_p(
    "Post-hoc explanations require random and exhaustive deletion controls. The early GNNExplainer–random difference was not significant, and the later matched C2 intervention measured differential model sensitivity rather than fraud causality. Elliptic++ benchmark overlap and possible future-feature leakage limit external validation claims. The contribution is careful comparative methodology, negative findings and transparent limitations, not a new attention primitive or deployed online adaptation.",
    space_after=6
)
add_p(
    "This study contributes a chronological evaluation of time-encoded graph attention, five-seed neural benchmarks, controlled graph ablations and post-hoc explanation sensitivity analyses for Bitcoin transaction classification. The findings motivate future work on leakage-controlled temporal features, matched architecture comparisons, more diverse graph datasets and independent replication of explanation results.",
    space_after=12
)

doc.add_page_break()

# ==============================================================================
# REFERENCES
# ==============================================================================
add_header("References", level=1)
refs = [
    "Amara, K., Ying, Z., Zhang, Z., Han, Z., Zhao, Y., Shan, Y., Brandes, U., Schemm, S., & Zhang, C. (2022). Proceedings of the First Learning on Graphs Conference, PMLR 198, 44:1–44:23.",
    "Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. Proceedings of the 29th ACM International Conference on Information and Knowledge Management, 315–324.",
    "Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. Advances in Neural Information Processing Systems, 30.",
    "Kazemi, S. M., Goel, R., Eghbali, S., Ramanan, J., Sahota, J., Thakur, S., Wu, S., Smyth, C., Poupart, P., & Brubaker, M. A. (2019). Time2Vec: Learning a vector representation of time. arXiv preprint.",
    "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. International Conference on Learning Representations.",
    "Liu, Y., Ao, X., Qin, Z., Chi, J., Feng, J., Yang, H., & He, Q. (2021). Pick and choose: A GNN-based imbalanced learning approach for fraud detection. Proceedings of the Web Conference 2021, 3168–3177.",
    "Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T., & Leiserson, C. E. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. Proceedings of the AAAI Conference on Artificial Intelligence, 34(4), 5363–5370.",
    "Rao, S. X., Zhang, S., Han, Z., Zhang, Z., Min, W., Chen, Z., Shan, Y., Zhao, Y., & Zhang, C. (2021). xFraud: Explainable fraud transaction detection. Proceedings of the VLDB Endowment, 15(3), 427–436.",
    "Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. M. (2020). Temporal graph networks for deep learning on dynamic graphs. arXiv preprint.",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. International Conference on Learning Representations.",
    "Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. arXiv preprint.",
    "Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating explanations for graph neural networks. Advances in Neural Information Processing Systems, 32."
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.5)
    r = p.add_run(ref)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK

doc.add_page_break()

# ==============================================================================
# APPENDIX
# ==============================================================================
add_header("Appendix", level=1)

# Appendix A
add_header("Appendix A: Experimental Reproducibility and Software Verification", level=2)
add_p(
    "The recorded QA execution on 5 November 2026 collected 27 automated tests, with 27 passed, 0 failed and 0 skipped (pytest exit code 0; 38.74 seconds) in a Windows environment supporting localhost connections. A separate network-isolated execution skipped 20 HTTP transport tests because loopback connections were restricted. These are recorded software verification results rather than independent re-execution of the experimental models.",
    space_after=6
)
add_p(
    "The reproducibility record comprises experiment configuration matrices, result CSV files, fixed random seeds, saved model checkpoints, explanation summaries and API test logs. Chronological partitions use training snapshots 1–32, validation 33–40 and held-out test 41–49. Model results should be interpreted with their original feature sets, time encodings and evaluation splits.",
    space_after=8
)

# Appendix B
add_header("Appendix B: Experimental Provenance and Baseline Reconciliation", level=2)
add_p(
    "Phase 13 results are documented in the ETGNN_Phase13_Results_Bundle.zip, including PHASE13_TEST_METRICS.csv and saved checkpoints. The reported five-seed summary uses seeds 42–46. The historical checkpoint and later neural experiments are distinct evaluation conditions; no additional model training was performed when preparing this document.",
    space_after=6
)
add_p(
    "The canonical historical M0 checkpoint used interleaved sinusoidal time encoding and obtained test AUPRC 0.225030 and ROC-AUC 0.806248 on 9,973 labelled held-out nodes (524 illicit). A Colab evaluation with block-concatenated encoding reported AUPRC 0.255970 and ROC-AUC 0.813184. This input-order difference prevents treating the historical M0 comparison as a controlled estimate of architectural improvement.",
    space_after=6
)
add_p(
    "Phase 4 graph-control comparisons are validation experiments; Phase H uses a 30-node explanation pilot; Phase 13 summarizes five-seed held-out neural results; and C2 evaluates 275 matched false-positive/true-positive validation pairs. Their sample sizes, interventions and splits are not interchangeable.",
    space_after=6
)
add_p(
    "Automated endpoint verification supports software functionality within the tested environment. It does not establish production deployment, model fairness, real-world fraud causation or regulatory compliance.",
    space_after=8
)

# Appendix C
add_header("Appendix C: Five-Seed Neural Model Performance", level=2)
headers_ac = ["Configuration", "Mean AUPRC", "Sample SD", "Seeds"]
rows_ac = [
    ["M1 Residual ETGNN", "0.312623", "0.055945", "42–46"],
    ["M2 Deep MLP", "0.338011", "0.038850", "42–46"],
    ["M3 Reliability Gate CE", "0.241915", "0.046583", "42–46"],
    ["M3 Reliability Gate Focal", "0.227318", "0.035741", "42–46"],
    ["M4 NoGate", "0.312666", "0.050831", "42–46"]
]
add_styled_table(headers_ac, rows_ac, [2.5, 1.5, 1.5, 1.2])
add_p("Table C.1: Five-seed held-out test AUPRC by neural configuration (mean and sample standard deviation).", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)

# Appendix D
add_header("Appendix D: Matched C2 Edge-Intervention Results", level=2)
headers_ad = ["Matched cohort", "Flipped to licit", "Did Not flip"]
rows_ad = [
    ["False positive (n = 275)", "33 (12.00%)", "242 (88.00%)"],
    ["True positive (n = 275)", "12 (4.36%)", "263 (95.64%)"]
]
add_styled_table(headers_ad, rows_ad, [2.5, 2.0, 2.0])
add_p("Table D.1: Matched C2 one-hop edge-deletion classification outcomes.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
add_p(
    "Reported odds ratio = 2.99 (95% CI [1.51, 5.92]); Fisher exact p = 0.0016; Holm-adjusted p = 0.0032. The matched validation observations quantify model sensitivity to a defined one-hop edge deletion, not causal fraud determinations. The cohort was concentrated at in-degree one, limiting generalization.",
    space_after=12
)

doc.save(docx_path)
print(f"Successfully generated final NIBM Project Report DOCX: {docx_path} ({docx_path.stat().st_size:,} bytes)")
