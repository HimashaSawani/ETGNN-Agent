"""
Phase 11 — Evidence-Based Thesis Integration Generator
Generates:
1. thesis/phase11_final_integration/ETGNN_Phase11_Final_Integrated_Thesis.docx
2. thesis/phase11_final_integration/ETGNN_Phase11_Final_Integrated_Thesis.pdf
3. thesis/phase11_final_integration/chapters/01_Introduction.docx .. 06_Conclusion.docx

All numerical claims are loaded or cross-verified directly against the authoritative
Phase 10 CSV artifacts and original results/tables/ files. Original thesis files remain untouched.
"""

import hashlib
import os
from pathlib import Path
import pandas as pd
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "thesis" / "phase11_final_integration"
CHAP_DIR = OUT_DIR / "chapters"
FIG_DIR = ROOT / "results" / "figures"
P10_DIR = ROOT / "research_extension" / "phase10"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHAP_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DOCX STYLING HELPERS
# ==============================================================================
def set_cell_background(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def create_styled_document():
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Times New Roman"
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    style_normal.paragraph_format.line_spacing = 1.25
    style_normal.paragraph_format.space_after = Pt(5)
    return doc


def add_h1(doc, text: str):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    return h


def add_h2(doc, text: str):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
    return h


def add_h3(doc, text: str):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(2)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return h


def add_p(doc, text: str, bold_prefix: str = None):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(5)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(11)
        r_pre.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    return p


def add_eq(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.italic = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    return p


def add_note_box(doc, text: str, title: str = "METHODOLOGICAL & SCIENTIFIC DISCLOSURE"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F2F4F8")
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1F3864"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r_t = p.add_run(f"[{title}]\n")
    r_t.bold = True
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(10)
    r_t.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    r_b = p.add_run(text)
    r_b.font.name = "Times New Roman"
    r_b.font.size = Pt(9.5)
    r_b.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(1)
    p_after.paragraph_format.space_after = Pt(4)


def add_img(doc, img_name: str, caption: str, width: float = 5.5):
    p_path = FIG_DIR / img_name
    if p_path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run()
        r.add_picture(str(p_path), width=Inches(width))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def add_tbl(doc, headers, rows, col_widths=None, caption=None, font_size_pt=8.5):
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(6)
        p_cap.paragraph_format.space_after = Pt(2)
        p_cap.paragraph_format.keep_with_next = True
        r_c = p_cap.add_run(caption)
        r_c.font.name = "Times New Roman"
        r_c.font.size = Pt(9.5)
        r_c.font.bold = True
        r_c.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    # Header row
    trPr0 = tbl.rows[0]._tr.get_or_add_trPr()
    trPr0.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    trPr0.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    for idx, title in enumerate(headers):
        cell = tbl.cell(0, idx)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cell.text = title
        set_cell_background(cell, "1F3864")
        set_cell_margins(cell, top=70, bottom=70, left=70, right=70)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(font_size_pt)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data rows
    for r_idx, row in enumerate(rows):
        trPr = tbl.rows[r_idx + 1]._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        bg = "F7F9FB" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell.text = str(val)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=55, bottom=55, left=65, right=65)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx <= 1 else WD_ALIGN_PARAGRAPH.RIGHT
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(font_size_pt)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    if col_widths:
        for r in tbl.rows:
            for idx, w in enumerate(col_widths):
                r.cells[idx].width = Inches(w)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(5)
    return tbl


# ==============================================================================
# CHAPTER BUILDERS (CHAPTERS 1 TO 6)
# ==============================================================================
def build_chapter_1(doc):
    add_h1(doc, "Chapter 1: Introduction")

    add_h2(doc, "1.1 Background and Context")
    add_p(
        doc,
        "The emergence of decentralized cryptographic ledgers over the past decade has fundamentally transformed "
        "digital payment infrastructure. Bitcoin, introduced by Satoshi Nakamoto in 2008, established a peer-to-peer "
        "electronic payment network operating via unspent transaction outputs (UTXOs) without intermediary financial "
        "institutions. While public blockchains provide cryptographic auditability of transaction amounts and topological "
        "payment flows, their pseudonymous design—wherein participants transact via alphanumeric public keys rather than "
        "verified legal identities—creates severe compliance vulnerabilities. Illicit actors routinely exploit this "
        "pseudonymity to execute ransomware extortions, darknet marketplace settlements, Ponzi schemes, and multi-hop "
        "money laundering."
    )
    add_p(
        doc,
        "To satisfy international Anti-Money Laundering (AML) and Counter-Financing of Terrorism (CFT) mandates issued "
        "by bodies such as the Financial Action Task Force (FATF), Virtual Asset Service Providers (VASPs) and forensic "
        "investigators require automated transaction-screening algorithms. Traditional rule-based heuristics and static "
        "address blacklists degrade rapidly when confronted with modern laundering typologies such as peeling chains, "
        "high-frequency coin mixing services, and multi-hop layering across high-degree exchange hubs."
    )

    add_h2(doc, "1.2 The Financial Fraud Detection Challenge in Transaction Graphs")
    add_p(
        doc,
        "Supervised machine learning on cryptocurrency payment networks confronts three intersecting structural challenges:"
    )
    add_p(
        doc,
        "1. Severe Class Imbalance and Prevalence Drift: Confirmed illicit transactions constitute a small fraction of "
        "total network volume (only 4,545 illicit vs. 42,019 licit and 157,205 unlabelled transactions in the benchmark "
        "Elliptic dataset; 5.25% positive prevalence among labelled test nodes). Moreover, illicit prevalence shifts "
        "abruptly over time—most notably following the international law-enforcement takedown of the AlphaBay darknet "
        "market at Elliptic time step 43, where positive prevalence drops from 11.10% (step 42) to 1.75% (step 43) and "
        "0.28% (step 46)."
    )
    add_p(
        doc,
        "2. Heterophilic Mixing and Feature Redundancy: While Bitcoin payment flows form directed graphs within each "
        "observation window, illicit actors frequently transact through high-degree licit exchange and mixer hubs. "
        "Consequently, unconstrained neighborhood message passing can dilute intrinsic local transaction signatures "
        "(features 1–93), especially when precomputed one-hop neighborhood summary statistics (features 94–165) are "
        "already present in the tabular feature matrix."
    )
    add_p(
        doc,
        "3. Black-Box Opacity and Uncalibrated Post-Hoc Explanations: In regulated financial forensics, flagging a "
        "transaction requires auditable justification. While post-hoc explainers such as GNNExplainer (Ying et al., 2019) "
        "optimize continuous edge masks on local computation subgraphs, standard evaluations rely on visual inspection "
        "or uncalibrated population averages on correctly classified instances, neglecting rigorous random-deletion "
        "calibration, candidate-edge sparsity constraints, and matched comparisons between true detections and false alarms."
    )

    add_h2(doc, "1.3 Research Problem Statement")
    add_p(
        doc,
        "Public transaction-level benchmarks such as the Elliptic Bitcoin dataset record directed payment edges strictly "
        "within discrete bi-weekly time steps (t in {1, ..., 49}), with zero cross-time-step edges (V_t intersect V_t' = empty). "
        "Consequently, two core scientific questions arise: First, when evaluated under strict forward-in-time chronological "
        "splitting without data leakage, how does a Graph Attention Network augmented with deterministic sinusoidal time-step "
        "encodings (ETGNN) perform relative to competitive non-graph baselines (Random Forest, XGBoost, Deep MLPs, and "
        "Logistic Regression) and static GNNs (GCN, GraphSAGE, GAT, and a parameter-matched zero-time GAT control), and how "
        "do local transaction features (1–93), precomputed 1-hop aggregates (94–165), and graph edges interact? Second, under "
        "controlled post-hoc edge-deletion interventions on frozen checkpoints, do GNNExplainer subgraphs exhibit differential "
        "sensitivity between ETGNN and Matched GAT under shared interventions (Protocol C1) or between correct predictions "
        "and systematic classification errors under prediction-margin and degree-matched bipartite pairing (Protocol C2)?"
    )

    add_h2(doc, "1.4 Approved Research Questions")
    add_p(
        doc,
        "Preserving the approved revised research proposal (`reports/research_questions.md`), this thesis investigates "
        "three primary Research Questions (RQ1–RQ3), where RQ3 is operationalized through two formal interventional "
        "protocols (Contrast C1 and Contrast C2):"
    )
    add_p(
        doc,
        "How does the implemented time-encoded Graph Attention Network (ETGNN) compare with non-relational baselines "
        "(Random Forest, XGBoost, Multi-Layer Perceptrons, Logistic Regression) and graph neural network baselines "
        "(GCN, GraphSAGE, standard GAT) on held-out future Elliptic time steps (41–49) when illicit-class Average "
        "Precision (AP) is the primary evaluation metric, and how do local features (1–93), precomputed neighborhood "
        "aggregates (94–165), and explicit graph edges govern this performance hierarchy?",
        bold_prefix="• Research Question 1 (RQ1 — Fair Chronological Benchmarking & Feature/Topology Ablation): "
    )
    add_p(
        doc,
        "Does concatenating a deterministic 16-dimensional sinusoidal time-step encoding to node features (ETGNN) "
        "yield a statistically discernible improvement in illicit-class Average Precision over a strictly parameter-matched "
        "zero-time GAT control (11,910 parameters) across validation seeds and on the held-out chronological test split?",
        bold_prefix="• Research Question 2 (RQ2 — Isolated Temporal Positional Encoding Contribution): "
    )
    add_p(
        doc,
        "What do post-hoc GNNExplainer edge masks and random-calibrated edge-deletion interventions reveal about "
        "structural prediction sensitivity on the Elliptic transaction graph, and what methodological constraints limit "
        "their interpretation? Specifically:\n"
        "  (a) Protocol C1 (Shared-Intervention Model Sensitivity): When subjected to identical top-k edge deletions "
        "selected by ETGNN's GNNExplainer across N = 240 validation transactions (steps 37–40), do frozen ETGNN and "
        "Matched GAT checkpoints exhibit a statistically significant difference in random-adjusted probability drop?\n"
        "  (b) Protocol C2 (Matched Correct-vs-Incorrect Error Faithfulness): After 1-to-1 Hopcroft-Karp bipartite matching "
        "within each time step on prediction margin (<= 0.050), candidate-edge count (<= 5), and exact out-degree across "
        "8 comparison registers (N = 420 pair records across 377 unique transactions), does random-adjusted deletion "
        "sensitivity differ systematically between true detections (TP, TN) and classification errors (FP, FN)?",
        bold_prefix="• Research Question 3 (RQ3 — Controlled Interventional Explainability Auditing via Protocols C1 & C2): "
    )

    add_h2(doc, "1.5 Approved Research Objectives")
    add_p(
        doc,
        "To answer RQ1–RQ3 rigorously, the five approved research objectives (RO1–RO5) are executed and verified:"
    )
    add_p(
        doc,
        "1. Objective 1 (RO1 — Leak-Free Chronological Pipeline & Multi-Paradigm Benchmarking): Construct a zero-leakage "
        "chronological data pipeline on the Elliptic dataset (fitting StandardScaler strictly on training steps 1–32) and "
        "benchmark 15 distinct local model configurations spanning tree ensembles, linear classifiers, deep MLPs, and static/temporal "
        "GNNs on held-out test steps 41–49 using Average Precision as the primary metric.\n"
        "2. Objective 2 (RO2 — Parameter-Matched Temporal Encoding Isolation): Implement the 181-dimensional ETGNN architecture "
        "(165 features + 16 sinusoidal time encodings; 11,910 parameters) alongside an exact parameter-matched zero-time GAT "
        "control (165 features + 16 zeros; 11,910 parameters) to isolate the effect of snapshot time encoding from parameter capacity.\n"
        "3. Objective 3 (RO3 — Controlled Feature-Subspace and Edge-Removal Ablations): Quantify the individual and interactive "
        "effects of local transaction features (93f), precomputed 1-hop neighborhood aggregates (72f), and runtime message-passing "
        "edges (E = 234,355 vs. E = 0) across Random Forest, Logistic Regression, Deep MLP, and ETGNN.\n"
        "4. Objective 4 (RO4 — Protocol C1 Shared-Edge Intervention Audit): Execute and statistically audit Contrast 1 across "
        "N = 240 stratified validation transactions (steps 37–40), calibrating GNNExplainer top-k edge deletions against matched "
        "random edge deletions across frozen ETGNN and Matched GAT checkpoints.\n"
        "5. Objective 5 (RO5 — Protocol C2 Matched Error-Register Audit, Topology Sparsity Analysis & Interactive Compliance System): "
        "Execute Contrast 2 across 8 Hopcroft-Karp matched registers (N = 420 pair records across 377 unique transactions) using "
        "B = 2,000 cluster-bootstrap resamples, audit the structural impact of candidate-edge sparsity (E_cand = 2 budget inflation), "
        "and deploy a verified FastAPI + Streamlit compliance demonstration system."
    )

    add_h2(doc, "1.6 Scope Delimitations and Architectural Honesty")
    add_p(
        doc,
        "In accordance with strict scientific integrity standards, the boundaries of this thesis are explicitly delimited:"
    )
    add_p(
        doc,
        "• Static Sinusoidal Time Encoding vs. Dynamic Recurrent Graph Learning: Inspection of `src/models/etgnn.py` and "
        "`src/models/time_encoding.py` confirms that ETGNN is a 2-layer Graph Attention Network (`GATConv`) operating on "
        "discrete intra-step transaction graphs with a deterministic 16-dimensional sinusoidal time vector concatenated to node "
        "features. It does NOT implement persistent node-state memory (e.g., TGN, Rossi et al., 2020) or recurrent weight-matrix "
        "evolution across snapshots (e.g., EvolveGCN, Pareja et al., 2020).\n"
        "• Meaning of 'Adaptive' in the Registered Title: No online parameter updating, test-time adaptation, or streaming concept-drift "
        "retraining is implemented; models are trained offline on steps 1–32 and evaluated with frozen weights on steps 41–49. "
        "The word 'Adaptive' in the officially registered title refers to evaluating model resilience under adaptive adversarial "
        "regime shifts on the blockchain rather than an online adaptive learning algorithm.\n"
        "• Non-Causal Epistemic Boundary of Edge Deletions: Post-hoc edge-deletion metrics quantify local model prediction sensitivity "
        "to topological omission conditional on fixed tabular features; they do NOT establish real-world causal money-laundering "
        "pathways or automated error repair."
    )

    add_h2(doc, "1.7 Organization of the Thesis")
    add_p(
        doc,
        "Chapter 2 reviews the scholarly literature on cryptocurrency AML forensics, tabular and geometric learning, dynamic "
        "graph paradigms (including a rigorous mathematical comparison with EvolveGCN), and explainability benchmarks (GraphFramEx, "
        "ROAR). Chapter 3 formalizes the dataset, chronological split, model equations, feature ablations, C1/C2 protocols, "
        "evaluation metric definitions, and statistical testing protocols. Chapter 4 documents the software implementation, "
        "checkpoint provenance, verification assertions, and interactive FastAPI/Streamlit system. Chapter 5 presents and "
        "discusses all empirical results—including the 15-model local benchmark, feature/edge ablations, node-level and "
        "time-step-clustered statistical tests, Colab GPU reproducibility analysis, and C1/C2 explainability audits. "
        "Chapter 6 synthesizes evidence-backed answers to RQ1–RQ3, enumerates all material methodological limitations, and "
        "outlines concrete future work."
    )


def build_chapter_2(doc):
    add_h1(doc, "Chapter 2: Literature Review")

    add_h2(doc, "2.1 Anti-Money Laundering Forensics on Bitcoin Transaction Graphs")
    add_p(
        doc,
        "Early cryptocurrency forensic research relied on address-clustering heuristics—such as multi-input co-spending "
        "and peel-chain change-address identification—to link pseudonymous Bitcoin addresses into entity clusters. However, "
        "criminal syndicates increasingly bypass heuristic clustering via automated mixers, privacy wallets, and rapid layering "
        "through high-volume exchanges. Weber et al. (2019) established the foundational supervised machine learning benchmark "
        "for Bitcoin AML by releasing the Elliptic dataset, comprising 203,769 Bitcoin transactions and 234,355 directed payment "
        "edges partitioned into 49 discrete bi-weekly time steps. Crucially, Weber et al. (2019) demonstrated two enduring "
        "empirical regularities: (1) non-relational Random Forest classifiers trained on the 165 numeric features substantially "
        "outperformed standard Graph Convolutional Networks (GCN) on chronological test splits, and (2) all models suffered "
        "acute performance degradation following step 43, coinciding with the July 2017 law-enforcement shutdown of the "
        "AlphaBay darknet market."
    )

    add_h2(doc, "2.2 Tabular Classifiers and Feature Provenance in Elliptic")
    add_p(
        doc,
        "Ensemble tree algorithms remain dominant baselines for structured financial data. Breiman (2001) introduced Random "
        "Forests, leveraging bootstrap aggregation and random feature subspacing to achieve low-variance, non-linear decision "
        "boundaries without sensitivity to monotonic feature scaling. Chen and Guestrin (2016) formalized XGBoost, incorporating "
        "second-order regularized gradient boosting and sparsity-aware split finding. Alongside linear baselines (Logistic "
        "Regression) and feed-forward Multi-Layer Perceptrons (MLPs), tree ensembles provide essential non-message-passing "
        "controls."
    )
    add_p(
        doc,
        "A critical methodological insight reported by Weber et al. (2019) is that the 165 published Elliptic features are "
        "partitioned into 93 local transaction features (columns 1–93: inputs, outputs, fees, volumes) and 72 precomputed "
        "one-hop neighborhood aggregates (columns 94–165: minimum, maximum, mean, and standard deviation of neighboring "
        "transactions' local features). Therefore, any classifier trained on the full 165-feature matrix already ingests "
        "explicit 1-hop graph summary statistics. Separating intrinsic feature learning from relational aggregation requires "
        "controlled ablations comparing the 93-feature local subspace against the 165-feature full space across both tabular "
        "and neural paradigms."
    )

    add_h2(doc, "2.3 Static Graph Representation Learning (GCN, GraphSAGE, GAT)")
    add_p(
        doc,
        "Message Passing Neural Networks propagate representations across graph edges by iteratively aggregating neighbor "
        "states. Kipf and Welling (2017) formulated the Graph Convolutional Network (GCN) using symmetric degree-normalized "
        "adjacency averaging, which implicitly assumes homophily—that connected nodes share identical class labels. Hamilton, "
        "Ying, and Leskovec (2017) proposed GraphSAGE, decoupling the ego-node representation from the mean-aggregated "
        "neighborhood representation via separate weight matrices (W_1 h_v + W_2 MEAN({h_u})), thereby preserving central-node "
        "features even when neighbor labels are mixed. Veličković et al. (2018) introduced Graph Attention Networks (GAT), "
        "replacing fixed degree normalization with learned multi-head softmax attention coefficients alpha_vu over "
        "1-hop neighborhoods (including self-loops). However, in standard GAT without an explicit ego-residual skip projection, "
        "the ego-node's features compete inside the same softmax normalization against all incoming neighbors; in heterophilic "
        "hubs, attention weights distributed across numerous licit neighbors can dilute discriminative local features."
    )

    add_h2(doc, "2.4 Temporal Graph Paradigms and Rigorous Feasibility Analysis of EvolveGCN")
    add_p(
        doc,
        "Temporal graph learning methods are divided into two distinct mathematical categories based on how they model "
        "time and entity persistence:"
    )
    add_p(
        doc,
        "Architectures such as Temporal Graph Networks (TGN; Rossi et al., 2020) maintain a persistent state vector s_v(t) "
        "for each node v, updated via a recurrent module whenever node v participates in a new timestamped interaction. "
        "Because Elliptic nodes represent one-time Bitcoin transactions partitioned into disjoint bi-weekly sets "
        "(V_t intersect V_t' = empty for t != t') with zero cross-snapshot edges, persistent node-state recurrence is "
        "structurally inapplicable on the transaction-level Elliptic DAG.",
        bold_prefix="1. Persistent Node-State Recurrence (Inapplicable to Disjoint Transaction Sets): "
    )
    add_p(
        doc,
        "To handle dynamic graphs where node sets change or are completely disjoint across snapshots, Pareja, Domeniconi, "
        "Chen, Ma, Suzumura, Kanezashi, Kaler, Schardl, and Leiserson (2020) introduced EvolveGCN at AAAI 2020—and benchmarked "
        "it directly on the Elliptic Bitcoin dataset. Rather than evolving node states, EvolveGCN uses a recurrent neural "
        "network to evolve the fixed-dimension GCN weight matrices W_t^(l) in R^(d^(l) x d^(l+1)) across snapshots t = 1..49:\n"
        "  • EvolveGCN-O evolves weight matrices via an LSTM conditioned purely on prior weight matrices: "
        "W_t^(l) = LSTM(W_(t-1)^(l)). This recurrence is completely independent of whether nodes persist across snapshots.\n"
        "  • EvolveGCN-H evolves weight matrices via a GRU conditioned on a fixed-size Top-k pooling summary of the current "
        "snapshot's node embeddings: W_t^(l) = GRU(TopK(H_t^(l), k), W_(t-1)^(l)). Because Top-k pooling projects variable-sized "
        "node sets |V_t| into a fixed k x d^(l+1) matrix matching the column dimension of W_(t-1)^(l), EvolveGCN-H is also "
        "mathematically well-defined on disjoint node sets, although pooling across disjoint transaction cohorts captures "
        "macro-level snapshot feature distributions rather than persistent entity trajectories.\n"
        "Crucially, in both Weber et al. (2019) and Pareja et al. (2020), EvolveGCN remained substantially below Random "
        "Forest on the Elliptic benchmark, demonstrating that evolving GCN weights across snapshots does not overcome the "
        "underlying dilution of local transaction features by GCN message passing. In contrast, our implemented ETGNN "
        "adopts the deterministic sinusoidal positional encoding of Vaswani et al. (2017) concatenated to GAT node inputs, "
        "allowing controlled isolation of static snapshot time encoding against a parameter-matched zero-time GAT baseline.",
        bold_prefix="2. Weight-Matrix Evolution Across Disjoint Snapshots (EvolveGCN-O and EvolveGCN-H; Pareja et al., 2020): "
    )

    add_h2(doc, "2.5 Explainable Graph Neural Networks and Faithfulness Benchmarks")
    add_p(
        doc,
        "Post-hoc explainability methods for GNNs aim to identify a compact subgraph G_S subset G_c that accounts for a "
        "target prediction. Ying, Bourgeois, You, Zitnik, and Leskovec (2019) proposed GNNExplainer, which optimizes a "
        "continuous edge mask M in [0, 1]^|E_c| by maximizing mutual information between the original prediction and the "
        "masked subgraph prediction. Luo et al. (2020) subsequently proposed PGExplainer, using a parameterized network "
        "to predict discrete edge masks across instances."
    )
    add_p(
        doc,
        "Evaluating post-hoc explanations without ground-truth motifs requires interventional fidelity testing. Hooker, "
        "Erhan, Kindermans, and Kim (2019) demonstrated in the ROAR benchmark that feature-deletion curves must be calibrated "
        "against random deletion baselines to separate genuine attribution selectivity from generic out-of-distribution "
        "perturbation shock. In the graph domain, Amara, Ying, Zhang, Han, Zhao, Shan, Brandes, Schemm, and Zhang (2022) "
        "formalized GraphFramEx (Proceedings of Machine Learning Research, Vol. 198:44:1–44:23), defining Fidelity+ "
        "(sufficiency via top-k edge deletion drop) and Fidelity- (necessity) across graph explanation methods, while "
        "Agarwal, Queen, Lakkaraju, and Zitnik (2023) introduced GraphXAI for standardized explanation evaluation."
    )

    add_h2(doc, "2.6 Precision–Recall Evaluation Under Severe Class Imbalance")
    add_p(
        doc,
        "Saito and Rehmsmeier (2015) proved that on severely imbalanced binary classification tasks, Receiver Operating "
        "Characteristic (ROC-AUC) curves remain deceptively high even under large false-positive counts because false "
        "positives are normalized by the massive true-negative pool, whereas Precision–Recall (PR) curves directly penalize "
        "false alarms relative to true detections. Furthermore, Davis and Goadrich (2006) established that linear trapezoidal "
        "interpolation between points on the Precision–Recall curve (`sklearn.metrics.auc(recall, precision)`) is mathematically "
        "inaccurate because precision does not vary linearly with recall between operating thresholds; when a classifier "
        "exhibits probability ties or large recall steps, trapezoidal interpolation overestimates the area under the curve. "
        "Step-function Average Precision (`sklearn.metrics.average_precision_score`) avoids linear interpolation bias and "
        "serves as the rigorous primary metric in this thesis."
    )

    add_h2(doc, "2.7 Comparative Literature Matrix and Research Gap")
    headers_lit = ["Author & Year", "Domain / Architecture", "Verified Primary Contribution", "Relationship to Present Thesis"]
    rows_lit = [
        ["Weber et al. (2019)", "Bitcoin AML / Elliptic", "Released Elliptic (203,769 nodes, 49 steps); showed RF beats GCN and EvolveGCN; noted step 43 market closure.", "Primary dataset; motivates our 15-model tabular vs. GNN benchmark and step 41–49 regime analysis."],
        ["Pareja et al. (2020)", "Dynamic GNN / EvolveGCN", "Evolved GCN weight matrices via RNN (EvolveGCN-O LSTM & EvolveGCN-H Top-k GRU) on disjoint snapshots.", "Clarifies weight-matrix evolution vs. node-state recurrence; motivates our scope delimitation and future work."],
        ["Breiman (2001) & Chen & Guestrin (2016)", "Tabular Ensembles (RF & XGBoost)", "Formalized bagged random subspace trees (RF) and regularized second-order gradient boosting (XGBoost).", "Primary non-relational benchmarks achieving 0.6323 AP (RF) and 0.5472 AP (XGBoost) on Elliptic."],
        ["Kipf & Welling (2017); Hamilton et al. (2017); Veličković et al. (2018)", "Static GNNs (GCN, GraphSAGE, GAT)", "Spectral convolution (GCN), ego-separated mean aggregation (GraphSAGE), and multi-head attention (GAT).", "GNN baselines and the 2-layer GAT backbone of Matched GAT and ETGNN."],
        ["Vaswani et al. (2017)", "Sinusoidal Positional Encoding", "Deterministic multi-frequency sine/cosine positional encodings for sequence order.", "Mathematical formulation of our 16-d snapshot time encoding PE(t) in ETGNN and RF + Time."],
        ["Rossi et al. (2020)", "Continuous-Time GNN (TGN)", "Persistent node memory vectors updated on continuous-time interaction streams.", "Demarcates what ETGNN does not implement (requires persistent node IDs across events)."],
        ["Ying et al. (2019) & Luo et al. (2020)", "Post-Hoc Graph XAI (GNNExplainer, PGExplainer)", "Mutual-information optimization of continuous subgraph edge masks.", "Core post-hoc explanation optimizer audited in Protocols C1 and C2."],
        ["Hooker et al. (2019)", "XAI Calibration (ROAR)", "Demonstrated that deletion curves must be calibrated against random perturbation baselines.", "Motivates our random-deletion subtraction (Adv_M = Delta_expl - Delta_rand) in C1 and C2."],
        ["Amara et al. (2022) & Agarwal et al. (2023)", "Graph XAI Benchmarks (GraphFramEx, GraphXAI)", "Formalized Fidelity+ deletion metrics and systematic graph explanation evaluation.", "Direct methodological foundation extended by our C1 shared-edge and C2 error-matched protocols."],
        ["Davis & Goadrich (2006) & Saito & Rehmsmeier (2015)", "Imbalanced Statistical Evaluation", "Proved PR superiority over ROC in imbalance and step-function AP superiority over trapezoidal PR-AUC.", "Establishes Average Precision (`average_precision_score`) as our primary standardized metric."]
    ]
    add_tbl(doc, headers_lit, rows_lit, col_widths=[1.3, 1.3, 2.0, 1.9],
            caption="Table 2.1: Verified Comparative Literature Matrix and Methodological Positioning.")

    add_p(
        doc,
        "While Weber et al. (2019) and Pareja et al. (2020) benchmarked predictive performance on Elliptic, and Amara et al. "
        "(2022) formalized general graph explanation fidelity in GraphFramEx, prior work has not jointly (1) isolated "
        "local vs. precomputed aggregate feature dilution across linear, MLP, tree, and time-encoded GAT models with "
        "time-step-clustered statistical sensitivity checks, (2) audited how Elliptic's extreme 2-hop candidate-edge "
        "sparsity (84.08% of nodes having E_cand = 2) distorts fractional deletion budgets, or (3) evaluated random-calibrated "
        "GNNExplainer edge deletions across Hopcroft-Karp confidence-margin-matched and exact-degree-matched true-detection "
        "vs. false-alarm pairs (Protocol C2). Addressing these three empirical questions forms the core contribution of this thesis."
    )


def build_chapter_3(doc):
    add_h1(doc, "Chapter 3: Methodology")

    add_h2(doc, "3.1 Research Design Overview")
    add_p(
        doc,
        "The methodology is organized into four rigorous analytical stages: (1) leak-free chronological data partitioning "
        "and feature standardization on the Elliptic Bitcoin network; (2) multi-paradigm classification benchmarking and "
        "controlled feature/topology ablations across 15 local model configurations; (3) standardized metric evaluation "
        "and dual node-level / time-step-clustered statistical hypothesis testing; and (4) retrospective post-hoc "
        "explainability auditing under Protocols C1 and C2."
    )

    add_h2(doc, "3.2 Elliptic Dataset Architecture and Feature Decomposition")
    add_p(
        doc,
        "The Elliptic Bitcoin dataset (`data/raw/elliptic/`) consists of |V| = 203,769 transaction nodes and |E| = 234,355 "
        "directed payment-flow edges partitioned across T = 49 discrete chronological time steps (t in {1, ..., 49}, sampled "
        "at approximately two-week intervals). Every directed edge (u -> v) connects two transactions occurring strictly within "
        "the same time step (t(u) = t(v)); there are zero cross-time-step edges. Ground-truth node labels comprise:\n"
        "  • Illicit (y = 1, raw class '1'): 4,545 transactions (2.23% of all nodes; 9.76% of labelled nodes).\n"
        "  • Licit (y = 0, raw class '2'): 42,019 transactions (20.62% of all nodes; 90.24% of labelled nodes).\n"
        "  • Unknown (y = -1, raw class 'unknown'): 157,205 transactions (77.15% of all nodes), which participate as "
        "intermediate structural vertices during GNN message passing but are strictly masked out of loss calculations and "
        "evaluation metrics."
    )
    add_p(
        doc,
        "Each transaction v is associated with a 165-dimensional continuous feature vector x_v in R^165 decomposed into:\n"
        "  • Local Transaction Features (x_v^(local) in R^93, columns 1–93): Intrinsic UTXO transaction attributes including "
        "input/output counts, transaction fees, total BTC volume, and average output values.\n"
        "  • Precomputed 1-Hop Neighborhood Aggregates (x_v^(agg) in R^72, columns 94–165): Summary statistics (minimum, "
        "maximum, mean, standard deviation) of local features computed across 1-hop neighboring transactions."
    )

    add_h2(doc, "3.3 Chronological Partitioning and Zero-Leakage Feature Scaling")
    add_p(
        doc,
        "To prevent temporal look-ahead leakage, the 49 time steps are partitioned into three disjoint chronological windows "
        "(`src/data/create_masks.py`, `configs/experiment_config.yaml`):\n"
        "  • Training Split (Steps 1–32): 130,628 total nodes; 29,721 labelled nodes (2,987 illicit [10.05%], 26,734 licit).\n"
        "  • Validation Split (Steps 33–40): 30,975 total nodes; 7,653 labelled nodes (619 illicit [8.09%], 7,034 licit).\n"
        "  • Held-Out Test Split (Steps 41–49): 42,166 total nodes; 9,973 labelled nodes (524 illicit [5.25%], 9,449 licit)."
    )
    add_p(
        doc,
        "Feature standardization (`src/data/scale_features.py`) fits `sklearn.preprocessing.StandardScaler` exclusively on "
        "the 130,628 training nodes (steps 1–32), freezing the training mean vector mu_train and standard deviation vector "
        "sigma_train to transform validation and test nodes without future distribution leakage."
    )

    add_h2(doc, "3.4 Mathematical Formulation of ETGNN and Baseline Architectures")
    add_h3(doc, "3.4.1 Sinusoidal Temporal Positional Encoding and Proposed ETGNN")
    add_p(
        doc,
        "For a transaction v occurring at discrete time step t in {1, ..., 49}, the deterministic 16-dimensional sinusoidal "
        "temporal encoding PE(t) in R^16 (`src/models/time_encoding.py`) is defined for channel index i in {0, ..., 7} as:"
    )
    add_eq(doc, "PE(t, 2i) = sin( t / 10000^(2i / 16) ),    PE(t, 2i + 1) = cos( t / 10000^(2i / 16) )")
    add_p(
        doc,
        "The initial node representation concatenates the standardized feature vector x_v in R^d_in (where d_in = 165 for "
        "full features or d_in = 93 for local features) with PE(t): h_v^(0) = [x_v || PE(t)] in R^(d_in + 16). The ETGNN "
        "backbone (`src/models/etgnn.py`) applies two multi-head Graph Attention (`GATConv`) layers:\n"
        "  • Layer 1: 4 attention heads with 16 hidden channels per head, concatenated to 64 channels (d_in + 16 -> 64), "
        "followed by ELU activation and dropout (p = 0.3).\n"
        "  • Layer 2: 1 attention head mapping 64 -> 64 channels with ELU activation, followed by a linear classification "
        "head mapping 64 -> 2 logits.\n"
        "Total trainable parameters are 11,910 for `ETGNN (165f Full)` (`181 -> 64 -> 64 -> 2`) and 7,302 for `ETGNN (93f Local)` "
        "(`109 -> 64 -> 64 -> 2`)."
    )

    add_h3(doc, "3.4.2 Parameter-Matched Zero-Time GAT Control and Static GNN Baselines")
    add_p(
        doc,
        "To isolate the effect of PE(t) while holding network architecture and parameter capacity strictly constant, "
        "`Matched GAT (165f Full)` (`src/models/matched_gat.py`) replaces PE(t) with a 16-dimensional zero vector 0_16, "
        "yielding h_v^(0) = [x_v || 0_16] in R^181 with the exact same 11,910 trainable parameters.\n"
        "The three unpadded static GNN baselines (`src/models/gcn.py`, `graphsage.py`, `gat.py`) operate directly on "
        "x_v in R^165:\n"
        "  • `GCN (165f Full)`: Two `GCNConv` layers (`165 -> 64 -> 64`) + `Linear(64 -> 2)` (10,754 parameters).\n"
        "  • `GraphSAGE (165f Full)`: Two `SAGEConv` mean-aggregator layers (`165 -> 64 -> 64`) + `Linear(64 -> 2)` (21,442 parameters).\n"
        "  • `GAT Baseline (165f Full)`: Two `GATConv` layers (`165 -> 16x4 -> 64`) + `Linear(64 -> 2)` (10,886 parameters)."
    )

    add_h3(doc, "3.4.3 Non-Relational Baselines and Feature/Topology Ablation Matrix")
    add_p(
        doc,
        "To benchmark message-passing GNNs fairly against non-graph paradigms across both `165f Full` and `93f Local` "
        "feature spaces, the following non-graph baselines and ablations are evaluated:\n"
        "  • `Random Forest (165f Full)`, `Random Forest (93f Local)`, and `RF + Time (165f + 16t)`: 100 decision trees "
        "(`class_weight='balanced_subsample'`, `random_state=42`).\n"
        "  • `XGBoost (165f Full)`: 300 boosted trees (`max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, "
        "`scale_pos_weight=10.355`).\n"
        "  • `ETGNN (93f Local No Edges)`: Trained from scratch (`src/training/train_etgnn_local_no_edges.py`) with identical "
        "architecture and 7,302 parameters as `ETGNN (93f Local)` but with an empty edge index (E = 0). Because `GATConv` "
        "adds self-loops (v, v) with softmax attention weight alpha_vv = 1.0, this model functions as a strictly parameter-matched "
        "2-layer node-wise neural network isolating the exact impact of removing graph edges.\n"
        "  • `DeepParameterMatchedMLP (165f Full)` and `DeepParameterMatchedMLP (93f Local)` (`P3-05`, seeds 42, 43, 44): "
        "3-layer feed-forward network (`d_in + 16 -> 96 -> 96 -> 64 -> 2`) with `LayerNorm` and `ELU` activations (31,682 "
        "parameters on 165f+16t; 22,466 parameters on 93f+16t). Because `DeepParameterMatchedMLP` differs in depth, width, "
        "and normalization from the 2-layer ETGNN (11,910 / 7,302 parameters), it is evaluated as a higher-capacity feed-forward "
        "neural baseline rather than a single-variable edge ablation.\n"
        "  • `Logistic Regression (165f Full)` and `Logistic Regression (93f Local)`: L2-regularized linear classifiers "
        "(`C=1.0`, `class_weight='balanced'`, `max_iter=1000`, `random_state=42`; 166 and 94 parameters, respectively).\n"
        "  • Explicitly Missing Configurations: `XGBoost (93f Local)`, `Matched GAT (93f Local)`, and `GCN/GraphSAGE/GAT (93f Local)` "
        "were not trained in the original historical repository and are transparently marked as `MISSING` in our feature-ablation matrix."
    )

    add_h2(doc, "3.5 Standardization of Evaluation Metrics")
    add_p(
        doc,
        "To resolve metric terminology discrepancies across prior scripts, all models are evaluated on the N = 9,973 test "
        "nodes using two strictly separated area-under-the-PR-curve estimators:\n"
        "  1. Primary Thesis Metric — Average Precision (`AP_Primary`, `sklearn.metrics.average_precision_score`):"
    )
    add_eq(doc, "AP = sum_n (R_n - R_(n-1)) * P_n")
    add_p(
        doc,
        "  2. Supplementary Metric — Trapezoidal PR-AUC (`PR_AUC_Trapz`, `sklearn.metrics.auc(recall, precision)`):"
    )
    add_eq(doc, "PR-AUC_trapz = sum_n (R_n - R_(n-1)) * (P_n + P_(n-1)) / 2")
    add_p(
        doc,
        "Threshold-dependent metrics (Precision, Recall, F1-score, and confusion-matrix counts TP, FP, TN, FN) are evaluated "
        "at the canonical probability threshold tau = 0.50, alongside threshold-free ROC-AUC."
    )

    add_h2(doc, "3.6 Statistical Testing Methodology: Paired Bootstrap and Time-Step Clustering")
    add_p(
        doc,
        "For M = 12 pre-specified model contrasts on the primary metric (Average Precision), two complementary statistical "
        "procedures are executed (`research_extension/phase10/run_phase10_audit.py`):\n"
        "  1. Pooled Node-Level Paired Bootstrap (B = 2,000, Common Random Numbers seed 42): Resamples N = 9,973 test indices "
        "with replacement, computing Delta_AP^(b) = AP(y^(b), p_A^(b)) - AP(y^(b), p_B^(b)), percentile 95% confidence intervals, "
        "and the exact unbiased finite-sample two-sided Monte Carlo p-value (Phipson & Smyth, 2010):"
    )
    add_eq(doc, "p_exact = min( 1.0,  2 * (k_opp + 1) / (B + 1) )")
    add_p(
        doc,
        "where k_opp is the count of bootstrap replicates with sign opposite to the observed Delta_AP (yielding p_min = 2/2001 = 0.0010 "
        "when k_opp = 0). Family-wise error rate across all M = 12 contrasts is controlled via the monotonic step-down "
        "Holm-Bonferroni procedure at alpha = 0.05.\n"
        "  2. Time-Step-Clustered Sensitivity Analysis (T = 9 test snapshots, steps 41–49): Because transactions within the "
        "same snapshot share connected graph components and macroeconomic regime shifts, i.i.d. node resampling understates "
        "cross-period variance. We therefore compute within-step AP_t for each t in {41..49}, report step win counts "
        "(Steps A > B out of 9), paired Student's t-test and exact Wilcoxon signed-rank p-values across the 9 steps, and "
        "a B = 2,000 Time-Step Block Bootstrap resampling whole time-step clusters with replacement."
    )

    add_h2(doc, "3.7 Post-Hoc Explainability Protocols C1 and C2")
    add_p(
        doc,
        "Protocols C1 and C2 (`configs/main_c1_c2_run_config.json`) audit frozen `ETGNN (165f Full)` (SHA-256 `b2f5535f...`) "
        "and `Matched GAT (165f Full)` (SHA-256 `1e9b43ea...`) checkpoints on validation time steps 37–40. For each target "
        "node v, the 2-hop directed computation subgraph G_c(v) is extracted (verified to reproduce full-graph forward-pass "
        "probabilities within <= 3.58e-7). Given the candidate edge set E_cand(v) (excluding self-loops), the deletion budget "
        "follows the discrete ceiling rule:"
    )
    add_eq(doc, "k(v) = max( 1, ceil( 0.25 * |E_cand(v)| ) )")
    add_p(
        doc,
        "Protocol C1 (Shared-Intervention Model Sensitivity, N = 240): Across 240 stratified validation nodes (60 per step "
        "in steps 37–40: 30 illicit, 30 licit), ETGNN serves as the ranking model: GNNExplainer selects the top-k edges "
        "Delta E_shared(v) wrt ETGNN's predicted class c* = y^_ETGNN(v). The identical edge subset Delta E_shared(v) is "
        "deleted from both frozen ETGNN and frozen Matched GAT, calibrated against matched random edge deletions to compute "
        "the paired random-adjusted difference Delta_adj(v) = Adv_ETGNN(v) - Adv_MatchedGAT(v).\n"
        "Protocol C2 (Within-Model Matched Error-vs-Correct Faithfulness, N = 420 pair records across 377 unique nodes): "
        "Each model is evaluated using its own GNNExplainer rankings and its own predicted class. Within each time step "
        "t in {37..40}, Hopcroft-Karp bipartite matching pairs erroneous predictions (FP or FN) with correct predictions "
        "(TP or TN) sharing the same predicted class under:\n"
        "  • Primary Calipers: |margin(v_err) - margin(v_corr)| <= 0.050 (where margin(v) = |p_illicit(v) - 0.50|) and "
        "|E_cand(v_err) - E_cand(v_corr)| <= 5.\n"
        "  • Degree-Sensitivity Calipers: Enforces both Primary calipers plus exact out-degree equality (deg_out(v_err) = deg_out(v_corr)).\n"
        "Paired differences across the 8 registers are evaluated via B = 2,000 cluster-bootstrap resamples (seed 20261020), "
        "reporting both the original marginal 95% CIs and post-hoc Holm-Bonferroni multiplicity-adjusted p-values (M = 8)."
    )


def build_chapter_4(doc):
    add_h1(doc, "Chapter 4: Implementation")

    add_h2(doc, "4.1 System Architecture and Software Environment")
    add_p(
        doc,
        "The ETGNN research platform is implemented as a modular Python software package (`src/`, `configs/`, `tests/`, "
        "and `research_extension/`) executed in a reproducible virtual environment (`.venv`) on Windows (CPU primary "
        "benchmark) and verified on Google Colab (Linux CUDA T4 GPU sensitivity benchmark). Core library versions include "
        "PyTorch (`2.6.0`), PyTorch Geometric (`2.6.1`), Scikit-learn (`1.6.1`), XGBoost (`2.1.4`), NetworkX, FastAPI, "
        "Streamlit, and ReportLab."
    )

    add_h2(doc, "4.2 Data Engineering and Leakage-Prevention Implementation")
    add_p(
        doc,
        "The raw Elliptic CSV files (`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`) "
        "are processed through a deterministic six-stage pipeline in `src/data/`:\n"
        "  1. `create_id_map.py`: Maps raw transaction IDs to contiguous zero-based integer node indices {0, ..., 203768}.\n"
        "  2. `map_labels.py`: Encodes class '1' (illicit) -> 1, class '2' (licit) -> 0, and 'unknown' -> -1.\n"
        "  3. `create_masks.py`: Generates boolean masks for `train_mask` (steps 1–32), `val_mask` (steps 33–40), and "
        "`test_mask` (steps 41–49).\n"
        "  4. `scale_features.py`: Fits `StandardScaler` strictly on `features[train_rows]` (steps 1–32) and serializes "
        "`data/processed/scaled_features.npy` and `data/processed/nodes_with_masks.csv`.\n"
        "  5. `create_visible_edges.py` & `check_edge_times.py`: Verifies that 100% of the 234,355 directed edges connect "
        "nodes within the exact same time step (zero cross-step leakage).\n"
        "  6. `build_stage_graphs.py`: Serializes PyTorch Geometric `Data` objects `train_graph.pt`, `validation_graph.pt`, "
        "and `test_graph.pt` in `data/processed/`."
    )

    add_h2(doc, "4.3 Model Training Scripts and Cryptographic Checkpoint Inventory")
    add_p(
        doc,
        "Individual training scripts in `src/training/` (`train_random_forest.py`, `train_rf_with_time.py`, `train_rf_local.py`, "
        "`train_xgboost.py`, `train_gcn.py`, `train_graphsage.py`, `train_gat.py`, `train_matched_gat.py`, `train_etgnn.py`, "
        "`train_etgnn_local.py`, `train_etgnn_local_no_edges.py`), Phase 3 MLP scripts (`results/optimization_experiments/checkpoints/`), "
        "and the isolated Phase 9 Logistic Regression script (`research_extension/phase9/scripts/train_logistic_regression.py`) "
        "produce the 15 verified local checkpoints inventoried in Table 4.1. All 11 original historical checkpoints reproduce "
        "`data/processed/final_test_probabilities.npz` (`model_0` through `model_10`) within floating-point precision (< 1e-6)."
    )

    df_master = pd.read_csv(P10_DIR / "PHASE10_MASTER_LOCAL_METRICS.csv")
    headers_ckpt = ["#", "Model Name", "Feature Space", "Edges", "Params", "Checkpoint Path", "Bytes", "SHA-256 Prefix"]
    rows_ckpt = []
    for idx, r in enumerate(df_master.itertuples(), start=1):
        rows_ckpt.append([
            str(idx),
            r.Model,
            r.Feature_Space,
            r.Uses_Graph_Edges,
            str(r.Parameters),
            r.Checkpoint_Path,
            f"{int(r.Checkpoint_Bytes):,}",
            str(r.SHA256)[:16] + "..."
        ])
    add_tbl(
        doc, headers_ckpt, rows_ckpt,
        col_widths=[0.3, 1.35, 0.75, 0.65, 0.65, 1.5, 0.55, 0.75],
        caption="Table 4.1: Cryptographic Provenance Inventory of All 15 Verified Local Model Checkpoints.",
        font_size_pt=8.0
    )

    add_h2(doc, "4.4 Explainability and Bipartite Matching Engine Implementation")
    add_p(
        doc,
        "The C1 and C2 interventional explainability pipeline is implemented in `src/evaluation/run_main_c1_c2_benchmark.py` "
        "(65,944 bytes, SHA-256 `ef67f742...`) and configured via `configs/main_c1_c2_run_config.json`. Key engineering "
        "safeguards include:\n"
        "  • Two-Hop Subgraph Forward-Pass Equivalence (`src/evaluation/verify_subgraph_equivalence.py`): Asserts at runtime "
        "that extracting the 2-hop directed subgraph around target node v reproduces full-graph softmax probabilities within "
        "<= 1e-5 across all 377 evaluated validation nodes (empirical max discrepancy: 3.58e-7 for ETGNN, 8.94e-8 for Matched GAT).\n"
        "  • Deterministic Hopcroft-Karp Bipartite Matching: Constructs bipartite eligibility graphs within each validation "
        "snapshot t in {37..40} enforcing the confidence-margin caliper (<= 0.050), candidate-edge caliper (<= 5), and exact "
        "out-degree constraint, solving for maximum-cardinality 1-to-1 matchings without replacement.\n"
        "  • Resume-Safe Checkpointing and Cluster Bootstrapping: Serializes per-node intervention records (`detailed_intervention_records.csv`, "
        "39,585 trials) and executes B = 2,000 cluster-bootstrap resamples (`seed = 20261020`)."
    )

    add_h2(doc, "4.5 Interactive Compliance Demonstration System (`src/demo/`)")
    add_p(
        doc,
        "To translate the research artifacts into a functional forensic inspection tool, a decoupled two-tier web application "
        "is implemented in `src/demo/` and `src/web/`:\n"
        "  • FastAPI Backend (`src/demo/backend.py`): Exposes REST endpoints `/health`, `/examples`, and `/transactions/{tx_id}`, "
        "serving real-time comparative probabilities (Random Forest vs. ETGNN), local vs. neighborhood feature summaries, "
        "and precomputed 2-hop GNNExplainer subgraphs, returning HTTP 404 on invalid transaction IDs.\n"
        "  • Streamlit Compliance Dashboard (`src/demo/frontend.py`): Renders side-by-side model risk gauges, interactive "
        "subgraph visualizations, and methodological disclosure banners warning analysts against interpreting edge masks as "
        "causal proof of illicit activity.\n"
        "  • Automated Quality Assurance (`tests/` and `src/evaluation/test_demo_http_server.py`): Validated by 24 automated "
        "pytest unit, integration, and live HTTP server tests."
    )


def build_chapter_5(doc):
    add_h1(doc, "Chapter 5: Results and Discussion")

    add_h2(doc, "5.1 Authoritative Local Classification Benchmark (15 Verified Models)")
    add_p(
        doc,
        "Table 5.1 presents the authoritative held-out test split evaluation (N = 9,973 labelled transactions across time "
        "steps 41–49: 524 illicit [5.25%], 9,449 licit [94.75%]) across all 15 verified local models, ranked by the primary "
        "thesis metric, Average Precision (`AP_Primary`, `average_precision_score`). Trapezoidal PR-AUC (`PR_AUC_Trapz`, "
        "`auc(recall, precision)`) is reported in an adjacent column to illustrate why the two estimators must never be "
        "conflated—most notably for Historical CPU `XGBoost (165f Full)`, where probability ties inflate Trapezoidal PR-AUC "
        "to 0.5857 compared to its true Average Precision of 0.5472 (Delta = -0.0384)."
    )

    df_master = pd.read_csv(P10_DIR / "PHASE10_MASTER_LOCAL_METRICS.csv")
    headers_m = [
        "Rank", "Model", "Feat Space", "Edges", "Primary AP", "Trapz PR-AUC",
        "ROC-AUC", "Prec@0.5", "Rec@0.5", "F1@0.5", "TP / FP / TN / FN"
    ]
    rows_m = []
    for idx, r in enumerate(df_master.itertuples(), start=1):
        rows_m.append([
            str(idx),
            r.Model,
            r.Feature_Space,
            r.Uses_Graph_Edges,
            f"{r.AP_Primary:.4f}",
            f"{r.PR_AUC_Trapz:.4f}",
            f"{r.ROC_AUC:.4f}",
            f"{r.Precision_0_5 if hasattr(r, 'Precision_0_5') else getattr(r, '_14'):.4f}",
            f"{r.Recall_0_5 if hasattr(r, 'Recall_0_5') else getattr(r, '_15'):.4f}",
            f"{r.F1_0_5 if hasattr(r, 'F1_0_5') else getattr(r, '_16'):.4f}",
            f"{int(r.TP)}/{int(r.FP)}/{int(r.TN)}/{int(r.FN)}"
        ])
    add_tbl(
        doc, headers_m, rows_m,
        col_widths=[0.35, 1.35, 0.65, 0.55, 0.55, 0.6, 0.55, 0.5, 0.5, 0.5, 0.9],
        caption="Table 5.1: Authoritative Held-Out Test Split Benchmark Across All 15 Verified Local Models (Steps 41–49, N = 9,973).",
        font_size_pt=7.5
    )

    add_h2(doc, "5.2 Analysis of RQ1: Tabular Tree Supremacy vs. Graph Neural Networks")
    add_p(
        doc,
        "The empirical results in Table 5.1 refute Hypothesis H1 and provide a clear, evidence-bounded answer to RQ1: "
        "non-relational decision tree ensembles decisively outperform all evaluated Graph Neural Networks on the held-out "
        "Elliptic test period. `Random Forest (165f Full)` achieves the highest Average Precision of 0.6323 (F1 = 0.6319, "
        "Precision = 69.66%, Recall = 57.82%, ROC-AUC = 0.8735), followed closely by `Random Forest (93f Local)` at 0.6301 AP "
        "(F1 = 0.5736) and `RF + Time (165f + 16t)` at 0.6295 AP (F1 = 0.6173). `XGBoost (165f Full)` ranks fourth overall "
        "at 0.5472 AP (F1 = 0.3849). By contrast, the strongest message-passing GNN is `GraphSAGE (165f Full)` at 0.2427 AP "
        "(F1 = 0.2125), followed by `ETGNN (93f Local)` at 0.2344 AP (F1 = 0.3132), `ETGNN (165f Full)` at 0.2250 AP "
        "(F1 = 0.3308), `Matched GAT (165f Full)` at 0.2099 AP (F1 = 0.2074), `GCN (165f Full)` at 0.1306 AP (F1 = 0.1808), "
        "and `GAT Baseline (165f Full)` at 0.1155 AP (F1 = 0.1627)."
    )
    add_p(
        doc,
        "Examining the confusion-matrix columns (`TP / FP / TN / FN`) reveals the mechanism of GNN underperformance: "
        "under class-weighted cross-entropy, standard GNNs achieve high recall (`GAT Baseline`: 91.41% recall, 479/524 TP; "
        "`GCN`: 86.45% recall, 453/524 TP; `Matched GAT`: 74.24% recall, 389/524 TP) at the cost of massive false-positive "
        "inflation (`4,885`, `4,035`, and `2,839` false alarms, respectively). `ETGNN (165f Full)` moderates false alarms "
        "to 667 FP (precision 26.14%, F1 0.3308), but its ranking precision across thresholds (0.2250 AP) remains 0.4072 AP "
        "points below `Random Forest (165f Full)`."
    )
    add_img(
        doc, "pr_curves.png",
        "Figure 5.1: Precision–Recall Curves on Held-Out Test Steps 41–49 Demonstrating Tabular Ensemble Dominance over Message-Passing GNNs.",
        width=5.3
    )

    add_h2(doc, "5.3 Feature-Subspace Ablations (93f Local vs. 165f Full) and Topological Edge Ablation")
    add_p(
        doc,
        "Comparing `93f Local` (intrinsic transaction features 1–93) against `165f Full` (adding precomputed 1-hop neighborhood "
        "aggregates 94–165) across model families exposes a striking divergence between tree ensembles and continuous neural/linear models:\n"
        "  • Random Forest Is Invariant to Precomputed Aggregates: `Random Forest (93f Local)` achieves 0.6301 AP vs. 0.6323 AP "
        "for `Random Forest (165f Full)` (Delta = +0.0022 AP), showing that 99.66% of Random Forest's ranking precision "
        "originates purely from local transaction attributes 1–93, while column subsampling prevents noisy neighborhood "
        "aggregates from degrading tree splits (and improves threshold precision at tau = 0.50 from 56.35% to 69.66%).\n"
        "  • Linear and Feed-Forward Neural Models Degrade When 1-Hop Aggregates Are Added: Adding features 94–165 reduces "
        "held-out test AP in `Logistic Regression` from 0.2512 (`93f Local`) to 0.1905 (`165f Full`, Delta = -0.0607 AP) "
        "and reduces `DeepParameterMatchedMLP (s42)` from 0.4662 (`93f Local`) to 0.3603 (`165f Full`, Delta = -0.1060 AP). "
        "Table 5.2 reconciles all three random seeds (42, 43, 44) of `DeepParameterMatchedMLP` across both the Validation "
        "Split (N = 7,653, steps 33–40, originally logged in `PHASE3_ABLATION_MATRIX.csv`) and the Held-Out Test Split "
        "(N = 9,973, steps 41–49), confirming that the 93f MLP outperforms the 165f MLP across every seed (3-seed Test AP "
        "mean: 0.4934 +/- 0.0306 for 93f vs. 0.3336 +/- 0.0283 for 165f).\n"
        "  • Single-Variable Topological Edge Ablation (`ETGNN 93f No Edges` vs. `ETGNN 93f Local`): Holding the architecture, "
        "109-dimensional input (`93f + 16t`), and 7,302 parameters completely identical, training ETGNN without graph edges "
        "(`ETGNN 93f Local No Edges`, E = 0) increases test Average Precision from 0.2344 to 0.4812 (Delta_AP = +0.2468, "
        "+105.3%). As established in Chapter 3, this provides strong descriptive empirical evidence consistent with the "
        "hypothesis that unconstrained 2-layer GAT message passing across Elliptic's payment edges dilutes discriminative "
        "local transaction signatures."
    )

    df_mlp = pd.read_csv(P10_DIR / "PHASE10_MLP_MULTISEED_VAL_VS_TEST.csv")
    headers_mlp = ["MLP Condition", "Seed", "Val AP (N=7,653)", "Val F1@0.5", "Val ROC-AUC", "Test AP Primary (N=9,973)", "Test Trapz PR-AUC", "Test F1@0.5", "Test ROC-AUC"]
    rows_mlp = []
    for r in df_mlp.itertuples():
        rows_mlp.append([
            r.MLP_Condition, str(r.Seed),
            f"{r.Val_AP_from_JSON:.4f}", f"{r.Val_F1_from_JSON:.4f}", f"{r.Val_ROCAUC_from_JSON:.4f}",
            f"{r.Test_AP_Primary:.4f}", f"{r.Test_PR_AUC_Trapz:.4f}", f"{r.Test_F1_0_5 if hasattr(r, 'Test_F1_0_5') else getattr(r, '_8'):.4f}", f"{r.Test_ROCAUC:.4f}"
        ])
    rows_mlp.append(["165f Full (Mean +/- SD)", "3 seeds", "0.7196 +/- 0.0018", "0.3380 +/- 0.0302", "0.9380 +/- 0.0027", "0.3336 +/- 0.0283", "0.3326 +/- 0.0284", "0.1937 +/- 0.0152", "0.8226 +/- 0.0018"])
    rows_mlp.append(["93f Local (Mean +/- SD)", "3 seeds", "0.7888 +/- 0.0163", "0.4969 +/- 0.0423", "0.9457 +/- 0.0023", "0.4934 +/- 0.0306", "0.4924 +/- 0.0306", "0.4037 +/- 0.0246", "0.8076 +/- 0.0047"])
    add_tbl(
        doc, headers_mlp, rows_mlp,
        col_widths=[1.15, 0.55, 0.8, 0.65, 0.7, 0.85, 0.75, 0.6, 0.65],
        caption="Table 5.2: Multi-Seed Deep MLP (`P3-05`) Validation Split (N = 7,653) vs. Held-Out Test Split (N = 9,973) Reconciliation.",
        font_size_pt=8.0
    )
    add_img(
        doc, "ablation_graph_vs_features.png",
        "Figure 5.2: Feature and Topological Edge Ablation Comparison Across Local (93f) and Full (165f) Representations.",
        width=5.0
    )

    add_h2(doc, "5.4 Analysis of RQ2: Sinusoidal Temporal Encoding (`ETGNN` vs. `Matched GAT`)")
    add_p(
        doc,
        "Evaluating RQ2 requires examining both the 3-seed validation ablation (steps 33–40) and the held-out test split "
        "(steps 41–49):\n"
        "  • Validation Multi-Seed Comparison (`research_log.md`): Across seeds 7, 42, and 123, `ETGNN (165f Full)` achieved "
        "validation APs of 0.5096, 0.5488, and 0.5293 (mean 0.5292 +/- 0.0196) compared to `Matched GAT (165f Full)` at "
        "0.5572, 0.4836, and 0.3934 (mean 0.4781 +/- 0.0820). While ETGNN achieved a +0.0511 higher three-seed validation "
        "mean, Matched GAT scored higher on Seed 7 (+0.0476).\n"
        "  • Held-Out Test Comparison: On the N = 9,973 test nodes, `ETGNN (165f Full)` achieved 0.2250 AP (F1 = 0.3308) "
        "versus 0.2099 AP (F1 = 0.2074) for `Matched GAT (165f Full)`, a positive point gain of Delta_AP = +0.0151 "
        "(and +0.1234 in F1 at tau = 0.50 due to reducing false positives from 2,839 to 667). However, as shown in Table 5.3 "
        "(Contrast C01), the node-level paired bootstrap 95% CI spans zero ([-0.0085, +0.0381], exact p = 0.2149) and "
        "does not survive Holm-Bonferroni correction. Similarly, adding sinusoidal time encodings to Random Forest (`RF + Time`, "
        "0.6295 AP) yielded no gain over standard `Random Forest (165f Full)` (0.6323 AP). Thus, static sinusoidal snapshot "
        "encoding provides a modest, statistically inconclusive ranking lift on extrapolated future time steps."
    )

    add_h2(doc, "5.5 Paired Bootstrap & Time-Step-Clustered Sensitivity Analysis (Steps 41–49)")
    add_p(
        doc,
        "Table 5.3 reports all M = 12 pre-specified paired model comparisons evaluated via B = 2,000 node-level paired "
        "bootstraps on Primary AP (`AP_Primary`), exact Phipson–Smyth Monte Carlo p-values, monotonic Holm-Bonferroni "
        "correction, and the T = 9 Time-Step-Clustered Sensitivity Analysis across test steps 41–49."
    )

    df_stat = pd.read_csv(P10_DIR / "PHASE10_STATISTICAL_COMPARISONS.csv")
    headers_st = [
        "Rank", "Contrast (Model A vs. Model B)", "Delta AP", "Node 95% Boot CI",
        "Exact p", "Holm alpha", "Holm Sig?", "Step Wins", "Cluster 95% Block CI", "Wilcoxon p"
    ]
    rows_st = []
    for r in df_stat.itertuples():
        rows_st.append([
            str(r.Holm_Rank),
            r.Contrast_ID,
            f"{r.Delta_AP:+.4f}",
            f"[{r.Node_CI95_Low:+.4f}, {r.Node_CI95_High:+.4f}]",
            f"{r.Node_P_Value:.4f}",
            f"{r.Holm_Alpha:.4f}",
            "YES" if r.Holm_Significant else "NO",
            str(r.Steps_A_Wins_of_9),
            f"[{r.Cluster_CI95_Low:+.4f}, {r.Cluster_CI95_High:+.4f}]",
            f"{r.Step_Wilcoxon_Pval:.4f}"
        ])
    add_tbl(
        doc, headers_st, rows_st,
        col_widths=[0.35, 1.65, 0.55, 1.0, 0.5, 0.5, 0.45, 0.5, 1.0, 0.5],
        caption="Table 5.3: Paired Node-Level Bootstrap (B = 2,000, Primary AP) and Time-Step-Clustered Sensitivity Analysis (T = 9 Steps).",
        font_size_pt=7.5
    )

    add_p(
        doc,
        "To explain why 9 of the 12 contrasts are highly significant under pooled node-level bootstrap (p <= 0.0030) while "
        "exhibiting wide intervals under T = 9 time-step block bootstrap, Table 5.4 disaggregates Average Precision across "
        "each individual test snapshot (t in {41..49}). Crucially, 355 of the 524 illicit test transactions (67.75%) reside "
        "in pre-shutdown steps 41 (116 illicit, 10.25% prevalence) and 42 (239 illicit, 11.10% prevalence). On steps 41 and "
        "42, `Random Forest (165f)` achieves 0.9583 and 0.8857 AP (`XGBoost`: 0.9523 and 0.8570 AP; `ETGNN 93f No Edges`: "
        "0.7711 and 0.7950 AP; `MLP 93f`: 0.7223 and 0.7319 AP), decisively outperforming `ETGNN (165f)` (0.3563 and 0.6413 AP) "
        "and `Matched GAT` (0.2590 and 0.5661 AP). Following the AlphaBay takedown at step 43, illicit prevalence collapses "
        "to 1.75% (step 43), 0.41% (step 45, 5 illicit), and 0.28% (step 46, 2 illicit), causing severe finite-sample score "
        "volatility across all paradigms (`C09: Random Forest 93f vs. ETGNN 93f` remains strictly positive in both the node "
        "bootstrap [+0.3623, +0.4271] and the time-step block bootstrap [+0.0006, +0.5032])."
    )

    df_ts = pd.read_csv(P10_DIR / "PHASE10_PER_TIMESTEP_AP.csv")
    headers_ts = [
        "Step", "N Lab", "Illicit (%)", "RF 165f", "RF 93f", "XGB 165f",
        "NoEdge 93f", "MLP 93f", "LogReg 93f", "SAGE 165f", "ETGNN 93f", "ETGNN 165f", "MGAT 165f"
    ]
    rows_ts = []
    for _, d in df_ts.iterrows():
        rows_ts.append([
            str(int(d["time_step"])),
            f"{int(d['N_labeled']):,}",
            f"{int(d['N_illicit'])} ({d['prevalence']*100:.2f}%)",
            f"{d['AP_Random Forest (165f Full)']:.4f}",
            f"{d['AP_Random Forest (93f Local)']:.4f}",
            f"{d['AP_XGBoost (165f Full)']:.4f}",
            f"{d['AP_ETGNN (93f Local No Edges)']:.4f}",
            f"{d['AP_MLP (93f Local, s42)']:.4f}",
            f"{d['AP_Logistic Regression (93f Local)']:.4f}",
            f"{d['AP_GraphSAGE (165f Full)']:.4f}",
            f"{d['AP_ETGNN (93f Local)']:.4f}",
            f"{d['AP_ETGNN (165f Full)']:.4f}",
            f"{d['AP_Matched GAT (165f Full)']:.4f}",
        ])
    add_tbl(
        doc, headers_ts, rows_ts,
        col_widths=[0.35, 0.45, 0.75, 0.5, 0.5, 0.5, 0.55, 0.5, 0.55, 0.5, 0.5, 0.5, 0.5],
        caption="Table 5.4: Per-Time-Step Average Precision Trajectory Across Held-Out Test Steps 41–49 (Showing Step 43 Regime Shift).",
        font_size_pt=7.0
    )
    add_img(
        doc, "test_auprc_by_time_step.png",
        "Figure 5.3: Held-Out Test Average Precision by Time Step Alongside Illicit Prevalence Collapse After Step 42.",
        width=5.4
    )

    add_h2(doc, "5.6 Google Colab GPU Reproducibility and Cross-Environment Sensitivity")
    add_p(
        doc,
        "To audit cross-environment reproducibility on the authentic Elliptic dataset, the 7 full-feature (`165f`) models "
        "were independently retrained from scratch on a Google Colab Linux CUDA T4 GPU runtime (`ETGNN_Colab_Results_20261103.zip`, "
        "`results/colab_extracted_20261103/`). Table 5.5 reconciles the Historical Local CPU benchmark against the Colab GPU "
        "run under identical `AP_Primary` and `PR_AUC_Trapz` definitions."
    )

    df_col = pd.read_csv(P10_DIR / "PHASE10_LOCAL_VS_COLAB_RECONCILIATION.csv")
    headers_col = [
        "Model (165f Full)", "Local CPU AP", "Colab GPU AP", "Delta AP",
        "Local Trapz", "Colab Trapz", "Local F1", "Colab F1", "Pearson r", "Spearman rho"
    ]
    rows_col = []
    for r in df_col.itertuples():
        rows_col.append([
            r.Model,
            f"{r.Local_AP:.4f}",
            f"{r.Colab_AP:.4f}",
            f"{r.Delta_AP_Colab_minus_Local:+.4f}",
            f"{r.Local_Trapz_PRAUC:.4f}",
            f"{r.Colab_Trapz_PRAUC:.4f}",
            f"{r.Local_F1:.4f}",
            f"{r.Colab_F1:.4f}",
            f"{r.Pearson_r:.4f}",
            f"{r.Spearman_rho:.4f}"
        ])
    add_tbl(
        doc, headers_col, rows_col,
        col_widths=[1.3, 0.6, 0.6, 0.55, 0.6, 0.6, 0.55, 0.55, 0.55, 0.6],
        caption="Table 5.5: Reconciled Historical Local CPU vs. Google Colab GPU Benchmark (N = 9,973 Test Split).",
        font_size_pt=7.5
    )
    add_p(
        doc,
        "Three scientific conclusions emerge from Table 5.5:\n"
        "  1. Macro-Hierarchy Reproducibility: In both environments, tabular tree ensembles (`Random Forest` 0.6380 AP and "
        "`XGBoost` 0.6445 AP on Colab) outperform all GNN architectures (0.1670–0.3033 AP) by more than +0.33 AP points, "
        "and `ETGNN` outperforms `Matched GAT` in both runs (+0.0151 AP on CPU; +0.0483 AP on Colab GPU).\n"
        "  2. Single-Seed Optimization Sensitivity: While `Random Forest` is highly deterministic across platforms (Delta_AP = +0.0057, "
        "r = 0.9852), `XGBoost` improved from 0.5472 AP (CPU) to 0.6445 AP (Colab GPU) due to histogram binning/validation "
        "stopping differences, and `GAT Baseline` improved from 0.1155 AP (where historical CPU training early-stopped at "
        "epoch 3) to 0.2417 AP (epoch 11 in Colab, exceeding Colab ETGNN's 0.2153 AP).\n"
        "  3. Strict Separation of Cohorts: Because all C1/C2 explainability logs (`results/main_c1_c2_results/`) are "
        "cryptographically bound to the Historical CPU checkpoints (`etgnn.pt` SHA-256 `b2f5535f...` and `matched_gat.pt` "
        "SHA-256 `1e9b43ea...`) and the Colab run omitted the 8 local/ablation models, Historical CPU is maintained strictly "
        "as the Primary Thesis Benchmark and Colab GPU as a secondary reproducibility audit."
    )

    add_h2(doc, "5.7 Explainability Audit: Subgraph Sparsity and Ceiling Budget Inflation")
    add_p(
        doc,
        "Auditing the 2-hop candidate edge counts E_cand across the N = 377 unique validation transactions evaluated in "
        "Protocols C1 and C2 (`results/tables/candidate_edge_budget_summary.csv`) reveals severe structural sparsity: "
        "317 of 377 transactions (84.08%) possess exactly E_cand = 2 candidate edges (26 nodes [6.90%] have E_cand = 3; "
        "18 [4.77%] have E_cand = 4; 16 [4.24%] have E_cand >= 5). Under the ceiling budget rule k = max(1, ceil(0.25 * E_cand)), "
        "E_cand = 2 yields k = 1 deleted edge, inflating the actual deletion fraction to 1/2 = 50.0% (2.0x the nominal "
        "25% budget; overall mean actual deletion fraction = 47.08%, median = 50.0%). Furthermore, when E_cand = 2 and k = 1, "
        "there are only 2 possible single-edge subsets, so random deletion draws collide with the GNNExplainer top-1 edge "
        "50% of the time."
    )
    add_img(
        doc, "candidate_edges_deletion_budget.png",
        "Figure 5.4: Candidate Edge Count Distribution (84.08% at E_cand = 2) and Actual Deletion Fraction Inflation.",
        width=5.3
    )

    add_h2(doc, "5.8 Protocol C1 Results: Shared-Intervention Model Sensitivity (N = 240)")
    headers_c1 = ["Cohort", "N", "Mean Delta_adj", "Sample SD (s)", "Median Delta_adj", "IQR", "Mean Delta_raw", "ETGNN Adv", "Matched GAT Adv"]
    rows_c1 = [
        ["Overall C1 Cohort", "240", "+0.002903", "+/-0.096180", "-0.001534", "0.045904", "+0.038491", "+0.020239", "+0.017336"],
        ["True Positives (TP)", "95", "-0.005973", "+/-0.132491", "-0.001400", "0.044846", "+0.040412", "-0.003310", "+0.002663"],
        ["False Positives (FP)", "95", "+0.011367", "+/-0.051097", "-0.000205", "0.041682", "+0.040085", "+0.036034", "+0.024667"],
        ["True Negatives (TN)", "25", "+0.013444", "+/-0.085633", "0.000000", "0.052547", "+0.044734", "+0.063984", "+0.050539"],
        ["False Negatives (FN)", "25", "-0.006073", "+/-0.070797", "-0.017843", "0.065701", "+0.018893", "+0.005957", "+0.012030"]
    ]
    add_tbl(
        doc, headers_c1, rows_c1,
        col_widths=[1.25, 0.4, 0.75, 0.75, 0.75, 0.6, 0.75, 0.6, 0.65],
        caption="Table 5.6: Protocol C1 Shared-Intervention Response Summary Across N = 240 Validation Transactions (Steps 37–40).",
        font_size_pt=8.0
    )
    add_p(
        doc,
        "In Protocol C1 (Table 5.6), deleting the shared top-k edge subset selected by ETGNN's GNNExplainer produces a "
        "small positive random-adjusted sensitivity in both models, with a paired cross-model mean difference of "
        "Delta_adj = +0.002903 (cross-transaction sample standard deviation s = +/-0.096180, median = -0.001534, exploratory "
        "95% bootstrap CI [-0.009201, +0.015121], two-sided p = 0.6540). Because the 95% CI spans zero, C1 is statistically "
        "inconclusive: it neither detects a significant sensitivity difference nor proves formal model equivalence, and must "
        "be interpreted alongside the ranking asymmetry that shared edges were selected by ETGNN's explainer."
    )
    add_img(
        doc, "c1_response_distributions.png",
        "Figure 5.5: Protocol C1 Random-Adjusted (Delta_adj) and Raw (Delta_raw) Response Distributions Across N = 240 Nodes.",
        width=5.3
    )

    add_h2(doc, "5.9 Protocol C2 Results: Matched Correct-vs-Incorrect Error Registers (N = 420 Pairs)")
    headers_c2 = [
        "Register", "Model", "Matching Rule", "Pairs", "Mean Delta_pair",
        "Boot SE", "Marginal 95% Boot CI", "CI Excl 0?", "p_raw", "p_Holm (M=8)", "Final Audit Status"
    ]
    rows_c2 = [
        ["ETGNN Primary TP-FP", "ETGNN", "Primary Caliper", "95", "+0.039343", "0.009585", "[+0.020234, +0.057410]", "YES", "0.0005", "0.0040", "Robust Significant"],
        ["ETGNN Degree TP-FP", "ETGNN", "Exact Out-Degree", "57", "+0.043455", "0.012658", "[+0.019882, +0.069552]", "YES", "0.0010", "0.0070", "Robust Significant"],
        ["Matched GAT Primary TP-FP", "Matched GAT", "Primary Caliper", "94", "+0.020315", "0.008923", "[+0.003917, +0.038684]", "YES", "0.0160", "0.0960", "Marginal Only / Not Robust"],
        ["Matched GAT Degree TP-FP", "Matched GAT", "Exact Out-Degree", "82", "+0.010547", "0.011610", "[-0.013240, +0.032770]", "NO", "0.3718", "1.0000", "Inconclusive (Spans 0)"],
        ["ETGNN Primary TN-FN", "ETGNN", "Primary Caliper", "25", "-0.058027", "0.034398", "[-0.128831, +0.006393]", "NO", "0.2589", "1.0000", "Underpowered / Null"],
        ["ETGNN Degree TN-FN", "ETGNN", "Exact Out-Degree", "25", "-0.019748", "0.029750", "[-0.081593, +0.036312]", "NO", "0.3128", "1.0000", "Underpowered / Null"],
        ["Matched GAT Primary TN-FN", "Matched GAT", "Primary Caliper", "21", "+0.052255", "0.051517", "[-0.046153, +0.162460]", "NO", "0.8066", "1.0000", "Underpowered / Null"],
        ["Matched GAT Degree TN-FN", "Matched GAT", "Exact Out-Degree", "21", "+0.072489", "0.051881", "[-0.024721, +0.178587]", "NO", "0.8526", "1.0000", "Underpowered / Null"]
    ]
    add_tbl(
        doc, headers_c2, rows_c2,
        col_widths=[1.1, 0.65, 0.75, 0.35, 0.6, 0.5, 0.95, 0.45, 0.4, 0.45, 0.8],
        caption="Table 5.7: Protocol C2 Matched-Pair Cluster Bootstrap Results Across All 8 Registers (B = 2,000, Seed 20261020).",
        font_size_pt=7.0
    )
    add_p(
        doc,
        "Table 5.7 establishes a selective, nuanced finding for RQ3(b):\n"
        "  • Robust Positive Sensitivity in ETGNN's Predicted-Illicit (TP–FP) Regime: For ETGNN, matched False Positives "
        "exhibit a significantly larger random-adjusted probability drop under GNNExplainer top-k edge deletion than "
        "confidence-margin-matched True Positives in both the Primary register (N = 95 pairs, Mean = +0.039343, 95% CI "
        "[+0.020234, +0.057410], p_Holm = 0.0040) and the Exact Out-Degree Sensitivity register (N = 57 pairs, Mean = +0.043455, "
        "95% CI [+0.019882, +0.069552], p_Holm = 0.0070). This indicates that ETGNN false alarms rely more fragilely on "
        "top-ranked local edges than confidence-matched true illicit detections.\n"
        "  • Degree Sensitivity in Matched GAT: While Matched GAT's Primary TP–FP register is marginally positive (+0.020315, "
        "unadjusted 95% CI [+0.003917, +0.038684], p_raw = 0.0160), it fails post-hoc Holm-Bonferroni correction (p_Holm = 0.0960) "
        "and attenuates to +0.010547 (95% CI [-0.013240, +0.032770]) under exact out-degree matching.\n"
        "  • Inconclusiveness of All 4 Predicted-Licit (TN–FN) Registers: Due to small sample sizes (N = 25 pairs for ETGNN; "
        "N = 21 pairs for Matched GAT), all four TN–FN registers have wide confidence intervals spanning zero and are "
        "statistically inconclusive."
    )
    add_img(
        doc, "c2_paired_differences_ci.png",
        "Figure 5.6: Protocol C2 Forest Plot of Matched-Pair Differences and 95% Cluster-Bootstrap Confidence Intervals Across All 8 Registers.",
        width=5.4
    )
    add_img(
        doc, "explanation_tp_subgraph.png",
        "Figure 5.7: Representative 2-Hop Computation Subgraph and GNNExplainer Edge Mask for a True Positive Transaction (ID: 288717023).",
        width=4.5
    )
    add_img(
        doc, "explanation_fp_subgraph.png",
        "Figure 5.8: Representative 2-Hop Computation Subgraph and GNNExplainer Edge Mask for a False Positive Transaction (ID: 116898498).",
        width=4.5
    )


def build_chapter_6(doc):
    add_h1(doc, "Chapter 6: Conclusion and Future Work")

    add_h2(doc, "6.1 Evidence-Based Answers to Approved Research Questions")
    add_p(
        doc,
        "No—message-passing GNNs do not outperform tabular baselines on the Elliptic benchmark. On held-out test steps "
        "41–49 (N = 9,973), `Random Forest (165f Full)` achieves 0.6323 Average Precision (F1 = 0.6319), `Random Forest (93f Local)` "
        "achieves 0.6301 AP, `RF + Time` achieves 0.6295 AP, and `XGBoost (165f Full)` achieves 0.5472 AP (Colab GPU: 0.6445 AP), "
        "significantly outperforming `GraphSAGE` (0.2427 AP), `ETGNN (93f Local)` (0.2344 AP), and `ETGNN (165f Full)` (0.2250 AP) "
        "(node-bootstrap p = 0.0010, Holm-significant). Furthermore, removing graph edges in a single-variable ablation "
        "(`ETGNN 93f No Edges`, 7,302 params) increases test AP from 0.2344 to 0.4812 (+0.2468 AP), and non-relational "
        "`DeepParameterMatchedMLP (93f Local)` achieves 0.4662 AP (3-seed mean 0.4934 AP), providing strong empirical "
        "evidence that unconstrained 1-hop neighborhood aggregation across Elliptic's payment hubs dilutes discriminative "
        "local transaction features.",
        bold_prefix="• Answer to RQ1 (Baseline Comparison & Feature/Topology Ablation): "
    )
    add_p(
        doc,
        "Sinusoidal snapshot time encoding (`PE(t)`) provides a modest positive point improvement over the parameter-matched "
        "zero-time GAT control (11,910 params) across the 3-seed validation mean (0.5292 vs. 0.4781 AP), the Historical CPU "
        "test split (0.2250 vs. 0.2099 AP, Delta_AP = +0.0151; F1 = 0.3308 vs. 0.2074), and the Colab GPU test split "
        "(0.2153 vs. 0.1670 AP, Delta_AP = +0.0483). However, the pooled test AP contrast does not reach statistical "
        "significance (`Node 95% CI [-0.0085, +0.0381]`, exact p = 0.2149; `Time-Step Block 95% CI [-0.0725, +0.1095]`, "
        "Wilcoxon p = 0.2031). Therefore, static sinusoidal snapshot encoding offers a minor calibration and F1 improvement "
        "but does not confer a statistically significant ranking advantage on unseen future time steps.",
        bold_prefix="• Answer to RQ2 (Temporal Positional Encoding Contribution): "
    )
    add_p(
        doc,
        "In Protocol C1 (N = 240 validation nodes, steps 37–40), deleting shared top-k edges ranked by ETGNN's GNNExplainer "
        "yields a near-zero paired random-adjusted difference between ETGNN and Matched GAT (Delta_adj = +0.002903 +/- 0.096180, "
        "median -0.001534, exploratory 95% CI [-0.009201, +0.015121]), which is statistically inconclusive. In Protocol C2 "
        "(N = 420 matched pairs across 8 registers), ETGNN demonstrates a statistically robust positive difference in the "
        "predicted-illicit (TP–FP) regime under both Primary caliper matching (+0.039343, 95% CI [+0.020234, +0.057410], "
        "p_Holm = 0.0040) and Exact Out-Degree matching (+0.043455, 95% CI [+0.019882, +0.069552], p_Holm = 0.0070), "
        "whereas Matched GAT's TP–FP effect attenuates under exact degree matching (+0.010547, CI [-0.013240, +0.032770]), "
        "and all four predicted-licit (TN–FN) registers are underpowered and statistically null.",
        bold_prefix="• Answer to RQ3 (Explainability Protocols C1 & C2): "
    )

    add_h2(doc, "6.2 Separation of Supported Findings vs. Inconclusive Findings")
    add_p(
        doc,
        "1. Supported Empirical Findings:\n"
        "  • Tabular tree ensembles (`Random Forest` 0.6323 AP, `XGBoost` 0.5472 CPU / 0.6445 Colab AP) significantly "
        "outperform all evaluated GNN architectures on Elliptic.\n"
        "  • Removing runtime graph edges (`ETGNN 93f No Edges` 0.4812 AP vs. `ETGNN 93f` 0.2344 AP) and omitting precomputed "
        "neighborhood aggregates (features 94–165) in `Deep MLP` (0.4662 vs. 0.3603 AP) and `Logistic Regression` (0.2512 vs. "
        "0.1905 AP) substantially improves neural and linear classification performance.\n"
        "  • In Protocol C2, ETGNN's predicted-illicit (TP–FP) registers show a robust positive explanation-minus-random "
        "deletion difference that survives both exact out-degree matching and Holm-Bonferroni correction.\n"
        "2. Statistically Inconclusive / Null Findings:\n"
        "  • The test AP gain of `ETGNN (165f)` over `Matched GAT (165f)` (+0.0151 AP, p = 0.2149) and `GraphSAGE` over "
        "`ETGNN` (+0.0176 AP, p = 0.1709) are statistically indistinguishable from zero.\n"
        "  • Protocol C1 cross-model sensitivity difference (+0.002903, 95% CI [-0.0092, +0.0151]) is null.\n"
        "  • Protocol C2 Matched GAT Degree-Sensitivity TP–FP and all four TN–FN registers span zero."
    )

    add_h2(doc, "6.3 Explicit Methodological and Empirical Limitations")
    add_p(
        doc,
        "In compliance with Phase 10 and Phase 11 scientific integrity gates, eight material limitations are explicitly "
        "documented:\n"
        "  1. Weak / Statistically Inconclusive Temporal Encoding Gain: Concatenating sinusoidal snapshot encodings yields "
        "only +0.0151 test AP over the zero-time Matched GAT control (p = 0.2149), as unseen test steps 41–49 extrapolate "
        "beyond the training step index range 1–32.\n"
        "  2. Snapshot-Level Temporal Resolution & Disjoint Node Sets: Because all Elliptic edges are intra-step (Delta t = 0) "
        "and transaction IDs do not persist across steps, ETGNN cannot track multi-step entity trajectories.\n"
        "  3. Candidate-Edge Sparsity & 2.0x Deletion Budget Inflation: Across the 377 C1/C2 transactions, 84.08% (317/377) "
        "have exactly E_cand = 2 candidate edges, causing the ceiling rule k = max(1, ceil(0.25 * E_cand)) = 1 to delete "
        "50.0% of candidate edges and causing random deletions to collide with the top-1 edge 50% of the time.\n"
        "  4. Static Precomputed Neighborhood Features During Edge Deletion: Deleting edges in C1/C2 perturbs runtime GAT "
        "message passing while leaving precomputed 1-hop neighborhood features (columns 94–165) fixed in the node feature matrix.\n"
        "  5. Retrospective Validation-Split Reuse in C1/C2: Protocols C1 and C2 were executed on validation snapshots 37–40 "
        "(used for early stopping) rather than untouched test steps 41–49.\n"
        "  6. C1 Ranking-Model Asymmetry: Shared edge cuts in C1 are selected by ETGNN's GNNExplainer, evaluating Matched GAT "
        "on off-target edges relative to its own explanation masks.\n"
        "  7. Temporal Clustering and Post-Shutdown Class Imbalance: Because 67.75% (355/524) of test illicit nodes occur "
        "in steps 41–42 prior to the step 43 AlphaBay shutdown (and steps 45–46 contain only 5 and 2 illicit nodes), "
        "time-step block bootstrap intervals across T = 9 steps are substantially wider than pooled node-level intervals.\n"
        "  8. Cross-Environment Variance & Missing Local Baselines: Historical CPU and Colab GPU single-seed runs exhibit "
        "optimization sensitivity for `XGBoost` (0.5472 vs. 0.6445 AP) and `GAT Baseline` (0.1155 vs. 0.2417 AP), and "
        "`XGBoost (93f Local)` and `Matched GAT (93f Local)` were not trained in the original repository."
    )

    add_h2(doc, "6.4 Directions for Future Work")
    add_p(
        doc,
        "1. Weight-Evolving Dynamic GNNs (EvolveGCN-O and EvolveGCN-H): Extending our ego-residual and local-feature "
        "findings to weight-evolving architectures (Pareja et al., 2020) coupled with ego-skip projections to test whether "
        "LSTM/GRU weight evolution across snapshots can adapt to the step 43 darknet-market closure without diluting local "
        "transaction features.\n"
        "2. Point-in-Time Entity-Resolved Actor Graphs: Constructing strictly causal, point-in-time wallet interaction "
        "graphs (eliminating the static global lifetime feature leakage identified in raw Elliptic++ `wallets_features.csv`) "
        "where persistent wallet identities enable continuous-time memory networks (TGN; Rossi et al., 2020).\n"
        "3. Joint Feature-and-Edge Perturbation Benchmarks: Recomputing features 94–165 dynamically whenever graph edges "
        "are deleted during post-hoc explanation testing to eliminate static aggregate confounding.\n"
        "4. Exact Leave-One-Out Occlusion on Higher-Degree Subgraphs: Replacing continuous gradient mask relaxation "
        "(GNNExplainer) with fast exact Leave-One-Out (LOO) edge occlusion on multi-edge subgraphs (E_cand >= 4)."
    )

    add_h2(doc, "6.5 Concluding Remarks")
    add_p(
        doc,
        "By combining leak-free multi-paradigm benchmarking, controlled feature and topology ablations, dual node-level and "
        "time-step-clustered statistical inference, and rigorous C1/C2 interventional explainability audits—while reporting "
        "negative and inconclusive findings with complete transparency—this thesis establishes a scientifically defensible, "
        "reproducible foundation for machine learning and explainable AI in cryptocurrency financial forensics."
    )


def build_references_and_appendices(doc):
    add_h1(doc, "References")
    refs = [
        "Agarwal, C., Queen, O., Lakkaraju, H., & Zitnik, M. (2023). Evaluating explainability for graph neural networks. Scientific Data, 10, 144. https://doi.org/10.1038/s41597-023-01974-x",
        "Amara, K., Ying, Z., Zhang, Z., Han, Z., Zhao, Y., Shan, Y., Brandes, U., Schemm, S., & Zhang, C. (2022). GraphFramEx: Towards systematic evaluation of explainability methods for graph neural networks. Proceedings of the First Learning on Graphs Conference (LoG), Proceedings of Machine Learning Research (PMLR), 198:44:1-44:23. https://proceedings.mlr.press/v198/amara22a.html",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), 785-794. https://doi.org/10.1145/2939672.2939785",
        "Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. Proceedings of the 23rd International Conference on Machine Learning (ICML '06), 233-240. https://doi.org/10.1145/1143844.1143874",
        "Elmougy, Y., & Liu, L. (2023). Demystifying fraudulent transactions and illicit nodes in the Bitcoin network for financial forensics. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23), 3979-3990. https://doi.org/10.1145/3580305.3599803",
        "Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 1024-1034. https://proceedings.neurips.cc/paper/6703-inductive-representation-learning-on-large-graphs",
        "Hooker, S., Erhan, D., Kindermans, P. J., & Kim, B. (2019). A benchmark for interpretability methods. Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 9737-9748. https://proceedings.neurips.cc/paper/2019/hash/fe4b8555ed8f8ab149aa792f6da600f4-Abstract.html",
        "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. 5th International Conference on Learning Representations (ICLR 2017). https://arxiv.org/abs/1609.02907",
        "Luo, D., Cheng, W., Xu, D., Yu, F., Zong, B., Chen, H., & Zhang, X. (2020). Parameterized explainer for graph neural networks. Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 19620-19631.",
        "Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T. B., & Leiserson, C. E. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04), 5363-5370. https://doi.org/10.1609/aaai.v34i04.5984",
        "Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: Calculating exact P-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology, 9(1), Article 39. https://doi.org/10.2202/1544-6115.1585",
        "Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal graph networks for deep learning on dynamic graphs. ICML 2020 Workshop on Graph Representation Learning and Beyond, arXiv:2006.10637. https://arxiv.org/abs/2006.10637",
        "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998-6008.",
        "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. 6th International Conference on Learning Representations (ICLR 2018). https://arxiv.org/abs/1710.10903",
        "Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. KDD '19 Workshop on Anomaly Detection in Finance, arXiv:1908.02591. https://arxiv.org/abs/1908.02591",
        "Ying, Z., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating explanations for graph neural networks. Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 9240-9251."
    ]
    for r_str in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        p.paragraph_format.space_after = Pt(5)
        run = p.add_run(r_str)
        run.font.name = "Times New Roman"
        run.font.size = Pt(9.5)

    doc.add_page_break()
    add_h1(doc, "Appendix: Reproducibility Commands & Artifact Verification")
    add_p(
        doc,
        "1. One-Command Phase 10/11 Metric & Statistical Verification:\n"
        "   `\\.venv\\Scripts\\python.exe research_extension\\phase10\\run_phase10_audit.py`\n"
        "2. Automated Test Suite Execution (24 Unit, Integration & HTTP Server Tests):\n"
        "   `\\.venv\\Scripts\\python.exe -m pytest tests/`\n"
        "3. Primary Authoritative Evidence Tables:\n"
        "   • `research_extension/phase10/PHASE10_MASTER_LOCAL_METRICS.csv`\n"
        "   • `research_extension/phase10/PHASE10_MLP_MULTISEED_VAL_VS_TEST.csv`\n"
        "   • `research_extension/phase10/PHASE10_LOCAL_VS_COLAB_RECONCILIATION.csv`\n"
        "   • `research_extension/phase10/PHASE10_STATISTICAL_COMPARISONS.csv`\n"
        "   • `research_extension/phase10/PHASE10_PER_TIMESTEP_AP.csv`\n"
        "   • `results/tables/c1_response_summary.csv` & `results/tables/c2_registers_bootstrap_summary.csv`"
    )


def generate_master_docx():
    # Generate individual chapter files first
    chapter_builders = [
        ("01_Introduction.docx", build_chapter_1),
        ("02_Literature_Review.docx", build_chapter_2),
        ("03_Methodology.docx", build_chapter_3),
        ("04_Implementation.docx", build_chapter_4),
        ("05_Results_and_Discussion.docx", build_chapter_5),
        ("06_Conclusion_and_Future_Work.docx", build_chapter_6),
    ]
    for fname, builder_fn in chapter_builders:
        c_doc = create_styled_document()
        builder_fn(c_doc)
        c_doc.save(CHAP_DIR / fname)

    # Generate complete integrated thesis DOCX
    doc = create_styled_document()

    # Title Page
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(30)
    p_title.paragraph_format.space_after = Pt(10)
    r_t = p_title.add_run("EXPLAINABLE TEMPORAL GRAPH NEURAL NETWORKS FOR ADAPTIVE FINANCIAL FRAUD DETECTION")
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(20)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(20)
    r_sub = p_sub.add_run(
        "A Controlled Chronological, Feature-Ablated, and Interventional Explainability Audit "
        "on the Elliptic Bitcoin Transaction Network"
    )
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(32)
    p_meta.paragraph_format.space_after = Pt(32)
    r_meta = p_meta.add_run(
        "A Final Year Research Thesis Submitted in Partial Fulfillment of the Requirements for the\n"
        "Degree of Bachelor of Science (Honours) in Computer Science / Artificial Intelligence\n\n"
        "Department of Computer Science & Software Engineering\n"
        "Faculty of Computing / Information Technology\n"
        "November 2026"
    )
    r_meta.font.name = "Times New Roman"
    r_meta.font.size = Pt(10.5)
    r_meta.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    add_note_box(
        doc,
        "Official approved research title preserved above. This Phase 11 Final Integrated Thesis incorporates all "
        "verified findings and corrections from Phases 6–10: (1) Average Precision (`average_precision_score`) is "
        "standardized as the primary thesis metric across all 15 verified local models and 7 Colab GPU models, kept "
        "strictly separate from supplementary Trapezoidal PR-AUC (`auc(recall, precision)`); (2) Tabular tree ensembles "
        "(Random Forest 0.6323 AP, XGBoost 0.5472 CPU / 0.6445 Colab AP) decisively outperform all GNNs (ETGNN 0.2250 AP); "
        "(3) Removing edges in a parameter-matched local ETGNN (`7,302` params) increases test AP from 0.2344 to 0.4812 "
        "(+0.2468), and a 3-layer MLP (`P3-05`) achieves 0.4662 AP (3-seed mean 0.4934 AP), providing descriptive "
        "empirical support for neighborhood feature dilution; (4) Dual node-level (B=2,000) and time-step-clustered (T=9, "
        "steps 41–49) statistical tests are reported; (5) Protocol C1 (N=240, Delta_adj = +0.0029, 95% CI [-0.0092, +0.0151]) "
        "is statistically null, while Protocol C2 (N=420 pairs across 8 registers) shows robust positive faithfulness "
        "difference specifically in ETGNN's TP–FP registers (Primary +0.0393, p_Holm = 0.0040; Degree-Sensitivity +0.0435, "
        "p_Holm = 0.0070) under explicit disclosure of validation reuse (steps 37–40), static aggregate features (94–165), "
        "and 50% ceiling budget inflation at E_cand = 2 (84.08% of nodes); and (6) EvolveGCN weight evolution (Pareja et al., "
        "2020) and GraphFramEx (Amara et al., 2022) are accurately cited.",
        title="PHASE 11 SCIENTIFIC INTEGRITY & SUPERVISOR REVIEW SUMMARY"
    )
    doc.add_page_break()

    # Abstract
    add_h1(doc, "Abstract")
    add_p(
        doc,
        "Detecting illicit financial transactions across decentralized cryptocurrency networks requires models that "
        "withstand extreme class imbalance, forward chronological distribution shift, and regulatory scrutiny. While "
        "Graph Neural Networks (GNNs) are widely assumed to outperform tabular classifiers on transaction graphs, "
        "empirical evaluations frequently conflate precomputed neighborhood features with runtime message passing and "
        "lack rigorous post-hoc explainability controls. This thesis investigates an Explainable Time-Encoded Graph "
        "Attention Network (ETGNN)—combining a 2-layer Graph Attention Network with deterministic 16-dimensional "
        "sinusoidal time-step encodings—on the benchmark Elliptic Bitcoin dataset (203,769 transactions, 234,355 "
        "intra-step directed edges across 49 bi-weekly time steps: 1–32 train, 33–40 validation, 41–49 test). We "
        "evaluate 15 verified local model configurations (spanning Random Forest, XGBoost, Logistic Regression, Deep "
        "MLPs, GCN, GraphSAGE, GAT, a parameter-matched zero-time GAT control, and proposed ETGNN variants) alongside "
        "a 7-model Google Colab GPU reproducibility benchmark, using step-function Average Precision (`average_precision_score`) "
        "as the primary metric."
    )
    add_p(
        doc,
        "Empirical results on the held-out test split (N = 9,973 labelled transactions; 524 illicit [5.25%], 9,449 licit) "
        "demonstrate that tabular tree ensembles decisively outperform all GNN architectures: `Random Forest (165f Full)` "
        "achieves 0.6323 AP (F1 = 0.6319), `Random Forest (93f Local)` achieves 0.6301 AP, `RF + Time` achieves 0.6295 AP, "
        "and `XGBoost (165f Full)` achieves 0.5472 AP (Colab GPU: 0.6445 AP), whereas `GraphSAGE` achieves 0.2427 AP, "
        "`ETGNN (93f Local)` achieves 0.2344 AP, `ETGNN (165f Full)` achieves 0.2250 AP (F1 = 0.3308), and `Matched GAT (165f Full)` "
        "achieves 0.2099 AP (F1 = 0.2074). Controlled feature and topology ablations reveal why: in a single-variable "
        "edge ablation holding architecture and parameter count constant (7,302 parameters), removing the 234,355 payment "
        "edges from `ETGNN (93f Local)` increases test AP from 0.2344 to 0.4812 (+0.2468 AP, p_Holm < 0.005), while "
        "`DeepParameterMatchedMLP (93f Local)` achieves 0.4662 AP (3-seed mean 0.4934 AP) and degrades to 0.3603 AP when "
        "precomputed 1-hop neighborhood aggregates (features 94–165) are added. Sinusoidal time encoding provides a "
        "modest point gain over zero-time Matched GAT (+0.0151 AP on CPU; +0.0483 AP on Colab GPU) that is statistically "
        "inconclusive under node-level paired bootstrap (95% CI [-0.0085, +0.0381], p = 0.2149) and time-step block "
        "bootstrap across steps 41–49 (where 67.75% of test positives precede the step 43 AlphaBay shutdown). In "
        "retrospective post-hoc GNNExplainer audits on validation steps 37–40, Protocol C1 (N = 240 shared ETGNN-ranked "
        "edge deletions) finds a null cross-model difference (Delta_adj = +0.002903 +/- 0.096180, exploratory 95% CI "
        "[-0.009201, +0.015121]). In Protocol C2 (N = 420 Hopcroft-Karp matched pair records across 377 unique nodes "
        "and 8 registers), ETGNN exhibits a statistically robust positive explanation-minus-random deletion difference "
        "between false positives and true positives in both the Primary register (N = 95 pairs, +0.039343, 95% CI "
        "[+0.020234, +0.057410], p_Holm = 0.0040) and the Exact Out-Degree register (N = 57 pairs, +0.043455, 95% CI "
        "[+0.019882, +0.069552], p_Holm = 0.0070), whereas Matched GAT attenuates under degree matching and all four "
        "TN–FN registers span zero. We further prove that because 84.08% of evaluated nodes have exactly 2 candidate "
        "edges, the 25% ceiling rule deletes 50.0% of edges, bounding edge-deletion masks strictly as local topological "
        "sensitivity indicators rather than causal explanations."
    )
    doc.add_page_break()

    # Table of Contents
    add_h1(doc, "Table of Contents")
    toc_entries = [
        ("Chapter 1: Introduction", "1.1 Background, 1.2 Fraud Detection Challenges, 1.3 Problem Statement, 1.4 Approved Research Questions (RQ1–RQ3), 1.5 Approved Research Objectives (RO1–RO5), 1.6 Scope Delimitations, 1.7 Thesis Organization"),
        ("Chapter 2: Literature Review", "2.1 Bitcoin AML Forensics, 2.2 Tabular Baselines & Feature Provenance, 2.3 Static GNNs, 2.4 Temporal GNNs & EvolveGCN Weight-Evolution Analysis, 2.5 Explainable GNNs & GraphFramEx, 2.6 Precision–Recall Evaluation, 2.7 Comparative Literature Matrix & Gap"),
        ("Chapter 3: Methodology", "3.1 Research Design, 3.2 Elliptic Dataset & Feature Decomposition, 3.3 Chronological Split & Zero-Leakage Scaling, 3.4 ETGNN & Baseline Formulations, 3.5 Metric Standardization (AP vs. Trapz PR-AUC), 3.6 Paired Bootstrap & Time-Step-Clustered Statistics, 3.7 Protocols C1 & C2"),
        ("Chapter 4: Implementation", "4.1 Software Environment, 4.2 Data Engineering Pipeline, 4.3 Training Scripts & 15-Model SHA-256 Checkpoint Inventory, 4.4 Subgraph Equivalence & Hopcroft-Karp Engine, 4.5 FastAPI + Streamlit Compliance Dashboard"),
        ("Chapter 5: Results and Discussion", "5.1 15-Model Master Local Benchmark, 5.2 Tabular vs. GNN Analysis, 5.3 Feature & No-Edge Ablations (with Multi-Seed MLP Val vs. Test), 5.4 Temporal Encoding Analysis, 5.5 Paired Bootstrap & Time-Step-Clustered Sensitivity (Steps 41–49), 5.6 Colab GPU Reproducibility, 5.7 Candidate-Edge Sparsity, 5.8 Protocol C1 Results, 5.9 Protocol C2 Results"),
        ("Chapter 6: Conclusion and Future Work", "6.1 Evidence-Based Answers to RQ1–RQ3, 6.2 Supported vs. Inconclusive Findings, 6.3 Eight Explicit Methodological Limitations, 6.4 Directions for Future Work (EvolveGCN, TGN, Joint Perturbation), 6.5 Concluding Remarks"),
        ("References & Appendix", "18 Verified Academic Citations (PMLR, AAAI, NeurIPS, KDD, ICLR, ICML, PLOS ONE) & Reproducibility Commands")
    ]
    for title, subs in toc_entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(title)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
        ps = doc.add_paragraph(f"    {subs}")
        ps.paragraph_format.space_after = Pt(4)
        for rs in ps.runs:
            rs.font.name = "Times New Roman"
            rs.font.size = Pt(9.0)
            rs.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_page_break()

    build_chapter_1(doc)
    doc.add_page_break()
    build_chapter_2(doc)
    doc.add_page_break()
    build_chapter_3(doc)
    doc.add_page_break()
    build_chapter_4(doc)
    doc.add_page_break()
    build_chapter_5(doc)
    doc.add_page_break()
    build_chapter_6(doc)
    doc.add_page_break()
    build_references_and_appendices(doc)

    docx_out = OUT_DIR / "ETGNN_Phase11_Final_Integrated_Thesis.docx"
    doc.save(docx_out)
    print(f"[SUCCESS] Generated Master DOCX: {docx_out} ({docx_out.stat().st_size:,} bytes)")
    return docx_out


# ==============================================================================
# PDF BUILDER (REPORTLAB FULL INTEGRATED PDF)
# ==============================================================================
def generate_master_pdf(docx_path: Path):
    pdf_out = OUT_DIR / "ETGNN_Phase11_Final_Integrated_Thesis.pdf"
    doc_in = docx.Document(docx_path)

    pdf = SimpleDocTemplate(
        str(pdf_out),
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    styles = getSampleStyleSheet()
    st_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#1F3864"), spaceBefore=12, spaceAfter=6)
    st_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#2B2B2B"), spaceBefore=10, spaceAfter=4)
    st_h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Times-BoldItalic", fontSize=10.5, leading=13.5, textColor=colors.HexColor("#333333"), spaceBefore=7, spaceAfter=3)
    st_body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Times-Roman", fontSize=9.5, leading=13, textColor=colors.HexColor("#222222"), spaceAfter=5)
    st_cell_hdr = ParagraphStyle("CellHdr", parent=styles["Normal"], fontName="Times-Bold", fontSize=7.0, leading=8.5, textColor=colors.white, alignment=1)
    st_cell_body = ParagraphStyle("CellBody", parent=styles["Normal"], fontName="Times-Roman", fontSize=6.8, leading=8.2, textColor=colors.HexColor("#222222"))

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")

    story = []
    # Iterate in document body order over paragraphs and tables
    from docx.text.paragraph import Paragraph as DocxParagraph
    from docx.table import Table as DocxTable

    for child in doc_in.element.body.iterchildren():
        if child.tag.endswith("p"):
            p = DocxParagraph(child, doc_in)
            txt = p.text.strip()
            if not txt:
                continue
            sname = p.style.name if p.style else "Normal"
            if sname == "Heading 1":
                if story:
                    story.append(Spacer(1, 8))
                story.append(Paragraph(esc(txt), st_h1))
            elif sname == "Heading 2":
                story.append(Paragraph(esc(txt), st_h2))
            elif sname == "Heading 3":
                story.append(Paragraph(esc(txt), st_h3))
            else:
                story.append(Paragraph(esc(txt), st_body))
        elif child.tag.endswith("tbl"):
            t = DocxTable(child, doc_in)
            nrows = len(t.rows)
            ncols = len(t.columns)
            data = []
            for r_i, row in enumerate(t.rows):
                r_list = []
                for c_i, cell in enumerate(row.cells):
                    c_txt = esc(cell.text.strip())
                    if nrows == 1 and ncols == 1:
                        r_list.append(Paragraph(c_txt, st_body))
                    elif r_i == 0:
                        r_list.append(Paragraph(c_txt, st_cell_hdr))
                    else:
                        r_list.append(Paragraph(c_txt, st_cell_body))
                data.append(r_list)
            total_w = 512.0
            col_w = [total_w / ncols] * ncols
            rl_tbl = Table(data, colWidths=col_w, repeatRows=1 if nrows > 1 else 0)
            if nrows == 1 and ncols == 1:
                rl_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F4F8")),
                    ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#1F3864")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
            else:
                rl_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]))
            story.append(Spacer(1, 4))
            story.append(rl_tbl)
            story.append(Spacer(1, 6))

    pdf.build(story)
    print(f"[SUCCESS] Generated Master PDF: {pdf_out} ({pdf_out.stat().st_size:,} bytes)")
    return pdf_out


if __name__ == "__main__":
    docx_file = generate_master_docx()
    pdf_file = generate_master_pdf(docx_file)
    for f in [docx_file, pdf_file]:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"SHA256({f.name}) = {h}")
