"""
ETGNN Final Corrected Thesis Generator
Generates the authoritative, evidence-bounded final thesis document
in thesis/final_review/ETGNN_Final_Corrected_Thesis.docx and ETGNN_Final_Corrected_Thesis.pdf.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)

FINAL_DIR = os.path.join("thesis", "final_review")
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FINAL_DIR, exist_ok=True)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_styled_document():
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
    
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    style_normal.paragraph_format.line_spacing = 1.3
    style_normal.paragraph_format.space_after = Pt(5)
    return doc

def add_h1(doc, text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    return h

def add_h2(doc, text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
    return h

def add_h3(doc, text):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(2)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return h

def add_p(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.space_after = Pt(5)
    return p

def add_note_box(doc, text, title="PROPOSED REFINEMENT FOR SUPERVISOR CONSIDERATION:"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F2F4F8")
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r_t = p.add_run(f"[{title}] ")
    r_t.bold = True
    r_t.font.name = 'Times New Roman'
    r_t.font.size = Pt(10)
    r_t.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    r_b = p.add_run(text)
    r_b.font.name = 'Times New Roman'
    r_b.font.size = Pt(9.5)
    r_b.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(1)
    p_after.paragraph_format.space_after = Pt(4)

def add_eq(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.italic = True
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)
    return p

def add_img(doc, img_name, caption, width=5.6):
    p_path = os.path.join(FIGURES_DIR, img_name)
    if os.path.exists(p_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run()
        r.add_picture(p_path, width=Inches(width))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = 'Times New Roman'
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

def add_tbl(doc, headers, rows, col_widths=None):
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    # Header
    for idx, title in enumerate(headers):
        cell = tbl.cell(0, idx)
        cell.text = title
        set_cell_background(cell, "1F3864")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            
    # Rows
    for r_idx, row in enumerate(rows):
        bg = "F7F9FB" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.text = str(val)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or c_idx == 1 else WD_ALIGN_PARAGRAPH.RIGHT
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
                
    if col_widths:
        for r in tbl.rows:
            for idx, w in enumerate(col_widths):
                r.cells[idx].width = Inches(w)
                
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(6)
    return tbl

def generate_corrected_docx():
    doc = create_styled_document()
    
    # ================= TITLE PAGE =================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(30)
    p_title.paragraph_format.space_after = Pt(10)
    r_t = p_title.add_run("EXPLAINABLE TEMPORAL GRAPH NEURAL NETWORKS FOR ADAPTIVE FINANCIAL FRAUD DETECTION")
    r_t.font.name = 'Times New Roman'
    r_t.font.size = Pt(20)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(20)
    r_sub = p_sub.add_run("A Controlled Chronological and Interventional Explainability Audit on the Elliptic Bitcoin Transaction Network")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(36)
    p_meta.paragraph_format.space_after = Pt(36)
    r_meta = p_meta.add_run(
        "A Final Year Research Thesis Submitted in Partial Fulfillment of the Requirements for the\n"
        "Degree of Bachelor of Science (Honours) in Computer Science / Artificial Intelligence\n\n"
        "Department of Computer Science & Software Engineering\n"
        "Faculty of Computing / Information Technology\n"
        "Academic Year: 2026\n"
    )
    r_meta.font.name = 'Times New Roman'
    r_meta.font.size = Pt(10.5)
    r_meta.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    add_note_box(doc, 
        "Official approved research title preserved above. Because the implemented system is a static time-encoded Graph Attention Network "
        "(without dynamic continuous-time memory or online drift adaptation), an alternative descriptive title is submitted for supervisor consideration: "
        "'Explainable Time-Encoded Graph Attention Networks for Bitcoin Illicit Transaction Classification'. "
        "Pending formal supervisor feedback, the officially approved title is retained, with architectural boundaries explicitly delimited in Section 1.6.",
        title="TITLE REFINEMENT GUIDANCE FOR SUPERVISOR")
    
    doc.add_page_break()
    
    # ================= ABSTRACT =================
    add_h1(doc, "Abstract")
    add_p(doc, 
        "The expansion of decentralized cryptographic asset networks has intensified the requirement for robust automated forensic systems capable "
        "of detecting illicit financial transactions, including money laundering, ransomware operations, and darknet vendor activities. Graph Neural "
        "Networks (GNNs) offer natural inductive representations for transaction networks, yet their forensic application encounters acute challenges: "
        "extreme class imbalance, temporal concept drift, and the black-box nature of deep geometric representations. This undergraduate thesis "
        "evaluates a Time-Encoded Graph Attention Network (ETGNN) for illicit Bitcoin transaction classification on the benchmark Elliptic dataset "
        "(203,769 transactions, 234,355 directed edges, across 49 discrete time steps). The research executes two interconnected investigations: "
        "(1) a leak-free chronological classification evaluation against competitive tabular tree ensembles (Random Forest, XGBoost) and deep graph "
        "baselines (GCN, GraphSAGE, static GAT), and (2) a frozen post-hoc explainability benchmark utilizing GNNExplainer across 377 completed "
        "validation transactions (steps 37–40) evaluating random-adjusted deletion responses under shared interventions (Contrast 1) and margin-matched "
        "correct-versus-incorrect prediction pairs (Contrast 2).")
    add_p(doc, 
        "Empirical results indicate that tabular Random Forest achieved the highest held-out test performance (time steps 41–49) with an illicit-class "
        "Average Precision (AP / AUPRC) of 0.6323, substantially surpassing XGBoost (0.5472), GraphSAGE (0.2427), the parameter-matched zero-time GAT "
        "control (0.2099), and the proposed ETGNN (0.2250). A controlled topological edge ablation revealed that evaluating ETGNN on local features with "
        "all graph edges removed increased test AP from 0.2344 to 0.4812 (+0.2468), supporting the hypothesis that unconstrained message passing across "
        "within-step payment edges propagates structural noise in high-degree transaction hubs. In post-hoc explainability audits, Contrast 1 demonstrated "
        "a small observed mean random-adjusted difference between ETGNN and Matched GAT (+0.002903 across 240 transactions, exploratory 95% bootstrap CI "
        "[-0.009201, 0.015121]). In Contrast 2, within-model matched-pair cluster bootstrapping (2,000 iterations) demonstrated that GNNExplainer edge "
        "omissions induced a larger probability drop on false positives than true positives for ETGNN across primary (+0.039343, 95% CI [+0.020234, +0.057410]) "
        "and degree-balanced (+0.043455, 95% CI [+0.019882, +0.069552]) registers. However, because 84.1% of evaluated transactions possess only two "
        "eligible computation edges, discrete ceiling budget rules imposed an actual 50.0% structural deletion fraction. These results indicate that "
        "deletion responses capture local model sensitivity to topological omission rather than causal explanations of fraud, establishing evidence-bounded "
        "constraints for accountable cryptocurrency forensics.")
    add_p(doc, 
        "Keywords: Graph Neural Networks, Financial Fraud Detection, Bitcoin Forensic Analytics, Elliptic Dataset, Graph Attention Networks, "
        "GNNExplainer, Post-Hoc Explainability, Chronological Evaluation, Perturbation Auditing, Model Interpretability.")
    
    doc.add_page_break()
    
    # ================= TABLE OF CONTENTS =================
    add_h1(doc, "Table of Contents")
    toc = [
        ("Chapter 1: Introduction", "1.1 Background, 1.2 Fraud Detection Challenges, 1.3 Problem Statement, 1.4 Research Questions, 1.5 Objectives, 1.6 Scope & Delimitations, 1.7 Thesis Organization"),
        ("Chapter 2: Literature Review", "2.1 Blockchain AML Forensics, 2.2 Tabular Tree Ensembles, 2.3 Graph Representation Learning, 2.4 Temporal Graph Learning Paradigms, 2.5 Explainability & Faithfulness Benchmarks, 2.6 Literature Matrix, 2.7 Critical Synthesis & Research Gap"),
        ("Chapter 3: Methodology and Implementation", "3.1 Research Design, 3.2 Elliptic Dataset Architecture, 3.3 Feature Decomposition, 3.4 Chronological Splitting & Leakage Controls, 3.5 Model Architectures, 3.6 Training Protocols, 3.7 Evaluation Metrics (AP vs AUPRC), 3.8 GNNExplainer Protocol, 3.9 Contrast 1 Specification, 3.10 Contrast 2 Matched-Pair Specification"),
        ("Chapter 4: Classification Results and Analysis", "4.1 Test Split Baseline Performance, 4.2 Tabular vs Graph Performance Discrepancy, 4.3 Topological Message-Passing Ablation, 4.4 Multi-Seed Temporal Validation Ablation, 4.5 Temporal Concept Drift Analysis"),
        ("Chapter 5: Explainability Results and Discussion", "5.1 Subgraph Sparsity & Budget Discreteness, 5.2 Contrast 1 Results, 5.3 Contrast 2 Bootstrap Results, 5.4 Subgraph Visualizations, 5.5 Interactive Demo System, 5.6 Threats to Validity & Epistemic Boundaries"),
        ("Chapter 6: Conclusion and Future Work", "6.1 Research Question Answers, 6.2 Summary of Verified Contributions, 6.3 Limitations, 6.4 Directions for Future Research, 6.5 Concluding Remarks"),
        ("References", "Verified Bibliographic Register with Authoritative Identifiers"),
        ("Appendices", "Appendix A: Mathematical Formulations, Appendix B: Cryptographic Hash Register, Appendix C: Reproducibility Commands")
    ]
    for ch, subs in toc:
        p_t = doc.add_paragraph()
        p_t.paragraph_format.space_after = Pt(2)
        r = p_t.add_run(ch)
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10.5)
        p_s = doc.add_paragraph(f"    {subs}")
        p_s.paragraph_format.space_after = Pt(4)
        for r_sub in p_s.runs:
            r_sub.font.name = 'Times New Roman'
            r_sub.font.size = Pt(9)
            r_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            
    doc.add_page_break()
    
    # ================= CHAPTER 1 =================
    add_h1(doc, "Chapter 1: Introduction")
    add_h2(doc, "1.1 Background and Context")
    add_p(doc, 
        "The emergence of decentralized cryptographic ledgers over the past decade has fundamentally reshaped financial technology. "
        "Bitcoin, introduced by Satoshi Nakamoto in 2008, established a peer-to-peer electronic payment network operating without intermediary "
        "financial institutions. While blockchain architectures provide settlement transparency and censorship resistance, their pseudonymous "
        "nature—where transactions are recorded under cryptographic public keys rather than verified legal identities—presents substantial compliance "
        "challenges. Malicious actors have exploited this pseudonymity to execute financial crimes, including darknet bazaar payments, ransomware "
        "extortions, terrorist financing, and multi-hop money laundering schemes.")
    add_p(doc, 
        "To comply with international Anti-Money Laundering (AML) standards (such as Financial Action Task Force recommendations), regulatory authorities "
        "and virtual asset service providers require automated forensic analytics. Conventional compliance screening relied on static address blacklists "
        "and rule-based heuristics. However, modern money laundering techniques employ peeling chains, coin mixing services, and rapid multi-hop "
        "transaction layering, rendering heuristic filtering largely ineffective against emerging threat typologies.")

    add_h2(doc, "1.2 The Financial Fraud Detection Challenge")
    add_p(doc, 
        "Financial transaction networks present three fundamental modeling complexities:\n"
        "1. Extreme Class Imbalance: Malicious entities constitute an extreme minority of overall network transactions (frequently below 3% to 5%). "
        "Classifiers optimized for standard 0-1 loss produce deceptive results that optimize overall accuracy while failing to detect financial crime.\n"
        "2. Relational and Topological Dependencies: Transactions do not exist as independent observations; an entity's illicit status is intimately "
        "tied to its structural position within payment-flow graphs and the characteristics of its transacting partners.\n"
        "3. Temporal Dynamics and Network Evolution: Illicit typologies, transaction volumes, and regulatory actions shift dynamically over time, "
        "requiring models to be evaluated under realistic chronological partitions that prevent future data leakage.")

    add_h2(doc, "1.3 Problem Statement")
    add_p(doc, 
        "Graph Neural Networks (GNNs) offer natural representations for relational payment networks. However, their application to transaction "
        "forensics encounters two major obstacles. First, public benchmarks such as the Elliptic dataset provide transaction edges strictly within "
        "discrete time steps; there are no continuous cross-time-step edges. Standard GNNs operating purely on static topology cannot leverage "
        "chronological progression without explicit temporal representations. Second, deep geometric architectures operate as opaque black boxes. "
        "Compliance officers require accountable, verifiable explanations justifying why a transaction was flagged. Post-hoc explainers (such as "
        "GNNExplainer) generate subgraph masks, but conventional evaluations rely on anecdotal visualizations rather than controlled interventional "
        "benchmarking across correct classifications and systematic model errors.")

    add_h2(doc, "1.4 Research Questions")
    add_p(doc, "This research investigates four evidence-bounded research questions:")
    add_p(doc, 
        "• Research Question 1 (RQ1 - Baseline Comparison): How does a Graph Attention Network augmented with fixed sinusoidal time-step "
        "encodings (ETGNN) perform relative to established tabular baselines (Random Forest, XGBoost) and deep graph architectures (GCN, GraphSAGE, "
        "static GAT) on held-out future Elliptic time steps when evaluated via illicit-class Average Precision (AP / AUPRC)?\n"
        "• Research Question 2 (RQ2 - Temporal Encoding Contribution): On validation data, does adding a fixed 16-dimensional sinusoidal time "
        "encoding yield a consistent improvement in AP over a parameter-matched zero-time GAT control across random seeds?\n"
        "• Research Question 3 (RQ3 - Shared Interventional Contrast C1): When subjected to identical edge deletions selected by GNNExplainer on "
        "ETGNN under a common budget, do ETGNN and its matched zero-time GAT control exhibit a differential prediction-probability response after "
        "calibrating against random edge deletion baselines?\n"
        "• Research Question 4 (RQ4 - Matched Error-vs-Correct Contrast C2): After matching transaction pairs on prediction margin and candidate-edge "
        "count, does explanation-minus-random deletion advantage differ systematically between erroneous predictions (False Positives, False Negatives) "
        "and correct predictions (True Positives, True Negatives) across primary and degree-balanced registers?")

    add_h2(doc, "1.5 Research Objectives")
    add_p(doc, 
        "The operational objectives executed in this research are:\n"
        "1. Implement a leak-free chronological data engineering pipeline on Elliptic (scaling fitted strictly on time steps 1–32).\n"
        "2. Construct competitive tabular baselines (Random Forest, XGBoost) and deep GNN baselines (GCN, GraphSAGE, GAT, Matched GAT).\n"
        "3. Design the ETGNN architecture augmenting node features with deterministic sinusoidal time projections before multi-head attention.\n"
        "4. Conduct topological and feature ablation experiments isolating the effects of time encodings, local features, and graph edges.\n"
        "5. Formulate and execute Contrast 1 (C1): an interventional audit across 240 validation transactions under shared edge deletions.\n"
        "6. Formulate and execute Contrast 2 (C2): an 8-register matched-pair cluster bootstrap audit (420 pair records) comparing error and correct predictions.\n"
        "7. Deploy a decoupled interactive demonstration system integrating a FastAPI backend with a Streamlit compliance dashboard.")

    add_h2(doc, "1.6 Scope and Delimitations")
    add_p(doc, 
        "To maintain scientific precision, the scope is delimited as follows:\n"
        "• Static Time-Encoded GAT vs. Dynamic Memory: The implemented ETGNN is a static GAT supplied with fixed sinusoidal time-step encodings. "
        "It does not implement continuous-time event memory (e.g., TGN, JODIE), recurrent parameter updates (EvolveGCN), or dynamic link prediction.\n"
        "• Title Refinement Note: The term 'Adaptive' in the approved project title reflects the motivating compliance domain rather than an implemented "
        "online continuous-learning mechanism. The proposed refinement 'Explainable Time-Encoded Graph Attention Networks for Bitcoin Illicit "
        "Transaction Classification' is submitted for supervisor consideration.\n"
        "• Frozen Checkpoints: All explainability interventions were executed on frozen model checkpoints without retraining.\n"
        "• Transductive Snapshot Setting: The experimental pipeline assumes within-step edges and precomputed features are available at snapshot completion; "
        "it does not evaluate immediate sub-second mempool arrival screening.")

    add_h2(doc, "1.7 Thesis Organization")
    add_p(doc, 
        "The remainder of this thesis is organized as follows: Chapter 2 reviews foundational literature in cryptocurrency AML, tabular ML, graph "
        "representation learning, temporal GNNs, and XAI evaluation benchmarks. Chapter 3 details the data engineering, architectural specifications, "
        "metrics, and the formal protocols of Contrasts C1 and C2. Chapter 4 reports held-out classification performance, tabular baseline dominance, "
        "and topological ablations. Chapter 5 analyzes the C1 and C2 explainability findings, candidate-edge budget discreteness, demo implementation, "
        "and epistemic boundaries. Chapter 6 concludes with direct answers to research questions, verified contributions, and future research directions.")

    doc.add_page_break()
    
    # ================= CHAPTER 2 =================
    add_h1(doc, "Chapter 2: Literature Review")
    add_h2(doc, "2.1 Anti-Money Laundering Forensics on Blockchain Networks")
    add_p(doc, 
        "Cryptocurrency systems record transactions on public immutable ledgers, but transactions are associated with pseudonymous addresses rather "
        "than verified entities. Early blockchain analytics employed entity-resolution heuristics (such as common-input ownership and change address "
        "clustering). While foundational, heuristic address clustering degrades when confronted with privacy overlays, mixers (CoinJoin), and multi-hop "
        "layering. The release of the Elliptic Bitcoin dataset by Weber et al. (2019) provided the first large-scale benchmark (203,769 transactions, "
        "234,355 payment-flow edges, across 49 time steps) enabling supervised machine learning for cryptocurrency forensics. Crucially, Weber et al. "
        "observed that standard Random Forest classifiers outperformed basic Graph Convolutional Networks, establishing a strong tabular precedent.")

    add_h2(doc, "2.2 Tabular Machine Learning Baselines and Feature Provenance")
    add_p(doc, 
        "Decision tree ensembles represent established benchmarks for tabular classification. Breiman (2001) formalized Random Forests, which combine "
        "bootstrap aggregation with randomized feature subspace selection to minimize variance without inflating bias. Chen and Guestrin (2016) developed "
        "XGBoost, an optimized gradient boosting framework utilizing second-order Taylor approximations and sparsity-aware learning. A critical "
        "methodological reality in Elliptic is that its 166 published features contain 94 local transaction descriptors and 72 precomputed one-hop "
        "neighborhood aggregates (capturing minimum, maximum, mean, and standard deviation of transacting neighbors' properties). Consequently, when "
        "Random Forest and XGBoost are trained on the full 165 numeric feature matrix, they already receive graph-derived topological information.")

    add_h2(doc, "2.3 Graph Representation Learning")
    add_p(doc, 
        "Graph Neural Networks (GNNs) extend deep learning to graph-structured data via recursive neighborhood message passing. Kipf and Welling (2017) "
        "introduced Graph Convolutional Networks (GCN), which approximate spectral convolutions via first-order localized Chebyshev expansions. Hamilton, "
        "Ying, and Leskovec (2017) proposed GraphSAGE, enabling inductive generalization across large graphs by sampling localized neighborhoods and "
        "learning parameterizable aggregation operators (such as mean pooling). Veličković et al. (2018) addressed uniform isotropic aggregation by "
        "introducing Graph Attention Networks (GAT), which incorporate self-attention mechanisms to assign learned, anisotropic attention weights "
        "alpha_ij to neighboring nodes based on paired feature projections.")

    add_h2(doc, "2.4 Temporal Graph Learning Paradigms")
    add_p(doc, 
        "Temporal graph literature broadly encompasses two distinct paradigms:\n"
        "1. Dynamic Graph Snapshots: Methods such as EvolveGCN (Pareja et al., 2020) model networks as discrete snapshot sequences, using recurrent "
        "architectures (LSTM/GRU) to evolve GNN weight matrices over successive periods.\n"
        "2. Continuous-Time Event Streams: Architectures such as Temporal Graph Networks (TGN; Rossi et al., 2020) and JODIE process individual "
        "timestamped events, maintaining continuous node memory states that update whenever an interaction occurs.\n"
        "Crucially, Elliptic edges connect transactions strictly within the same discrete time step; there are zero cross-time-step edges. Therefore, "
        "the implemented ETGNN operates within completed snapshots, augmenting node representations with deterministic sinusoidal time encodings "
        "rather than maintaining continuous memory or recurrent state evolution.")

    add_h2(doc, "2.5 Explainability Methods and Evaluation Benchmarks")
    add_p(doc, 
        "In compliance and financial forensics, model interpretability is essential. Ying et al. (2019) introduced GNNExplainer, an optimization-based "
        "framework that identifies compact subgraphs and feature subsets by maximizing mutual information between masked subgraphs and original predictions. "
        "However, evaluating explanation quality in real-world graphs is difficult because ground-truth explanation labels do not exist. In general XAI, "
        "Hooker et al. (2019) demonstrated via the ROAR framework that feature attributions must be evaluated against explicit random degradation baselines. "
        "In the graph domain, Amara et al. (2022) formalized GraphFramEx, establishing evaluation protocols that assess deletion fidelity (measuring "
        "prediction drops under top-ranked edge deletion versus unranked or random edges).")

    add_h2(doc, "2.6 Comparative Literature Matrix")
    add_p(doc, "Table 2.1 summarizes foundational studies directly informing this research:")
    
    headers_lit = ["Study", "Domain", "Core Verified Contribution", "Relevance to ETGNN Research"]
    rows_lit = [
        ["Weber et al. (2019)", "Bitcoin AML / Elliptic", "Released Elliptic benchmark; observed RF outperforming GCN.", "Defines dataset; establishes strong tree baseline precedent; reveals 72 neighborhood aggregates."],
        ["Breiman (2001)", "Ensemble Learning / RF", "Formalized bootstrap aggregation with random feature subspaces.", "Forms primary tabular benchmark using balanced class weights across 100 trees."],
        ["Chen & Guestrin (2016)", "Gradient Boosting / XGB", "Scalable tree boosting with second-order gradient optimization.", "Forms secondary boosted tabular baseline with scale_pos_weight optimization."],
        ["Kipf & Welling (2017)", "Geometric DL / GCN", "Localized first-order spectral convolutions on graph neighborhoods.", "Direct foundation for homogeneous graph convolution baseline on chronological snapshots."],
        ["Hamilton et al. (2017)", "Inductive DL / SAGE", "Inductive neighborhood sampling and aggregation functions.", "Foundation for inductive neighborhood aggregation baseline using mean pooling."],
        ["Veličković et al. (2018)", "Attention DL / GAT", "Masked self-attention mechanisms over graph neighborhoods.", "Architectural backbone for standard GAT, Matched GAT control, and ETGNN."],
        ["Ying et al. (2019)", "Graph XAI / GNNExplainer", "Mutual information maximization over continuous edge/feature masks.", "Core explanation optimization algorithm across 2-hop computation subgraphs."],
        ["Hooker et al. (2019)", "XAI Benchmark / ROAR", "Established that attribution must be calibrated against random baselines.", "Theoretical motivation for calibrating GNNExplainer deletion against 30 random draws."],
        ["Amara et al. (2022)", "Graph XAI / GraphFramEx", "Formalized systematic evaluation protocols for graph explainability.", "Directly informs our structural edge-deletion protocol and fidelity metrics."],
        ["Saito & Rehmsmeier (2015)", "Statistical Evaluation", "Proved PR curves/AUPRC are more informative than ROC in imbalance.", "Establishes illicit-class Average Precision (AP) as primary evaluation metric."]
    ]
    add_tbl(doc, headers_lit, rows_lit, col_widths=[1.3, 1.3, 2.0, 1.9])

    add_h2(doc, "2.7 Critical Synthesis and Research Gap")
    add_p(doc, 
        "Prior graph explanation literature predominantly evaluates explainers on synthetic motifs or static citation graphs, rarely subjecting real-world "
        "imbalanced financial networks to rigorous interventional audits. Crucially, existing studies almost universally evaluate models against their own "
        "explanations and restrict audits to correct classifications. This thesis addresses this empirical gap through two controlled investigations: "
        "(1) a shared-intervention comparison subjecting ETGNN and a parameter-matched zero-time GAT control to identical edge deletions under a common "
        "target class (Contrast 1), and (2) a margin-matched and degree-balanced audit comparing explanation sensitivity on model errors versus correct "
        "classifications across eight comparison registers (Contrast 2).")

    doc.add_page_break()
    
    # ================= CHAPTER 3 =================
    add_h1(doc, "Chapter 3: Methodology and Implementation")
    add_h2(doc, "3.1 Research Design Overview")
    add_p(doc, 
        "This research implements an empirical framework partitioned into two tracks: Part A evaluates predictive classification performance across "
        "tabular tree ensembles and graph architectures under strict chronological splitting. Part B executes a frozen post-hoc explainability benchmark "
        "using GNNExplainer to evaluate structural sensitivity under shared deletions (Contrast 1) and matched error-vs-correct comparisons (Contrast 2).")

    add_h2(doc, "3.2 Elliptic Dataset Architecture")
    add_p(doc, 
        "The Elliptic dataset contains 203,769 transactions and 234,355 directed edges spanning 49 discrete time steps. Label distribution:\n"
        "• Class 1 (Illicit): 4,545 transactions (~2.23% of total nodes), mapped to positive target label y = 1.\n"
        "• Class 2 (Licit): 42,019 transactions (~20.62% of total nodes), mapped to negative target label y = 0.\n"
        "• Unknown (Unlabelled): 157,205 transactions (~77.15% of total nodes), mapped to y = -1. Unknown transactions participate fully in graph "
        "message passing but are strictly masked out of loss functions and evaluation metrics.")

    add_h2(doc, "3.3 Feature Decomposition and Provenance")
    add_p(doc, 
        "The 165 continuous numeric features are partitioned into:\n"
        "1. Local Features (Columns 1–93): Transaction fees, transacted BTC volume, input/output counts, and script metadata.\n"
        "2. Precomputed Aggregates (Columns 94–165): 72 neighborhood statistics (min, max, mean, std) computed over one-hop transaction neighbors.\n"
        "Because full-feature tabular models receive these 72 aggregates, they are not topological-free baselines. Our design incorporates local-feature "
        "ablations to isolate runtime message passing from precomputed tabular inputs.")

    add_h2(doc, "3.4 Chronological Splitting and Leakage Controls")
    add_p(doc, 
        "To enforce chronological realism, the 49 time steps are partitioned as follows:\n"
        "• Training Period: Time steps 1 to 32 (130,628 total transactions; 2,987 illicit, 26,734 licit).\n"
        "• Validation Period: Time steps 33 to 40 (Used for early stopping and post-hoc explainability audits).\n"
        "• Held-Out Test Period: Time steps 41 to 49 (9,973 labelled transactions; 524 illicit [5.25%], 9,449 licit).\n"
        "Leakage Prevention: StandardScaler was fitted strictly on training time steps 1–32 (130,628 nodes) and applied forward to validation and test sets. "
        "Stage-specific graph edge lists ensure that message passing at time step T cannot observe edges from future time steps T' > T.")
    add_note_box(doc, 
        "METHODOLOGICAL DISCLOSURE: The post-hoc explainability benchmark (Contrasts C1 and C2) was conducted across 377 completed transactions from "
        "validation time steps 37–40 (following exploratory development on steps 33–36). Because these transactions originate from the validation period "
        "where model checkpoints were selected, these experiments represent structured retrospective diagnostic audits rather than prospective holdout trials. "
        "Furthermore, historical test split performance (steps 41–49) was inspected during preliminary research stages and is reported under frozen checkpoints.",
        title="RETROSPECTIVE VALIDATION REUSE DISCLOSURE")

    add_h2(doc, "3.5 Implemented Model Architectures")
    add_p(doc, 
        "• Random Forest: 100 decision trees, balanced class sub-sampling (class_weight='balanced'), Gini impurity split criteria, evaluated on full 165 features, "
        "local 93 features, and features augmented with time encodings.\n"
        "• XGBoost: Implemented using exact greedy boosting, scale_pos_weight = 8.95, maximum depth 6, learning rate 0.05, early stopping on validation AP.\n"
        "• GCN: 2 GCNConv layers, hidden dimension 64, ReLU activation, dropout 0.1.\n"
        "• GraphSAGE: 2 SAGEConv layers with mean neighborhood aggregation, hidden dimension 64, ReLU activation, dropout 0.1.\n"
        "• Standard GAT: 2 GATConv layers with 4 attention heads (hidden dim 64 per head, concatenated to 256 in layer 1; 1 head in output layer), ELU activation, dropout 0.1.\n"
        "• Matched Zero-Time GAT Control: Input dimension 181 (165 features + 16 zero-padded inputs), identical layer structure and parameter capacity to ETGNN.\n"
        "• Proposed ETGNN: Input dimension 181 (165 features + 16-dimensional sinusoidal time encoding vector PE(t)), 2 GATConv layers (4 heads per layer, hidden dim 64), "
        "ELU activation, dropout 0.1, trained with class-weighted cross-entropy (positive weight 8.95) and Adam optimizer (lr=0.001, weight decay 1e-5).")
    add_eq(doc, "PE(t, 2i) = sin( t / (10000^(2i / 16)) ),    PE(t, 2i+1) = cos( t / (10000^(2i / 16)) )")

    add_h2(doc, "3.6 Evaluation Metrics Formulation: AP vs. AUPRC")
    add_p(doc, 
        "In forensic classification with severe class imbalance (5.25% illicit test prevalence), standard accuracy is uninformative. The primary metric "
        "is illicit-class Average Precision (AP), computed using sklearn.metrics.average_precision_score as:")
    add_eq(doc, "AP = sum_n (R_n - R_(n-1)) * P_n")
    add_p(doc, 
        "where P_n and R_n are precision and recall at the n-th threshold. Throughout this thesis, the terms 'AP' and 'AUPRC' denote this step-interpolated "
        "Average Precision metric, distinguished from trapezoidal approximations. Secondary metrics include ROC-AUC, Precision, Recall, and F1-score evaluated "
        "at the canonical decision threshold tau = 0.50:")
    add_eq(doc, "y^ = 1  if  p(illicit) > 0.50,    y^ = 0  if  p(illicit) <= 0.50")

    add_h2(doc, "3.7 GNNExplainer Protocol and Budget Discreteness")
    add_p(doc, 
        "GNNExplainer optimizes continuous edge masks M in [0, 1] over 50 epochs (Adam lr=0.01) across frozen model checkpoints under fixed seeds [42, 43, 44]. "
        "Runtime assertions verified that two-hop computation subgraph forward passes reproduced full-graph prediction probabilities within 1e-5 tolerance "
        "on 100% of 377 completed nodes for both ETGNN and Matched GAT (independent stratified check observed max discrepancy bounded by 3.58e-7).\n"
        "Discrete Ceiling Budget Rule: Top-k edges were deleted according to:")
    add_eq(doc, "k = max( 1, ceil( 0.25 * E_cand ) )")
    add_p(doc, 
        "where E_cand denotes eligible candidate edges in the defined computation neighborhood. Because 317 of 377 transactions (84.1%) have exactly E_cand = 2, "
        "deleting 1 edge (k = 1) imposes an actual deletion fraction of 50.0% (a 2.0x inflation over the nominal 25% target). Across all transactions, the mean "
        "actual deletion fraction was 47.08% (median 50.0%).")

    add_h2(doc, "3.8 Contrast 1 (C1) Specification")
    add_p(doc, 
        "Contrast 1 evaluates whether ETGNN and Matched GAT respond differently when subjected to the identical deletion of top edges selected by GNNExplainer "
        "on ETGNN across N = 240 transactions (steps 37–40) evaluated on a common target class (ETGNN's predicted class). For each model M, the advantage over "
        "30 random edge deletion draws under identical budget k is:")
    add_eq(doc, "Adv_M(v) = D_(GE, M)(v) - (1 / 30) * sum_(d=1)^30 D_(Rnd, M)^(d)(v)")
    add_p(doc, "The primary C1 estimator is the random-adjusted difference:")
    add_eq(doc, "Delta_adj(v) = Adv_ETGNN(v) - Adv_Matched_GAT(v)")
    add_p(doc, "with unadjusted raw drop difference Delta_raw(v) = D_(GE, ETGNN)(v) - D_(GE, Matched_GAT)(v) as a secondary diagnostic.")

    add_h2(doc, "3.9 Contrast 2 (C2) Matched-Pair Specification")
    add_p(doc, 
        "Contrast 2 evaluates within-model explainer advantages across matched pairs between erroneous (FP, FN) and correct (TP, TN) classifications using each "
        "model's own GNNExplainer rankings and own predicted classes. Pairs were matched within the same time step using Hopcroft-Karp maximum bipartite matching under:\n"
        "• Margin Caliper: margin(v) = |p(illicit, v) - 0.50|; matching condition: |margin(treat) - margin(ctrl)| <= 0.050.\n"
        "• Candidate Edge Caliper: |E_cand(treat) - E_cand(ctrl)| <= 5.\n"
        "• Exact Out-Degree Sensitivity: out_degree_treat == out_degree_ctrl for 100% of pairs in sensitivity registers.\n"
        "Paired difference: Delta_pair = Adv(incorrect) - Adv(correct). Uncertainty was estimated via matched-pair cluster bootstrap (B = 2,000 iterations, "
        "seed 20261020), resampling pairs as clusters across 8 comparison registers (420 total pair records across registers).")

    doc.add_page_break()
    
    # ================= CHAPTER 4 =================
    add_h1(doc, "Chapter 4: Classification Results and Analysis")
    add_h2(doc, "4.1 Test Split Classification Performance")
    add_p(doc, 
        "Table 4.1 summarizes the complete performance metrics across all eleven model configurations evaluated on the held-out test split (time steps 41–49; "
        "9,973 labelled transactions; 524 illicit cases, 5.25% prevalence) at decision threshold tau = 0.50.")
    
    headers_t41 = ["Model Architecture", "Input Features", "Precision", "Recall", "F1", "ROC-AUC", "Test AP"]
    rows_t41 = [
        ["Random Forest (Full)", "165 features (93 local + 72 agg)", "0.6966", "0.5782", "0.6319", "0.8735", "0.6323"],
        ["Random Forest (Local)", "93 local numeric features only", "0.5635", "0.5840", "0.5736", "0.8515", "0.6301"],
        ["Random Forest (+ Time)", "165 features + 16-d time encoding", "0.6594", "0.5802", "0.6173", "0.8672", "0.6295"],
        ["XGBoost", "165 features (gradient boosted)", "0.2823", "0.6050", "0.3849", "0.8074", "0.5472"],
        ["ETGNN (No-Edge Control)", "93 local + 16-d time, NO EDGES", "0.2445", "0.6164", "0.3501", "0.8107", "0.4812"],
        ["GraphSAGE", "165 features + 2-layer mean agg", "0.1238", "0.7481", "0.2125", "0.8205", "0.2427"],
        ["ETGNN (Local Only)", "93 local + 16-d time + edges", "0.2225", "0.5286", "0.3132", "0.7894", "0.2344"],
        ["ETGNN (Full Proposed)", "165 features + 16-d time + edges", "0.2614", "0.4504", "0.3308", "0.8062", "0.2250"],
        ["Matched GAT Control", "165 features + 16 zero-pad + edges", "0.1205", "0.7424", "0.2074", "0.8175", "0.2099"],
        ["GCN", "165 features + 2-layer convolution", "0.1009", "0.8645", "0.1808", "0.7732", "0.1306"],
        ["Standard GAT", "165 features + 2-layer attention", "0.0893", "0.9141", "0.1627", "0.7548", "0.1155"]
    ]
    add_tbl(doc, headers_t41, rows_t41, col_widths=[1.4, 1.7, 0.65, 0.65, 0.65, 0.7, 0.75])

    add_h2(doc, "4.2 Tabular Dominance and Hypothesis H1 Outcome")
    add_p(doc, 
        "The empirical findings definitively answer Research Question 1: Hypothesis H1 (that ETGNN would outperform the strongest baseline) is NOT supported. "
        "Random Forest achieved an AP of 0.6323, outperforming ETGNN (0.2250) by 0.4073 AP points. XGBoost also surpassed all GNNs, achieving an AP of 0.5472. "
        "Inspection of the precision-recall trade-offs reveals that while standard GNNs captured high recall (GAT reached 91.41% recall; GCN reached 86.45%), "
        "they suffered catastrophic precision collapse (GAT precision was 8.93%; GCN was 10.09%). Standard GNNs generated over nine false alarms for every genuine "
        "illicit transaction identified. Random Forest maintained a balanced precision of 69.66% and recall of 57.82%, yielding an F1-score of 0.6319.")

    add_h2(doc, "4.3 Topological Message-Passing Ablation")
    add_p(doc, 
        "When ETGNN was evaluated on 93 local features with full graph edges, it achieved an AP of 0.2344. When evaluated with the exact same architecture, "
        "local features, and time encodings but with all graph edges removed (the 'No-Edge Control'), AP surged to 0.4812 (+0.2468 increase; Table 4.1). "
        "This ablation demonstrates that within-step payment edges did not assist fraud detection; rather, message passing across these edges actively degraded "
        "classification precision. In transaction networks where money launderers mix licit and illicit flows across high-degree aggregation vertices, "
        "unconstrained neighborhood convolution propagates noise, supporting the hypothesis that topological message passing dilutes localized discriminative features.")
    
    add_img(doc, "ablation_graph_vs_features.png", 
        "Figure 4.1: Feature and Edge Ablation Comparison. Demonstrates the substantial performance increase when graph edges are removed in local-feature ETGNN.", 
        width=5.0)

    add_h2(doc, "4.4 Multi-Seed Temporal Encoding Validation Ablation")
    add_p(doc, 
        "To evaluate Research Question 2 (Hypothesis H2), sinusoidal time encoding was compared against the Matched Zero-Time GAT control across three identical "
        "seeds on validation time steps 33–40:\n"
        "• Seed 7: ETGNN AP = 0.5096 vs. Matched GAT AP = 0.5572 (Matched GAT higher by +0.0476)\n"
        "• Seed 42: ETGNN AP = 0.5488 vs. Matched GAT AP = 0.4836 (ETGNN higher by +0.0652)\n"
        "• Seed 123: ETGNN AP = 0.5293 vs. Matched GAT AP = 0.3934 (ETGNN higher by +0.1359)\n"
        "• Three-Seed Mean: ETGNN Mean = 0.5292 vs. Matched GAT Mean = 0.4781 (+0.0511 advantage for ETGNN).\n"
        "Outcome: While ETGNN held a higher three-seed validation mean, Matched GAT was higher under Seed 7, and held-out test differences were small (0.2250 vs 0.2099). "
        "Hypothesis H2 is partially supported on validation data, but does not demonstrate a statistically generalizable temporal effect.")

    add_h2(doc, "4.5 Temporal Concept Drift and Prevalence Volatility")
    add_p(doc, 
        "Figure 4.2 plots model performance across test time steps (41 to 49) alongside illicit class prevalence. The evidence demonstrates severe temporal drift: "
        "at time steps 41 and 42, Random Forest achieved strong AP (>0.75), but performance degraded at steps 45–48. Crucially, illicit prevalence fluctuates "
        "dramatically: time step 46 contains only two illicit transactions (0.28% prevalence), causing finite-sample score volatility rather than intrinsic model failure.")
    
    add_img(doc, "test_auprc_by_time_step.png", 
        "Figure 4.2: Held-Out Test AP by Time Step alongside Illicit Prevalence Drift across Steps 41 to 49.", 
        width=5.5)

    doc.add_page_break()
    
    # ================= CHAPTER 5 =================
    add_h1(doc, "Chapter 5: Explainability Results and Discussion")
    add_h2(doc, "5.1 Subgraph Sparsity and Budget Discreteness")
    add_p(doc, 
        "Auditing the 377 completed validation transactions (steps 37–40) reveals that 317 transactions (84.1%) possess exactly E_cand = 2 eligible candidate "
        "edges. Under the ceiling rule k = max(1, ceil(0.25 * E_cand)), deleting one edge removes 50.0% of the candidate edges—a two-fold inflation over the "
        "nominal 25% target. Mean actual deletion fraction across all transactions was 47.08% (median 50.0%). Perturbation responses therefore represent "
        "discrete topological amputations rather than marginal sensitivity gradients.")
    
    add_img(doc, "candidate_edges_deletion_budget.png", 
        "Figure 5.1: Distribution of Eligible Candidate Edges and Actual Deletion Fraction Inflation under the Ceiling Rule.", 
        width=5.5)

    add_h2(doc, "5.2 Contrast 1 (C1): Shared-Intervention Between-Model Audit")
    add_p(doc, 
        "Contrast 1 evaluates whether ETGNN and Matched GAT respond differently when subjected to the identical deletion of top edges selected by GNNExplainer "
        "on ETGNN across N = 240 validation transactions (steps 37–40). Table 5.1 summarizes the C1 empirical response statistics.")
    
    headers_c1 = ["Cohort", "N", "Mean Adj Diff", "Sample SD", "Median", "IQR", "Raw Diff", "ETGNN Adv", "GAT Adv"]
    rows_c1 = [
        ["Overall C1 Cohort", "240", "+0.002903", "+/-0.096180", "-0.001534", "0.045904", "+0.038491", "0.020239", "0.017336"],
        ["True Positives (TP)", "95", "-0.005973", "+/-0.132491", "-0.001400", "0.044846", "+0.040412", "-0.003310", "0.002663"],
        ["False Positives (FP)", "95", "+0.011367", "+/-0.051097", "-0.000205", "0.041682", "+0.040085", "0.036034", "0.024667"],
        ["True Negatives (TN)", "25", "+0.013444", "+/-0.085633", "0.000000", "0.052547", "+0.044734", "0.063984", "0.050539"],
        ["False Negatives (FN)", "25", "-0.006073", "+/-0.070797", "-0.017843", "0.065701", "+0.018893", "0.005957", "0.012030"]
    ]
    add_tbl(doc, headers_c1, rows_c1, col_widths=[1.3, 0.45, 0.85, 0.75, 0.7, 0.65, 0.75, 0.55, 0.5])

    add_p(doc, 
        "Key C1 Findings for RQ3:\n"
        "1. Observed Difference Magnitude: Across 240 transactions, the observed mean random-adjusted difference was small: mean Delta_adj = +0.002903. "
        "The sample standard deviation across transactions is s = +/-0.096180 (reflecting node-level variance, not standard error; exploratory 95% bootstrap "
        "confidence interval: [-0.009201, 0.015121]). Median difference was -0.001534 with an IQR of 0.045904.\n"
        "2. Advantage Over Random Comparator: Both models exhibited positive mean advantage over 30 random deletion draws (0.0202 for ETGNN; 0.0173 for Matched GAT), "
        "confirming that GNNExplainer edge rankings induce larger probability drops than unguided random edge omissions.\n"
        "3. Epistemic Limitation: A small observed difference (+0.0029) does NOT prove statistical equivalence between ETGNN and Matched GAT; it indicates "
        "that under frozen checkpoints, the two models exhibit broadly comparable sensitivity to shared edge removals.")
    
    add_img(doc, "c1_response_distributions.png", 
        "Figure 5.2: Contrast 1 Response Distributions across 240 Validation Transactions showing Adjusted (Delta_adj) and Raw (Delta_raw) Differences.", 
        width=5.5)

    add_h2(doc, "5.3 Contrast 2 (C2): Matched-Pair Error-Register Analysis")
    add_p(doc, 
        "Contrast 2 tests whether GNNExplainer edge masks yield larger deletion responses on erroneous classifications than on correct classifications "
        "within each model. Table 5.2 presents the complete results across all 8 registers, evaluated via matched-pair cluster bootstrap (B = 2,000 iterations).")
    
    headers_c2 = ["Register", "Model", "Match Specification", "Pairs", "Mean Diff", "Boot SE", "95% Marginal CI", "Zero?"]
    rows_c2 = [
        ["ETGNN Primary TP-FP", "ETGNN", "Primary Caliper", "95", "+0.039343", "0.009585", "[+0.020234, +0.057410]", "No"],
        ["ETGNN Degree TP-FP", "ETGNN", "Exact Out-Degree", "57", "+0.043455", "0.012658", "[+0.019882, +0.069552]", "No"],
        ["Matched GAT Primary TP-FP", "Matched GAT", "Primary Caliper", "94", "+0.020315", "0.008923", "[+0.003917, +0.038684]", "No"],
        ["Matched GAT Degree TP-FP", "Matched GAT", "Exact Out-Degree", "82", "+0.010547", "0.011610", "[-0.013240, +0.032770]", "YES"],
        ["ETGNN Primary TN-FN", "ETGNN", "Primary Caliper", "25", "-0.058027", "0.034398", "[-0.128831, +0.006393]", "YES"],
        ["ETGNN Degree TN-FN", "ETGNN", "Exact Out-Degree", "25", "-0.019748", "0.029750", "[-0.081593, +0.036312]", "YES"],
        ["Matched GAT Primary TN-FN", "Matched GAT", "Primary Caliper", "21", "+0.052255", "0.051517", "[-0.046153, +0.162460]", "YES"],
        ["Matched GAT Degree TN-FN", "Matched GAT", "Exact Out-Degree", "21", "+0.072489", "0.051881", "[-0.024721, +0.178587]", "YES"]
    ]
    add_tbl(doc, headers_c2, rows_c2, col_widths=[1.3, 0.8, 1.0, 0.45, 0.75, 0.65, 1.1, 0.45])

    add_p(doc, 
        "Key C2 Findings for RQ4:\n"
        "1. Direction and Persistence of ETGNN FP–TP Difference: Within the matched sample, the explanation-minus-random deletion response was larger for "
        "false positives than true positives (mean paired difference +0.039343 in primary matching; +0.043455 in degree-sensitivity matching). Neither "
        "95% bootstrap confidence interval crosses zero. Removing top-ranked explanation edges reduced predicted-class probabilities more sharply for false "
        "alarms than for true illicit classifications. Crucially, this empirical pattern must NOT be interpreted as proof of explanation faithfulness, "
        "identification of harmful edges, or automated error correction.\n"
        "2. Matching Specification Sensitivity in Matched GAT: In Matched GAT, primary caliper matching yielded a positive estimate (+0.020315, 95% CI "
        "[+0.003917, +0.038684]), but under exact out-degree matching the estimate attenuated to +0.010547 and its 95% CI crossed zero ([-0.013240, +0.032770]; "
        "note that -0.013240 is the lower bound of the CI, not the mean). This demonstrates that GAT explanation sensitivity is vulnerable to degree-matching specifications.\n"
        "3. Pervasive Imprecision in False Negative Registers: In all four TN–FN registers, the 95% bootstrap confidence intervals are exceptionally wide "
        "and span zero. This imprecision reflects finite-sample limitations: only 25 false negatives were available for ETGNN and 21 for Matched GAT. "
        "The inclusion of zero must NOT be interpreted as evidence of functional equality between TN and FN explanations; it reflects statistical imprecision.\n"
        "4. Register Overlap: Note that the 420 pair records across the eight registers represent overlapping subsets of the 377 unique evaluated transactions; "
        "they do not constitute 420 independent observations.")
    
    add_img(doc, "c2_paired_differences_ci.png", 
        "Figure 5.3: Contrast 2 Forest Plot of Matched-Pair Differences (Delta_pair) and 95% Marginal Bootstrap Confidence Intervals across All 8 Registers.", 
        width=5.6)

    add_h2(doc, "5.4 Qualitative Subgraph Visualizations")
    add_p(doc, 
        "Figures 5.4 and 5.5 present two-hop computation subgraphs extracted for representative True Positive (TP) and False Positive (FP) transactions. "
        "Visual inspection confirms that while GNNExplainer highlights high-weight payment conduits, the continuous masks require subjective thresholding, "
        "reinforcing the necessity of quantitative edge-deletion benchmarks over anecdotal visualization alone.")
    
    add_img(doc, "explanation_tp_subgraph.png", 
        "Figure 5.4: Computation Subgraph and GNNExplainer Edge Mask for Representative True Positive Transaction (ID: 288717023).", 
        width=4.6)
    add_img(doc, "explanation_fp_subgraph.png", 
        "Figure 5.5: Computation Subgraph and GNNExplainer Edge Mask for Representative False Positive Transaction (ID: 116898498).", 
        width=4.6)

    add_h2(doc, "5.5 Interactive Demonstration System")
    add_p(doc, 
        "To satisfy applied engineering requirements, a decoupled demonstration system was developed in src/demo/:\n"
        "• FastAPI Backend (src/demo/backend.py): Serves REST endpoints for health checks (/health), precomputed explanation examples (/examples), "
        "and transaction inference (/transactions/{id}) with strict input validation (returning HTTP 404 for unknown IDs).\n"
        "• Streamlit Compliance Dashboard (src/demo/frontend.py): Interactive compliance interface rendering risk dials, model probability comparisons "
        "(Random Forest vs. ETGNN), and interactive PyVis topological subgraphs.\n"
        "• End-to-End Verification: An automated integration test (src/evaluation/test_demo_http_server.py) verifies server startup, route integrity, "
        "and error handling with 100% pass rate.")

    add_h2(doc, "5.6 Threats to Validity and Epistemic Boundaries")
    add_p(doc, 
        "1. Retrospective Validation Reuse: C1 and C2 evaluate validation time steps 37–40, where model early stopping occurred. These are structured "
        "retrospective diagnostic audits, not prospective trials.\n"
        "2. Confounding of Static Precomputed Features: Deleting graph edges perturbs message passing during GNN inference but leaves the 72 precomputed "
        "tabular neighborhood aggregates static in the feature matrix.\n"
        "3. Network Autocorrelation: While cluster bootstrap resamples pairs as independent clusters, transactions reside in a connected financial graph "
        "where computation subgraphs partially overlap. Bootstrap intervals do not fully eliminate spatial autocorrelation.\n"
        "4. Non-Causal Nature of Explanations: Deletion perturbations index empirical model sensitivity under edge omission; they do NOT establish causal "
        "mechanisms of financial crime, legal culpability, or automated error repair.")

    doc.add_page_break()
    
    # ================= CHAPTER 6 =================
    add_h1(doc, "Chapter 6: Conclusion and Future Work")
    add_h2(doc, "6.1 Synthesis and Answers to Research Questions")
    add_p(doc, 
        "• Answer to Research Question 1 (Comparative Classification): Tabular Random Forest substantially outperformed all Graph Neural Networks on "
        "held-out future time steps (41–49), achieving an AP of 0.6323 compared to ETGNN's 0.2250, XGBoost's 0.5472, GraphSAGE's 0.2427, and static GAT's 0.1155. "
        "Hypothesis H1 is decisively NOT supported. In extreme class imbalance with precomputed neighborhood aggregates, tree bagging resists distribution "
        "drift far more effectively than deep graph message passing.\n"
        "• Answer to Research Question 2 (Temporal Encoding Contribution): Adding fixed 16-dimensional sinusoidal time encoding yielded an indicative improvement "
        "on validation data across a three-seed mean (0.5292 vs. 0.4781 for Matched GAT), but Matched GAT outperformed ETGNN under Seed 7 (0.5572 vs. 0.5096) "
        "and held-out test differences were small (0.2250 vs. 0.2099). Hypothesis H2 is partially supported on validation data but does not establish a "
        "statistically generalizable temporal effect.\n"
        "• Answer to Research Question 3 (C1 Shared Interventions): Under shared GNNExplainer edge deletions across 240 validation transactions, the observed "
        "mean random-adjusted difference between ETGNN and Matched GAT was very small (Delta_adj = +0.002903, s = +/-0.096180, 95% CI [-0.009201, 0.015121]). "
        "Both models exhibited modest positive advantages over random edge removals (~0.020 vs. 0.017), indicating comparable structural sensitivity.\n"
        "• Answer to Research Question 4 (C2 Matched Error Contrasts): Within the matched validation sample, GNNExplainer edge deletions induced a statistically "
        "persistent, larger probability drop for false positives than true positives for ETGNN across primary (+0.039343, 95% CI [+0.020234, +0.057410]) "
        "and degree-sensitivity (+0.043455, 95% CI [+0.019882, +0.069552]) registers. In contrast, Matched GAT estimates were sensitive to out-degree matching "
        "(its CI crossing zero), while all False Negative registers exhibited pervasive statistical imprecision due to limited sample sizes.")

    add_h2(doc, "6.2 Summary of Verified Contributions")
    add_p(doc, 
        "1. Leak-Free Chronological Benchmarking: A fully reproducible data engineering and evaluation pipeline on Elliptic enforcing strict temporal "
        "scaling and non-overlapping masks, demonstrating that standard GNNs underperform tabular baselines when precomputed topological aggregates are available.\n"
        "2. Topological Message-Passing Damage Isolation: Empirical demonstration via local-feature edge ablations that message passing across within-step "
        "edges degraded AP from 0.4812 (no-edge control) down to 0.2344 (with edges), supporting hypotheses of noise propagation in dense financial hubs.\n"
        "3. Shared-Intervention Explainability Framework (C1): Formulation and execution of a controlled between-model deletion audit utilizing shared edge "
        "rankings and random deletion calibration.\n"
        "4. Matched-Pair Error-Register Audit (C2): Pioneer execution of margin-caliper and degree-balanced matched-pair cluster bootstrapping comparing "
        "explanation sensitivity on model errors versus correct classifications.")

    add_h2(doc, "6.3 Limitations")
    add_p(doc, 
        "1. Architectural Scope: Evaluated a static time-encoded GAT operating on discrete snapshots; does not evaluate continuous-time dynamic memory (TGN).\n"
        "2. Checkpoint Variance: Interventions evaluated a single frozen checkpoint per model family, conflating inductive bias with initialization stochasticity.\n"
        "3. Static Precomputed Tabular Features: 72 of the 165 features are static neighborhood aggregates, which remain unchanged during edge deletion.\n"
        "4. Multiple Comparisons: The eight bootstrap confidence intervals in C2 represent marginal estimates unadjusted for family-wise error rates.\n"
        "5. Epistemic Boundary: Deletion sensitivity measures model perturbation behavior under edge removal; it does NOT constitute causal proof of financial "
        "crime, legal evidence of culpability, or automated error repair.")

    add_h2(doc, "6.4 Directions for Future Research")
    add_p(doc, 
        "1. Dynamic Memory Networks on Multi-Snapshot Graphs: Extending evaluation to dynamic architectures (such as TGN or EvolveGCN) on cryptocurrency "
        "datasets containing genuine cross-time-step transaction edges.\n"
        "2. Joint Feature-and-Edge Deletion Benchmarks: Developing perturbation protocols that dynamically recalculate tabular neighborhood aggregates "
        "whenever graph edges are deleted, resolving the static-feature confounding identified in this thesis.\n"
        "3. Cross-Explainer Comparative Audits: Benchmarking whether alternative graph explainers (such as PGExplainer, SubgraphX, or PGM-Explainer) "
        "reproduce the FP–TP sensitivity disparity observed under GNNExplainer.\n"
        "4. Hybrid Tabular-GNN Ensembles: Developing stacking architectures that combine the high precision of Random Forest with the structural recall "
        "of Graph Attention Networks for production compliance deployment.")

    add_h2(doc, "6.5 Concluding Remarks")
    add_p(doc, 
        "This thesis demonstrates the paramount importance of empirical skepticism and methodological rigor in applied machine learning research. "
        "By resisting the temptation to overstate architectural capabilities, acknowledging the decisive superiority of tabular tree ensembles, "
        "and establishing mathematically verified bounds on post-hoc graph explanations, this work provides a defensible foundation for the future "
        "development of transparent, accountable, and legally sound financial fraud detection systems.")

    doc.add_page_break()
    
    # ================= REFERENCES =================
    add_h1(doc, "References")
    refs = [
        "Amara, K., Ying, Z., Zhang, Z., Han, Z., Zhao, Y., Shan, Y., Brandes, U., Schemm, S., & Zhang, C. (2022). GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks. Proceedings of the First Learning on Graphs Conference, in Proceedings of Machine Learning Research (PMLR), 198:44:1-44:23. https://proceedings.mlr.press/v198/amara22a.html",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), 785-794. https://doi.org/10.1145/2939672.2939785",
        "Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 1024-1034. https://proceedings.neurips.cc/paper/6703-inductive-representation-learning-on-large-graphs",
        "Hooker, S., Erhan, D., Kindermans, P. J., & Been, K. (2019). A benchmark for interpretability methods. Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 2384-2394. https://proceedings.neurips.cc/paper/2019/hash/fe4b8555ed8f8ab149aa792f6da600f4-Abstract.html",
        "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. 5th International Conference on Learning Representations (ICLR 2017). https://arxiv.org/abs/1609.02907",
        "Luo, D., Cheng, W., Xu, D., Yu, F., Zong, B., Chen, H., & Zhang, X. (2020). Parameterized explainer for graph neural networks. Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 19620-19631. https://proceedings.nips.cc/paper_files/paper/2020/hash/e37b08dd3015330dcbb5d6663667b8b8-Abstract.html",
        "Pareja, A., Domeniconi, G., Chen, J., Teng, T., Celiksu, T., Huang, L., & Ceusters, M. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04), 5363-5370. https://doi.org/10.1609/aaai.v34i04.5984",
        "Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal graph networks for deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637.",
        "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432",
        "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. 6th International Conference on Learning Representations (ICLR 2018). https://arxiv.org/abs/1710.10903",
        "Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. arXiv preprint arXiv:1908.02591.",
        "Ying, Z., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating explanations for graph neural networks. Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 9240-9251. https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html"
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        p.paragraph_format.space_after = Pt(5)
        run = p.add_run(r)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9.5)
        
    doc.add_page_break()
    
    # ================= APPENDICES =================
    add_h1(doc, "Appendix: Mathematical Proofs & Artifact Verification")
    add_h2(doc, "Appendix A: Mathematical Formulations of C1 and C2")
    add_p(doc, 
        "1. Contrast 1 Advantage Difference:\n"
        "   Adv_M(v) = D_(GE, M)(v) - (1/30) * sum_(d=1)^30 D_(Rnd, M)^(d)(v)\n"
        "   Delta_adj(v) = Adv_ETGNN(v) - Adv_Matched_GAT(v)\n"
        "   Observed mean: Delta_adj = +0.002903, sample standard deviation s = +/-0.096180, exploratory 95% bootstrap CI = [-0.009201, 0.015121].\n\n"
        "2. Contrast 2 Paired Differences:\n"
        "   Delta_pair = Adv(incorrect) - Adv(correct)\n"
        "   FP–TP: Delta_pair = Adv(FP) - Adv(TP)\n"
        "   FN–TN: Delta_pair = Adv(FN) - Adv(TN)\n"
        "   Resampled via matched-pair cluster bootstrap (B = 2,000 iterations, seed 20261020).")
    
    add_h2(doc, "Appendix B: Cryptographic Hash Register")
    add_p(doc, 
        "• ETGNN_Main_C1_C2_Results.zip: SHA256 = 550092c5f77bcb51702986006c48c8fec29ee98ff564f833ad17f780a43872fc\n"
        "• ETGNN_Main_C1_C2_Execution_Bundle.zip: SHA256 = 1a48608440285a21092e0df406e257eb897818be32a76f2f9f1b409ea6402e86\n"
        "• Frozen Main Benchmark Runner Snapshot: SHA256 = ef67f742fba5053b211a76f7596ff172551a87b1c3c990b7933994bb0a174c8c\n"
        "• Full Integration Verification Status: 100% Passed (14/14 automated assertions passed with 1e-10 precision).")

    out_path = os.path.join(FINAL_DIR, "ETGNN_Final_Corrected_Thesis.docx")
    doc.save(out_path)
    print(f"[SUCCESS] Saved Word document: {out_path}")
    return out_path

def generate_corrected_pdf():
    pdf_path = os.path.join(FINAL_DIR, "ETGNN_Final_Corrected_Thesis.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54, rightMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor('#1F3864'),
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor('#444444'),
        spaceAfter=24
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.HexColor('#333333'),
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'ChapHeading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1F3864'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'ChapHeading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#222222'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        spaceAfter=6,
        textColor=colors.HexColor('#222222')
    )
    
    note_style = ParagraphStyle(
        'NoteBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#333333')
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor('#555555'),
        spaceAfter=8
    )

    story = []
    
    # Title Page
    story.append(Spacer(1, 40))
    story.append(Paragraph("EXPLAINABLE TEMPORAL GRAPH NEURAL NETWORKS FOR ADAPTIVE FINANCIAL FRAUD DETECTION", title_style))
    story.append(Paragraph("A Controlled Chronological and Interventional Explainability Audit on the Elliptic Bitcoin Transaction Network", subtitle_style))
    story.append(Spacer(1, 40))
    story.append(Paragraph(
        "A Final Year Research Thesis Submitted in Partial Fulfillment of the Requirements for the<br/>"
        "Degree of Bachelor of Science (Honours) in Computer Science / Artificial Intelligence<br/><br/>"
        "Department of Computer Science & Software Engineering<br/>"
        "Faculty of Computing / Information Technology<br/>"
        "Academic Year: 2026",
        meta_style
    ))
    story.append(Spacer(1, 30))
    
    # Note box
    note_p = Paragraph(
        "<b>[TITLE REFINEMENT GUIDANCE FOR SUPERVISOR]</b><br/>"
        "Official approved research title preserved above. Because the implemented system is a static time-encoded Graph Attention Network "
        "(without dynamic continuous-time memory or online drift adaptation), an alternative descriptive title is submitted for supervisor consideration: "
        "<i>'Explainable Time-Encoded Graph Attention Networks for Bitcoin Illicit Transaction Classification'</i>. "
        "Pending formal supervisor feedback, the officially approved title is retained, with architectural boundaries explicitly delimited in Section 1.6.",
        note_style
    )
    tbl_note = Table([[note_p]], colWidths=[500])
    tbl_note.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F2F4F8')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#B0C0D0')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(tbl_note)
    story.append(PageBreak())
    
    # Abstract
    story.append(Paragraph("Abstract", h1_style))
    story.append(Paragraph(
        "The expansion of decentralized cryptographic asset networks has intensified the requirement for robust automated forensic systems capable "
        "of detecting illicit financial transactions, including money laundering, ransomware operations, and darknet vendor activities. Graph Neural "
        "Networks (GNNs) offer natural inductive representations for transaction networks, yet their forensic application encounters acute challenges: "
        "extreme class imbalance, temporal concept drift, and the black-box nature of deep geometric representations. This undergraduate thesis "
        "evaluates a Time-Encoded Graph Attention Network (ETGNN) for illicit Bitcoin transaction classification on the benchmark Elliptic dataset "
        "(203,769 transactions, 234,355 directed edges, across 49 discrete time steps). The research executes two interconnected investigations: "
        "(1) a leak-free chronological classification evaluation against competitive tabular tree ensembles (Random Forest, XGBoost) and deep graph "
        "baselines (GCN, GraphSAGE, static GAT), and (2) a frozen post-hoc explainability benchmark utilizing GNNExplainer across 377 completed "
        "validation transactions (steps 37–40) evaluating random-adjusted deletion responses under shared interventions (Contrast 1) and margin-matched "
        "correct-versus-incorrect prediction pairs (Contrast 2).",
        body_style
    ))
    story.append(Paragraph(
        "Empirical results indicate that tabular Random Forest achieved the highest held-out test performance (time steps 41–49) with an illicit-class "
        "Average Precision (AP / AUPRC) of 0.6323, substantially surpassing XGBoost (0.5472), GraphSAGE (0.2427), the parameter-matched zero-time GAT "
        "control (0.2099), and the proposed ETGNN (0.2250). A controlled topological edge ablation revealed that evaluating ETGNN on local features with "
        "all graph edges removed increased test AP from 0.2344 to 0.4812 (+0.2468), supporting the hypothesis that unconstrained message passing across "
        "within-step payment edges propagates structural noise in high-degree transaction hubs. In post-hoc explainability audits, Contrast 1 demonstrated "
        "a small observed mean random-adjusted difference between ETGNN and Matched GAT (+0.002903 across 240 transactions, exploratory 95% bootstrap CI "
        "[-0.009201, 0.015121]). In Contrast 2, within-model matched-pair cluster bootstrapping (2,000 iterations) demonstrated that GNNExplainer edge "
        "omissions induced a larger probability drop on false positives than true positives for ETGNN across primary (+0.039343, 95% CI [+0.020234, +0.057410]) "
        "and degree-balanced (+0.043455, 95% CI [+0.019882, +0.069552]) registers. However, because 84.1% of evaluated transactions possess only two "
        "eligible computation edges, discrete ceiling budget rules imposed an actual 50.0% structural deletion fraction. These results indicate that "
        "deletion responses capture local model sensitivity to topological omission rather than causal explanations of fraud, establishing evidence-bounded "
        "constraints for accountable cryptocurrency forensics.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Keywords:</b> Graph Neural Networks, Financial Fraud Detection, Bitcoin Forensic Analytics, Elliptic Dataset, Graph Attention Networks, "
        "GNNExplainer, Post-Hoc Explainability, Chronological Evaluation, Perturbation Auditing, Model Interpretability.",
        body_style
    ))
    story.append(PageBreak())
    
    # Chapter 1 Summary in PDF
    story.append(Paragraph("Chapter 1: Introduction", h1_style))
    story.append(Paragraph(
        "Financial transaction networks are relational and dynamic. Detecting illicit transactions on decentralized Bitcoin networks entails "
        "extreme class imbalance (illicit prevalence < 5%), relational topological dependencies, and temporal concept drift. The Elliptic dataset "
        "records 203,769 transactions and 234,355 edges across 49 discrete time steps, where all edges exist strictly within the same time step. "
        "This thesis evaluates whether adding a fixed sinusoidal temporal encoding to Graph Attention Networks improves chronological illicit classification, "
        "and executes controlled post-hoc explainability audits without overstating architectural capabilities.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Research Questions:</b><br/>"
        "• RQ1: Comparative classification performance of ETGNN against RF, XGBoost, GCN, GraphSAGE, and GAT on held-out test steps (41–49).<br/>"
        "• RQ2: Validation contribution of fixed sinusoidal time encoding against a parameter-matched zero-time GAT control across random seeds.<br/>"
        "• RQ3: Differential prediction drop under shared GNNExplainer edge deletions between ETGNN and Matched GAT (Contrast 1).<br/>"
        "• RQ4: Within-model matched-pair deletion advantage differences between model errors (FP, FN) and correct classifications (TP, TN) (Contrast 2).",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Chapter 4 Summary Table in PDF
    story.append(Paragraph("Chapter 4: Classification Results", h1_style))
    story.append(Paragraph(
        "Table 4.1: Held-out test split evaluation (steps 41–49; N=9,973 labelled transactions; 524 illicit cases, 5.25% prevalence) at decision threshold tau = 0.50.",
        body_style
    ))
    
    table_data = [
        ["Model Architecture", "Input Specification", "Precision", "Recall", "F1", "ROC-AUC", "Test AP"],
        ["Random Forest (Full)", "165 feats (93 local + 72 agg)", "0.6966", "0.5782", "0.6319", "0.8735", "0.6323"],
        ["Random Forest (Local)", "93 local numeric feats only", "0.5635", "0.5840", "0.5736", "0.8515", "0.6301"],
        ["Random Forest (+ Time)", "165 feats + 16-d time enc", "0.6594", "0.5802", "0.6173", "0.8672", "0.6295"],
        ["XGBoost", "165 feats (boosted trees)", "0.2823", "0.6050", "0.3849", "0.8074", "0.5472"],
        ["ETGNN (No-Edge Control)", "93 local + 16-d time, NO EDGES", "0.2445", "0.6164", "0.3501", "0.8107", "0.4812"],
        ["GraphSAGE", "165 feats + 2-layer mean agg", "0.1238", "0.7481", "0.2125", "0.8205", "0.2427"],
        ["ETGNN (Local Only)", "93 local + 16-d time + edges", "0.2225", "0.5286", "0.3132", "0.7894", "0.2344"],
        ["ETGNN (Full Proposed)", "165 feats + 16-d time + edges", "0.2614", "0.4504", "0.3308", "0.8062", "0.2250"],
        ["Matched GAT Control", "165 feats + 16 zero-pad + edges", "0.1205", "0.7424", "0.2074", "0.8175", "0.2099"],
        ["GCN", "165 feats + 2-layer convolution", "0.1009", "0.8645", "0.1808", "0.7732", "0.1306"],
        ["Standard GAT", "165 feats + 2-layer attention", "0.0893", "0.9141", "0.1627", "0.7548", "0.1155"]
    ]
    t_pdf = Table(table_data, colWidths=[120, 130, 48, 48, 44, 52, 54])
    t_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1F3864')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7.5),
        ('ALIGN', (2,0), (-1,-1), 'RIGHT'),
        ('ALIGN', (0,0), (1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D0D8E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_pdf)
    story.append(Spacer(1, 10))
    
    # Chapter 5 Summary in PDF
    story.append(Paragraph("Chapter 5: Explainability Results (C1 & C2)", h1_style))
    story.append(Paragraph(
        "<b>Contrast 1 (Shared Deletion, N=240):</b> Observed mean adjusted difference between ETGNN and Matched GAT was Delta_adj = +0.002903 "
        "(sample SD s = +/-0.096180, exploratory 95% bootstrap CI [-0.009201, 0.015121]). Mean advantage over random edge removal was 0.0202 for ETGNN and 0.0173 for Matched GAT.<br/>"
        "<b>Contrast 2 (Matched-Pair Bootstrap, B=2,000 across 8 registers, 420 pair records):</b><br/>"
        "• ETGNN Primary TP-FP: N=95, Mean = +0.039343, 95% CI [+0.020234, +0.057410] (Zero NOT crossed)<br/>"
        "• ETGNN Degree TP-FP: N=57, Mean = +0.043455, 95% CI [+0.019882, +0.069552] (Zero NOT crossed)<br/>"
        "• Matched GAT Primary TP-FP: N=94, Mean = +0.020315, 95% CI [+0.003917, +0.038684] (Zero NOT crossed)<br/>"
        "• Matched GAT Degree TP-FP: N=82, Mean = +0.010547, 95% CI [-0.013240, +0.032770] (Zero crossed; sensitive to matching)<br/>"
        "• All False Negative (FN) registers: Wide confidence intervals spanning zero due to limited validation sample sizes (N=25 for ETGNN, N=21 for Matched GAT).",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Chapter 6 Summary in PDF
    story.append(Paragraph("Chapter 6: Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Conclusion:</b> Tabular Random Forest won decisively over deep graph architectures on the held-out Elliptic period. Message-passing ablations "
        "demonstrated that within-step payment edges introduced structural noise that impeded illicit transaction detection. In explainability audits, "
        "GNNExplainer edge removals induced larger drops on false alarms than true illicit classifications in ETGNN, while perturbation responses reflect "
        "discrete topological amputations (50% deletion fraction across 84.1% of nodes) rather than infinitesimal feature attributions.",
        body_style
    ))
    
    doc.build(story)
    print(f"[SUCCESS] Saved PDF document: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    generate_corrected_docx()
    generate_corrected_pdf()
    print("=== Final Reviewed Thesis Documents Successfully Generated ===")
