import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    BG_DARK = RGBColor(15, 23, 42)      # Slate 900
    BG_LIGHT = RGBColor(248, 250, 252)  # Slate 50
    CARD_BG = RGBColor(255, 255, 255)   # White
    CARD_BORDER = RGBColor(226, 232, 240) # Slate 200
    TEXT_DARK = RGBColor(15, 23, 42)    # Slate 900
    TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
    TEXT_LIGHT = RGBColor(255, 255, 255)
    PRIMARY_BLUE = RGBColor(37, 99, 235) # Blue 600
    ACCENT_CYAN = RGBColor(14, 165, 233) # Sky 500
    ACCENT_GREEN = RGBColor(22, 163, 74) # Green 600
    ACCENT_RED = RGBColor(220, 38, 38)   # Red 600
    HEADER_BG = RGBColor(30, 41, 59)     # Slate 800

    def add_header(slide, title_text, category_text="NIBM BSc (Hons) in Computer Science with AI — Thesis Defense"):
        # Header container
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = BG_DARK
        top_bar.line.fill.background()

        # Category text
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.5), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(9.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_CYAN

        # Title text
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.5), Inches(0.6))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.name = "Arial"
        p_t.font.size = Pt(20)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_LIGHT

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        return card

    # ==========================================
    # SLIDE 1: Title Slide (Dark Theme)
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = BG_DARK
    bg1.line.fill.background()

    # Title box
    t_box1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(2.2))
    tf1 = t_box1.text_frame
    tf1.word_wrap = True
    
    p0 = tf1.paragraphs[0]
    p0.text = "NATIONAL INSTITUTE OF BUSINESS MANAGEMENT (NIBM) — KANDY INNOVATION CENTRE"
    p0.font.name = "Arial"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_CYAN
    p0.space_after = Pt(16)

    p1 = tf1.add_paragraph()
    p1.text = "Explainable Temporal Graph Neural Networks\nfor Financial Fraud Detection"
    p1.font.name = "Arial"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT
    p1.space_after = Pt(16)

    p2 = tf1.add_paragraph()
    p2.text = "An Empirical Investigation into Topological Noise, Financial Heterophily, and Counterfactual Faithfulness on the Elliptic Bitcoin Benchmark"
    p2.font.name = "Calibri"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(203, 213, 225)

    # Details Box
    d_card = add_card(s1, Inches(1.0), Inches(4.8), Inches(11.333), Inches(1.8), bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    d_box = s1.shapes.add_textbox(Inches(1.3), Inches(5.0), Inches(10.7), Inches(1.4))
    df = d_box.text_frame
    
    dp1 = df.paragraphs[0]
    dp1.text = "Candidate: P.V.H.S. Vitharama  |  Registration No: [STUDENT-ID-REDACTED]"
    dp1.font.name = "Arial"
    dp1.font.size = Pt(14)
    dp1.font.bold = True
    dp1.font.color.rgb = TEXT_LIGHT
    dp1.space_after = Pt(6)

    dp2 = df.add_paragraph()
    dp2.text = "Degree: Bachelor of Science (Hons) in Computer Science with Artificial Intelligence"
    dp2.font.name = "Calibri"
    dp2.font.size = Pt(12)
    dp2.font.color.rgb = RGBColor(203, 213, 225)
    dp2.space_after = Pt(4)

    dp3 = df.add_paragraph()
    dp3.text = "Department of Computing  |  Kandy Innovation Centre, NIBM Sri Lanka  |  Submission Date: 8 November 2026"
    dp3.font.name = "Calibri"
    dp3.font.size = Pt(11)
    dp3.font.color.rgb = ACCENT_CYAN

    # ==========================================
    # SLIDE 2: Background & Forensic Challenge
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Background: The Forensic Challenge in Cryptocurrency Networks")

    # 3 Cards
    c1 = add_card(s2, Inches(0.8), Inches(1.5), Inches(3.6), Inches(5.2))
    tb1 = s2.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(3.2), Inches(4.8))
    f1 = tb1.text_frame
    f1.word_wrap = True
    p = f1.paragraphs[0]
    p.text = "Macro Problem Scale"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(10)
    p = f1.add_paragraph()
    p.text = "• Money laundering accounts for 2–5% of global GDP ($800B – $2T annually; UNODC).\n• Bitcoin transaction pseudonymity creates serious forensic evasion pathways.\n• Massive volume: 203,769 transactions & 234,355 directed payment edges (Weber et al., 2019)."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    c2 = add_card(s2, Inches(4.8), Inches(1.5), Inches(3.6), Inches(5.2))
    tb2 = s2.shapes.add_textbox(Inches(5.0), Inches(1.7), Inches(3.2), Inches(4.8))
    f2 = tb2.text_frame
    f2.word_wrap = True
    p = f2.paragraphs[0]
    p.text = "Technical Complexities"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = ACCENT_RED
    p.space_after = Pt(10)
    p = f2.add_paragraph()
    p.text = "• Extreme Class Imbalance: Only 9.76% of labelled transactions are illicit (Davis & Goadrich, 2006; He & Garcia, 2009).\n• Sparse Partial Labels: 77.1% (157,205 transactions) are completely unlabelled.\n• Temporal Distribution Shifts: Fraudulent patterns evolve over 49 chronological snapshots."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    c3 = add_card(s2, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.2))
    tb3 = s2.shapes.add_textbox(Inches(9.0), Inches(1.7), Inches(3.3), Inches(4.8))
    f3 = tb3.text_frame
    f3.word_wrap = True
    p = f3.paragraphs[0]
    p.text = "The Core Dilemma"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = ACCENT_GREEN
    p.space_after = Pt(10)
    p = f3.add_paragraph()
    p.text = "• Relational message passing in GNNs assumes homophily ('fraudsters connect to fraudsters').\n• Financial graphs exhibit camouflage & heterophily (Dou et al., 2020; Liu et al., 2021).\n• Central Question: Does adding GNN layers actually add value, or does it dilute local features with topological noise?"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 3: Evidence-Based Research Gaps (WITH IN-TEXT CITATIONS)
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Evidence-Based Research Gaps (Strict Examiner Requirement)")

    gaps = [
        ("Research Gap 1: Inconclusive Value of Temporal Graph Attention on Future Data",
         "Prior Elliptic literature reported mixed results: Weber et al. (2019) showed Random Forest beating GCN; Alarab et al. (2020), Alarab & Prakoonwit (2023), and Lo et al. (2022) combined embeddings with linear/recurrent branches using F1 scores. However, whether deterministic sinusoidal snapshot time encoding in GAT genuinely outperforms strong tabular models under leak-free chronological AUPRC remained unestablished.",
         "References: (Weber et al., 2019; Alarab et al., 2020; Alarab & Prakoonwit, 2023; Lo et al., 2022; Veličković et al., 2018)"),
        ("Research Gap 2: Dependence on Topological Connectivity vs Feature Redundancy",
         "GNN fraud studies motivate relational message passing (Dou et al., 2020; Liu et al., 2021). However, the Elliptic benchmark already supplies 72 precomputed neighbourhood aggregates. It remained untested whether explicit payment edges add genuine predictive value or act as topological noise under controlled empty-edge and rewired ablations.",
         "References: (Weber et al., 2019; Dou et al., 2020; Liu et al., 2021; Hamilton et al., 2017)"),
        ("Research Gap 3: Lack of Controlled Intervention Faithfulness for Graph XAI",
         "Post-hoc explainers like GNNExplainer (Ying et al., 2019) produce visually plausible attribution subgraphs. However, their true faithfulness was rarely tested against random edge deletions or across distinct prediction-error cohorts (False Positives vs True Positives) under standardized intervention budgets.",
         "References: (Ying et al., 2019; Amara et al., 2022; Rao et al., 2022)")
    ]

    for idx, (title, desc, refs) in enumerate(gaps):
        top_y = Inches(1.5 + idx * 1.85)
        add_card(s3, Inches(0.8), top_y, Inches(11.733), Inches(1.65))
        tb = s3.shapes.add_textbox(Inches(1.0), top_y + Inches(0.1), Inches(11.3), Inches(1.45))
        tf = tb.text_frame
        tf.word_wrap = True
        
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.name = "Arial"
        pt.font.size = Pt(13.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_BLUE
        pt.space_after = Pt(4)

        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.name = "Calibri"
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_DARK
        pd.space_after = Pt(3)

        pr = tf.add_paragraph()
        pr.text = refs
        pr.font.name = "Arial"
        pr.font.size = Pt(9.5)
        pr.font.bold = True
        pr.font.color.rgb = ACCENT_RED

    # ==========================================
    # SLIDE 4: Research Questions & Objectives
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Research Questions (RQs) & Objectives (ROs) Alignment")

    # Left card: RQs, Right card: ROs
    add_card(s4, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tbrq = s4.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tfrq = tbrq.text_frame
    tfrq.word_wrap = True
    p = tfrq.paragraphs[0]
    p.text = "Research Questions (RQs)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(10)

    rqs = [
        "RQ1 (Predictive Value): Does snapshot time-encoded ETGNN outperform tabular baselines (RF, XGBoost) and standard GNNs under chronological AUPRC?",
        "RQ2 (Graph Contribution): How do empty-edge and rewired graphs affect performance when compared to observed connectivity?",
        "RQ3 (Explanation Sensitivity): Does GNNExplainer top-1 deletion yield larger prediction drops than random deletions, and how do FPs vs TPs respond?",
        "RQ4 (Statistical Robustness): Which performance differences remain statistically significant after 5-seed uncertainty and Holm correction?",
        "RQ5 (Secondary Transfer): Does performance transfer to the Elliptic++ wallet representation without temporal leakage?"
    ]
    for rq in rqs:
        p = tfrq.add_paragraph()
        p.text = rq
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    add_card(s4, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tbro = s4.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tfro = tbro.text_frame
    tfro.word_wrap = True
    p = tfro.paragraphs[0]
    p.text = "Research Objectives (ROs)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = ACCENT_GREEN
    p.space_after = Pt(10)

    ros = [
        "RO1: Implement and evaluate canonical snapshot ETGNN against Random Forest, XGBoost, GCN, GraphSAGE, and GAT baselines.",
        "RO2: Quantify the effects of observed, empty, and rewired graphs across 93 local vs 165 total feature configurations.",
        "RO3: Evaluate explanation sensitivity using GNNExplainer, random edge deletion, Leave-One-Out search, and matched FP/TP cohorts.",
        "RO4: Determine statistical robustness across repeated random seeds (seeds 42–46) using Holm multiplicity correction.",
        "RO5: Assess cross-representation generalizability and potential feature leakage on the Elliptic++ actor dataset."
    ]
    for ro in ros:
        p = tfro.add_paragraph()
        p.text = ro
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # ==========================================
    # SLIDE 5: Full Traceability Matrix Table
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Research Traceability Matrix (Gap ➔ RQ ➔ RO ➔ Method ➔ Result ➔ Conclusion)")

    # Table layout
    rows, cols = 6, 6
    tbl_shape = s5.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.4))
    tbl = tbl_shape.table

    # Column widths
    tbl.columns[0].width = Inches(1.6)
    tbl.columns[1].width = Inches(1.2)
    tbl.columns[2].width = Inches(1.5)
    tbl.columns[3].width = Inches(2.6)
    tbl.columns[4].width = Inches(2.5)
    tbl.columns[5].width = Inches(2.333)

    headers = ["Research Gap", "RQ", "Objective", "Methodology", "Key Result", "Conclusion"]
    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.font.name = "Arial"
            p.font.size = Pt(10.5)
            p.font.bold = True
            p.font.color.rgb = TEXT_LIGHT

    matrix_data = [
        ("Gap 1 (Predictive Value)", "RQ1", "RO1: Baseline Benchmark", "Chronological split (1-32 / 33-40 / 41-49); ETGNN vs RF, XGB, GCN, SAGE", "RF 0.6323 beats ETGNN 0.2250 (+181% margin)", "Tabular models superior; GNN struggles with heterophily"),
        ("Gap 2 (Graph Reliance)", "RQ2", "RO2: Graph Ablation", "Retrained empty-graph and degree-preserving rewired models (93 vs 165f)", "Empty graph surges to 0.4812 (+0.2468 gain); Rewired p=0.1223", "Edges act as topological noise, diluting local signals"),
        ("Gap 3 (XAI Fidelity)", "RQ3", "RO3: Explanation Audit", "GNNExplainer Top-1 vs Random vs LOO; C2 matched FP vs TP cohorts", "Top-1 (0.015) = Random (0.017; p=0.68); FP flip OR = 2.99", "Post-hoc masks lack faithfulness; FPs 3x more fragile"),
        ("Gap 1, 2 Robustness", "RQ4", "RO4: Statistical Audit", "5-seed evaluation (seeds 42-46); Holm family-wise correction", "Residual ETGNN 0.3126 ± 0.0559; Deep MLP 0.3380 ± 0.0388", "Run-to-run variation high; gating adds no benefit"),
        ("Transferability Limits", "RQ5", "RO5: Elliptic++ Stress Test", "Actors graph evaluation (822k nodes, 56f); provenance audit", "Test AUPRC ~0.097; 33.0% illicit prevalence drop", "Cross-dataset transfer fragile; future leakage risk")
    ]

    for r_idx, row in enumerate(matrix_data, start=1):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 1 else RGBColor(241, 245, 249)
            for p in cell.text_frame.paragraphs:
                p.font.name = "Calibri"
                p.font.size = Pt(9.5)
                p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 6: Dataset & Chronological Protocol
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Dataset Profile & Chronological Evaluation Protocol")

    # Left card: Table, Right card: Key rules
    add_card(s6, Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.4))
    tbd = s6.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.6), Inches(5.0))
    tfd = tbd.text_frame
    tfd.word_wrap = True
    p = tfd.paragraphs[0]
    p.text = "Elliptic Benchmark Partitions (Table 4.1)"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    p = tfd.add_paragraph()
    p.text = "• Whole Benchmark (Steps 1–49): 203,769 nodes, 234,355 edges, 46,564 labelled (4,545 illicit = 9.76%).\n\n• Training Period (Steps 1–32):\n  - Labelled Nodes: 28,938\n  - Illicit Nodes: 3,402 (11.76% Illicit Share)\n\n• Validation Period (Steps 33–40):\n  - Labelled Nodes: 7,653\n  - Illicit Nodes: 619 (8.09% Illicit Share)\n\n• Held-Out Test Period (Steps 41–49):\n  - Labelled Nodes: 9,973\n  - Illicit Nodes: 524 (5.25% Illicit Share)"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    add_card(s6, Inches(7.1), Inches(1.5), Inches(5.4), Inches(5.4))
    tbr = s6.shapes.add_textbox(Inches(7.3), Inches(1.7), Inches(5.0), Inches(5.0))
    tfr = tbr.text_frame
    tfr.word_wrap = True
    p = tfr.paragraphs[0]
    p.text = "Methodological Safeguards"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = ACCENT_RED
    p.space_after = Pt(8)

    p = tfr.add_paragraph()
    p.text = "1. Strict Chronological Splitting:\n   Evaluated strictly on future snapshots (41–49). No random k-fold mixing to prevent temporal data leakage.\n\n2. Leak-Free Normalization:\n   StandardScaler fitted incrementally via partial_fit() strictly on snapshots 1–32. Zero test data snooping.\n\n3. Unlabelled Nodes (157,205):\n   Never treated as licit transactions. Maintained purely for structural message passing.\n\n4. Snapshot Edge Restriction:\n   Message passing restricted to intra-timestep subgraphs (t_u = t_v). No inter-timestep forward leakage."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 7: EDA Structural Collapse at Step 43
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Exploratory Data Analysis: Structural Collapse at Snapshot 43")

    # Embed Figure 4.1
    fig41_path = "results/figures/figure_4_1_temporal_eda.png"
    if os.path.exists(fig41_path):
        s7.shapes.add_picture(fig41_path, Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.4))
    else:
        add_card(s7, Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.4))

    # Right Card: Insights
    add_card(s7, Inches(8.6), Inches(1.5), Inches(3.9), Inches(5.4))
    tb_eda = s7.shapes.add_textbox(Inches(8.8), Inches(1.7), Inches(3.5), Inches(5.0))
    tf_eda = tb_eda.text_frame
    tf_eda.word_wrap = True
    p = tf_eda.paragraphs[0]
    p.text = "Key EDA Findings"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    p = tf_eda.add_paragraph()
    p.text = "• Darknet Market Takedown (Step 43):\n  Coincided with a major law enforcement shutdown of a darknet entity (Weber et al., 2019).\n\n• Degree Collapse:\n  Mean degree dropped by 75.5% (from 4.82 to 1.18).\n\n• Isolated Nodes Surge:\n  Nodes with zero candidate edges rose from 13.5% to 76.9%!\n\n• Consequence for GNNs:\n  GNN convolutions starved of message-passing edges, explaining sharp performance drop in test periods."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 8: Proposed ETGNN Architecture
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Model Architecture: Time-Encoded Graph Attention Network")

    # Embed Architecture Figure 3.2
    fig32_path = "results/figures/fig2_model_architectures_comparison.png"
    if not os.path.exists(fig32_path):
        fig32_path = "results/figures/figure_3_5_demo_interface.png" # fallback

    add_card(s8, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.4))
    tb_arch = s8.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(6.4), Inches(5.0))
    tf_arch = tb_arch.text_frame
    tf_arch.word_wrap = True
    p = tf_arch.paragraphs[0]
    p.text = "Architectural Formulation"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    p = tf_arch.add_paragraph()
    p.text = "1. Input Concatenation:\n   165 transaction features + 16-dimensional sinusoidal time encoding = 181 input dimensions.\n\n2. Sinusoidal Time Encoding (Vaswani et al., 2017):\n   TE(t, 2i) = sin(t / 10000^(2i/16))\n   TE(t, 2i+1) = cos(t / 10000^(2i/16))\n   Encodes discrete snapshot position t in [1, 49].\n\n3. Two-Layer GAT (Veličković et al., 2018):\n   - Layer 1: 4 attention heads (head dim 16 -> 64 hidden dim), ELU activation, dropout 0.3.\n   - Layer 2: 1 attention head -> 2 class logits.\n\n4. Proposed Ego-Residual Repair (M1):\n   h_v = GATConv(x_v, E) + W_res * x_v\n   Preserves distinctive local ego-features."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # Right Card: Hyperparameters (Table 3.4)
    add_card(s8, Inches(7.9), Inches(1.5), Inches(4.6), Inches(5.4))
    tb_hyp = s8.shapes.add_textbox(Inches(8.1), Inches(1.7), Inches(4.2), Inches(5.0))
    tf_hyp = tb_hyp.text_frame
    tf_hyp.word_wrap = True
    p = tf_hyp.paragraphs[0]
    p.text = "Implementation Specifications (Table 3.4)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = HEADER_BG
    p.space_after = Pt(8)

    p = tf_hyp.add_paragraph()
    p.text = "• Optimizer: Adam (Kingma & Ba, 2015)\n• Learning Rate: 0.001 (1e-3)\n• Weight Decay: 1e-4\n• Max Epochs: 100 with Early Stopping (patience 15 on validation AUPRC)\n• Loss Functions: Cross-Entropy & Focal Loss (gamma=2, alpha=0.25; Lin et al., 2017)\n• Decision Boundary: Canonical tau = 0.50\n• Framework: PyTorch Geometric 2.5 + PyTorch 2.3\n• Compute Hardware: Google Colab NVIDIA Tesla T4 GPU (16 GB VRAM)"
    p.font.size = Pt(10.5)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 9: Methodology Mapped to Objectives
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Methodology: 'What I Did and How I Did It' (Objective-by-Objective)")

    methods = [
        ("Objective 1 (Benchmark)", "Trained & evaluated 7 baseline architectures on held-out test snapshots 41–49: Random Forest (100 trees, depth 12), XGBoost (depth 6, lr 0.1), GCN (2-layer), GraphSAGE (mean pool), GAT, Matched Zero-Time GAT, and Canonical ETGNN. Evaluated on full 165 features."),
        ("Objective 2 (Ablations)", "Engineered controlled structural ablations: (1) Empty-Edge Retraining (complete topological deletion) on 93 local features; (2) Degree-Preserving Rewired Graph (scrambled edges preserving node in/out degrees); (3) 93 local vs 165 total feature factorial split."),
        ("Objective 3 (XAI Audit)", "Implemented post-hoc explainability protocols: (1) Phase H 30-node validation pilot comparing GNNExplainer Top-1 edge deletion vs Random deletion vs Leave-One-Out search; (2) C2 Matched Validation Cohort (275 False Positives vs 275 True Positives matched on degree and margin)."),
        ("Objective 4 (Robustness)", "Executed 5-seed uncertainty trials (seeds 42–46) across 5 neural configurations (Residual, Deep MLP, Gated CE, Gated Focal, NoGate). Computed Sample SD, SE, and 95% CIs. Controlled family-wise error rate using Holm-Bonferroni correction (Holm, 1979)."),
        ("Objective 5 (Transfer)", "Stress-tested model transfer on secondary Elliptic++ Actors Graph (822k wallets, 56 features). Investigated potential future-block data leakage in wallet-lifetime aggregates (first_block, last_block, lifetime_in_blocks).")
    ]

    for idx, (title, desc) in enumerate(methods):
        top_y = Inches(1.4 + idx * 1.15)
        add_card(s9, Inches(0.8), top_y, Inches(11.733), Inches(1.05))
        tb = s9.shapes.add_textbox(Inches(1.0), top_y + Inches(0.05), Inches(11.3), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.bold = True
        pt.font.size = Pt(13)
        pt.font.color.rgb = PRIMARY_BLUE
        pt.space_after = Pt(2)
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 10: Interactive Demonstration Platform
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. System Implementation: Interactive Demonstration Platform")

    fig35_path = "results/figures/figure_3_5_demo_interface.png"
    if os.path.exists(fig35_path):
        s10.shapes.add_picture(fig35_path, Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.4))
    else:
        add_card(s10, Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.4))

    add_card(s10, Inches(8.6), Inches(1.5), Inches(3.9), Inches(5.4))
    tb_demo = s10.shapes.add_textbox(Inches(8.8), Inches(1.7), Inches(3.5), Inches(5.0))
    tf_demo = tb_demo.text_frame
    tf_demo.word_wrap = True
    p = tf_demo.paragraphs[0]
    p.text = "Platform Architecture"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    p = tf_demo.add_paragraph()
    p.text = "• Backend (FastAPI - Port 8000):\n  - PyTorch Geometric model inference\n  - Real-time GNNExplainer optimization\n  - In-memory 203k node graph indexing\n  - Fully covered by 27 automated tests\n\n• Frontend (React / Vite - Port 5173):\n  - Interactive 2-Hop Subgraph Canvas\n  - Transaction Risk Explorer\n  - Counterfactual 'Re-run Intervention' button\n  - Live Feature Sensitivity Sliders"
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 11: Objective 1 Results — Main Benchmark
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "10. Objective 1 Results: Main Benchmark Classification (Steps 41–49)")

    # Table 4.2 & 4.2a combined
    rows, cols = 8, 6
    tbl_shape = s11.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.733), Inches(4.5))
    tbl = tbl_shape.table

    tbl.columns[0].width = Inches(2.6)
    tbl.columns[1].width = Inches(1.8)
    tbl.columns[2].width = Inches(1.8)
    tbl.columns[3].width = Inches(1.8)
    tbl.columns[4].width = Inches(1.8)
    tbl.columns[5].width = Inches(1.933)

    b_headers = ["Model Architecture", "Precision", "Recall", "F1-Score", "ROC-AUC", "Test AUPRC"]
    for c_idx, h in enumerate(b_headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.font.name = "Arial"
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = TEXT_LIGHT

    b_data = [
        ("Random Forest (165 features)", "0.6966", "0.5782", "0.6319", "0.8735", "0.6323"),
        ("XGBoost (165 features)", "0.2823", "0.6050", "0.3849", "0.8074", "0.5472"),
        ("GraphSAGE (Hamilton et al., 2017)", "0.1238", "0.7481", "0.2125", "0.8205", "0.2427"),
        ("Canonical ETGNN (M0 checkpoint)", "0.2614", "0.4504", "0.3308", "0.8062", "0.2250"),
        ("Matched Zero-Time GAT (Control)", "0.1205", "0.7424", "0.2074", "0.8175", "0.2099"),
        ("GCN (Kipf & Welling, 2017)", "0.1009", "0.8645", "0.1808", "0.7732", "0.1306"),
        ("Static GAT (Veličković et al., 2018)", "0.0893", "0.9141", "0.1627", "0.7548", "0.1155")
    ]

    for r_idx, row in enumerate(b_data, start=1):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            if r_idx == 1:
                cell.fill.fore_color.rgb = RGBColor(220, 252, 231) # highlight RF
            else:
                cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 1 else RGBColor(241, 245, 249)
            for p in cell.text_frame.paragraphs:
                p.font.name = "Calibri"
                p.font.size = Pt(10.5)
                p.font.bold = (r_idx == 1)
                p.font.color.rgb = ACCENT_GREEN if (r_idx == 1 and c_idx == 5) else TEXT_DARK

    # Summary callout at bottom
    add_card(s11, Inches(0.8), Inches(6.2), Inches(11.733), Inches(0.85), bg_color=RGBColor(239, 246, 255), border_color=PRIMARY_BLUE)
    tbc = s11.shapes.add_textbox(Inches(1.0), Inches(6.25), Inches(11.3), Inches(0.75))
    tfc = tbc.text_frame
    tfc.word_wrap = True
    p = tfc.paragraphs[0]
    p.text = "Key Finding: Random Forest achieves 0.6323 AUPRC, heavily outperforming canonical ETGNN (0.2250) by +181.0% relative gain. Graph attention does not beat strong tabular models on future-period financial transactions."
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = PRIMARY_BLUE

    # ==========================================
    # SLIDE 12: Why Tabular Beat Graph (Heterophily & Smearing)
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "11. Theoretical Defense: Why Tabular Beat Graph Models")

    reasons = [
        ("1. Neighborhood Smearing in Imbalanced Graphs",
         "The Elliptic network is ~95% licit. When a GNN aggregates 1-hop and 2-hop neighbor vectors (sum/mean pooling), the features of rare illicit nodes are mathematically diluted by dominant licit neighbor embeddings, pushing illicit representations into the majority licit cluster."),
        ("2. Financial Graph Heterophily (Camouflage)",
         "Unlike citation networks where homophily holds ('similar papers cite each other'), financial money launderers intentionally route funds through legitimate intermediaries to camouflage activity (Dou et al., 2020; Lo et al., 2022). Message passing across heterophilous edges acts as topological noise."),
        ("3. Precomputed Neighborhood Feature Redundancy",
         "The Elliptic dataset already supplies 72 aggregate features (e.g. transacted volume sum, min, max, neighbor fee ratios) generated from the graph. Tabular decision trees split on these non-linear features directly without suffering from GNN oversmoothing or structural dilution.")
    ]

    for idx, (title, desc) in enumerate(reasons):
        top_y = Inches(1.5 + idx * 1.85)
        add_card(s12, Inches(0.8), top_y, Inches(11.733), Inches(1.65))
        tb = s12.shapes.add_textbox(Inches(1.0), top_y + Inches(0.1), Inches(11.3), Inches(1.45))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.bold = True
        pt.font.size = Pt(14)
        pt.font.color.rgb = PRIMARY_BLUE
        pt.space_after = Pt(4)
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11.5)
        pd.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 13: Objective 2 Results — Graph Ablations
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "12. Objective 2 Results: Controlled Graph Ablations (Topological Noise)")

    # Left Card: The Smoking Gun
    add_card(s13, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_ab1 = s13.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_ab1 = tb_ab1.text_frame
    tf_ab1.word_wrap = True
    p = tf_ab1.paragraphs[0]
    p.text = "The 'Smoking Gun' Ablation (Table 4.3)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = ACCENT_RED
    p.space_after = Pt(10)

    p = tf_ab1.add_paragraph()
    p.text = "• Experiment Condition:\n  Non-residual ETGNN evaluated on 93 local features on held-out test steps 41–49.\n\n• ETGNN with Observed Edges:\n  Test AUPRC = 0.2344\n\n• ETGNN Retrained with NO EDGES (Empty Graph):\n  Test AUPRC = 0.4812 (+0.2468 gain!)\n\n• Scientific Proof:\n  Removing all edges more than doubles performance (+105.3% relative surge)! This proves payment edges introduce destructive topological noise during standard message passing."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # Right Card: Rewired & Residual Controls
    add_card(s13, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_ab2 = s13.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_ab2 = tb_ab2.text_frame
    tf_ab2.word_wrap = True
    p = tf_ab2.paragraphs[0]
    p.text = "Degree-Preserving Rewired Control"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(10)

    p = tf_ab2.add_paragraph()
    p.text = "• Phase 4 Factorial Validation (5 seeds):\n  - Residual True Graph (93f): 0.7758 ± 0.0111\n  - Residual Rewired Graph (93f): 0.7252 ± 0.0420\n  - Difference: +0.0506 (unadjusted p = 0.0245)\n\n• Multiplicity Reality Check:\n  After Holm-Bonferroni correction over the 6 comparison contrasts, p = 0.1223 (not statistically significant at alpha = 0.05)!\n\n• Full 165 Features Validation:\n  Deep MLP (0.7189) outperforms Residual GNN (0.5812; Holm p = 0.0498)."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 14: Objective 3 Results — Explainability Audit
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "13. Objective 3 Results: GNNExplainer Fidelity & Matched Cohort Analysis")

    # Left Card: Phase H Pilot
    add_card(s14, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_x1 = s14.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_x1 = tb_x1.text_frame
    tf_x1.word_wrap = True
    p = tf_x1.paragraphs[0]
    p.text = "Phase H: Explanation Sensitivity (Table 4.4)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(10)

    p = tf_x1.add_paragraph()
    p.text = "• Evaluated on 30 stratified multi-edge validation nodes.\n\n• GNNExplainer Top-1 Deletion:\n  Mean illicit probability shift = 0.0150 (260.5 ms runtime).\n\n• Random Single-Edge Deletion:\n  Mean illicit probability shift = 0.0170 (1.1 ms runtime).\n\n• Paired t-test: p = 0.6827 (no significant difference)!\n  GNNExplainer Top-1 edges do not produce larger prediction drops than random single-edge deletions."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # Right Card: C2 Matched Cohort
    add_card(s14, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_x2 = s14.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_x2 = tb_x2.text_frame
    tf_x2.word_wrap = True
    p = tf_x2.paragraphs[0]
    p.text = "C2 Matched Error Cohorts (Table D.1)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = ACCENT_GREEN
    p.space_after = Pt(10)

    p = tf_x2.add_paragraph()
    p.text = "• 275 False Positives vs 275 True Positives matched on timestep and candidate degree.\n\n• Classification Flips after 1-Hop Edge Deletion:\n  - False Positives: 33 / 275 flipped to licit (12.00%)\n  - True Positives: 12 / 275 flipped to licit (4.36%)\n\n• Statistical Significance:\n  - Odds Ratio (OR) = 2.99 (95% CI [1.51, 5.92])\n  - Fisher Exact Test: p = 0.0016 (Holm p = 0.0032)\n  - Conclusion: False positives are nearly 3x more fragile to edge deletion than true positives."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 15: Objectives 4 & 5 — Robustness & Transfer
    # ==========================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "14. Objectives 4 & 5: Statistical Robustness & Secondary Transfer")

    # Left: 5-seed neural results
    add_card(s15, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_s1 = s15.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_s1 = tb_s1.text_frame
    tf_s1.word_wrap = True
    p = tf_s1.paragraphs[0]
    p.text = "Objective 4: Five-Seed Test Audit (Table C.1)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(10)

    p = tf_s1.add_paragraph()
    p.text = "• Multi-seed evaluation across seeds 42–46 (Test steps 41–49):\n  - M2 Deep MLP: 0.3380 ± 0.0388\n  - M4 NoGate Control: 0.3127 ± 0.0508\n  - M1 Residual ETGNN: 0.3126 ± 0.0559 (95% CI [0.243, 0.382])\n  - M3 Reliability-Gated CE: 0.2419 ± 0.0466\n  - M3 Reliability-Gated Focal: 0.2273 ± 0.0357\n\n• Insights:\n  - Feature-only Deep MLP beats all neural graph variants.\n  - Gate values collapsed near zero; gating adds no value."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # Right: Elliptic++
    add_card(s15, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_s2 = s15.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_s2 = tb_s2.text_frame
    tf_s2.word_wrap = True
    p = tf_s2.paragraphs[0]
    p.text = "Objective 5: Elliptic++ Transfer (Table 4.5)"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = ACCENT_RED
    p.space_after = Pt(10)

    p = tf_s2.add_paragraph()
    p.text = "• Secondary Actors Benchmark (822k wallets, 56 features):\n  - Deep Feature MLP: 0.0991 ± 0.0268\n  - Residual ETGNN: 0.0968 ± 0.0214\n\n• Prevalence Collapse:\n  Illicit rate dropped by 33.0% (6.48% validation -> 4.34% test).\n\n• Provenance Audit (Future Leakage):\n  Features such as last_block_appeared_in and lifetime_in_blocks incorporate future activity, restricting claims of leakage-free production transfer."
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 16: Limitations & Threats to Validity
    # ==========================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "15. Limitations & Threats to Validity (Point-Form Summary)")

    lims = [
        ("1. Temporal Encoding Contract", "Historical M0 checkpoint used interleaved sinusoidal encoding, whereas later Colab evaluations used block-concatenation; they represent different input formats rather than matched architectural comparisons."),
        ("2. Multi-Seed Uncertainty", "5 random seeds (42–46) capture variance across initialization but do not establish asymptotic stability across all financial domains."),
        ("3. Statistical Dependence", "Matched C2 validation pairs share localized subgraphs; observations are not strictly independent i.i.d. draws."),
        ("4. Sparse Edge Budgets", "317 of 377 explanation transactions had only 2 candidate edges. Ceiling rule k = ceil(0.25*2) caused an actual 50% edge deletion (2x nominal budget)."),
        ("5. Model Sensitivity != Real-World Fraud", "Edge deletion tests model prediction sensitivity under topological omission; it does NOT constitute legal, judicial, or causal proof of criminal conspiracy."),
        ("6. Single Primary Benchmark", "Evaluated on Elliptic Bitcoin data; findings cannot be generalized to traditional banking networks (SWIFT, ACH) without validation.")
    ]

    for idx, (title, desc) in enumerate(lims):
        top_y = Inches(1.4 + idx * 0.95)
        add_card(s16, Inches(0.8), top_y, Inches(11.733), Inches(0.85))
        tb = s16.shapes.add_textbox(Inches(1.0), top_y + Inches(0.04), Inches(11.3), Inches(0.77))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.bold = True
        pt.font.size = Pt(12)
        pt.font.color.rgb = ACCENT_RED
        pt.space_after = Pt(1)
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10)
        pd.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 17: Separate Conclusions by Objective (MANDATORY)
    # ==========================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "16. Conclusions by Research Objective (Strict Examiner Requirement)")

    conclusions = [
        ("Addressing Objective 1 (RO1 — Predictive Performance):",
         "Under chronological testing, Random Forest achieved 0.6323 AUPRC, heavily outperforming canonical ETGNN (0.2250) and all GNN variants. Snapshot time-encoded graph attention did NOT achieve superior ranking performance."),
        ("Addressing Objective 2 (RO2 — Graph & Feature Contributions):",
         "Payment edges act as topological noise. Retraining non-residual ETGNN without edges surged test AUPRC from 0.2344 to 0.4812 (+0.2468 gain), while the true vs rewired contrast was non-significant after Holm correction (p = 0.1223)."),
        ("Addressing Objective 3 (RO3 — Explanation Reliability & Error Sensitivity):",
         "GNNExplainer Top-1 deletion did not produce larger probability drops than random edge deletion (p = 0.6827). However, in matched cohorts, False Positives were nearly 3x more likely to flip to licit than True Positives (12.00% vs 4.36%; OR 2.99, p = 0.0016)."),
        ("Addressing Objective 4 (RO4 — Statistical & Methodological Robustness):",
         "5-seed trials demonstrated that feature-only Deep MLP (0.3380) outperformed all neural GNNs, and reliability gating collapsed. Historical time-encoding mismatches restrict claims of architectural superiority."),
        ("Addressing Objective 5 (RO5 — Cross-Representation Generalization):",
         "Elliptic++ transfer resulted in low test AUPRC (~0.097) due to a 33.0% illicit prevalence drop and potential future-block data leakage in wallet-lifetime features.")
    ]

    for idx, (title, desc) in enumerate(conclusions):
        top_y = Inches(1.4 + idx * 1.15)
        add_card(s17, Inches(0.8), top_y, Inches(11.733), Inches(1.05), bg_color=RGBColor(240, 253, 244), border_color=ACCENT_GREEN)
        tb = s17.shapes.add_textbox(Inches(1.0), top_y + Inches(0.05), Inches(11.3), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.bold = True
        pt.font.size = Pt(12.5)
        pt.font.color.rgb = ACCENT_GREEN
        pt.space_after = Pt(2)
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_DARK

    # ==========================================
    # SLIDE 18: Future Work & Key References
    # ==========================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "17. Future Work & Key Academic References")

    # Left: Future work
    add_card(s18, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_fw = s18.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_fw = tb_fw.text_frame
    tf_fw.word_wrap = True
    p = tf_fw.paragraphs[0]
    p.text = "Recommendations for Future Research"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    p = tf_fw.add_paragraph()
    p.text = "1. Continuous-Time Memory GNNs:\n   Evaluate Temporal Graph Networks (TGNs; Rossi et al., 2020) with dynamic memory states on streaming event graphs.\n\n2. Financial Heterophily Mitigation:\n   Incorporate edge-pruning mechanisms (CARE-GNN; Dou et al., 2020) and ego-residual projections to prevent neighborhood dilution.\n\n3. Provable XAI Fidelity Benchmarks:\n   Establish synthetic ground-truth attribution benchmarks (Amara et al., 2022) with matched integer deletion budgets.\n\n4. Leakage-Free Wallet Reconstruction:\n   Reconstruct wallet features strictly from historical ledger transactions."
    p.font.size = Pt(10.5)
    p.font.color.rgb = TEXT_DARK

    # Right: References
    add_card(s18, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_ref = s18.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_ref = tb_ref.text_frame
    tf_ref.word_wrap = True
    p = tf_ref.paragraphs[0]
    p.text = "Key Verified References"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = HEADER_BG
    p.space_after = Pt(8)

    p = tf_ref.add_paragraph()
    p.text = "• Alarab & Prakoonwit (2023). Neural Processing Letters, 55, 689–707.\n• Alarab et al. (2020). ICMLT 2020, 23–27.\n• Amara et al. (2022). PMLR, 198, 44:1–44:23.\n• Breiman (2001). Machine Learning, 45, 5–32.\n• Chen & Guestrin (2016). ACM SIGKDD 2016, 785–794.\n• Dou et al. (2020). ACM CIKM 2020, 315–324.\n• Elmougy & Liu (2023). ACM SIGKDD 2023.\n• Hamilton et al. (2017). NeurIPS 2017, 30.\n• Kipf & Welling (2017). ICLR 2017.\n• Lo et al. (2022). arXiv:2203.10465.\n• Vaswani et al. (2017). NeurIPS 2017, 30.\n• Veličković et al. (2018). ICLR 2018.\n• Weber et al. (2019). arXiv:1908.02591.\n• Ying et al. (2019). NeurIPS 2019, 32."
    p.font.size = Pt(9.5)
    p.font.color.rgb = TEXT_DARK

    # Save
    out_path = "ETGNN_NIBM_FINAL_VIVA_PRESENTATION.pptx"
    prs.save(out_path)
    print(f"Successfully generated 18-slide presentation at: {out_path}")

if __name__ == "__main__":
    build_presentation()
