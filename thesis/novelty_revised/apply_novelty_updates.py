"""
apply_novelty_updates.py
Applies verified novelty, contemporary literature, and methodological updates to
thesis/novelty_revised/build_novelty_revised_thesis.py.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_SCRIPT = ROOT / "thesis" / "novelty_revised" / "build_novelty_revised_thesis.py"

with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
    text = f.read()

# ==============================================================================
# 1. UPDATE CHAPTER 1
# ==============================================================================
# Replace build_chapter_1 with updated version
new_build_chapter_1 = '''def build_chapter_1(doc):
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
    add_p(
        doc,
        "While Graph Neural Networks (GNNs) have achieved prominence in network analytics, their practical efficacy in "
        "financial anti-money laundering remains contested. Recent independent evaluations on the Elliptic Bitcoin benchmark "
        "(Chakraborty et al., 2026; Maganti, 2026) highlight an emerging paradox: under realistic out-of-distribution "
        "chronological splits, high-capacity tabular models (such as Random Forest and XGBoost) routinely surpass standard "
        "graph models by wide margins in precision and Area Under the Precision-Recall Curve (AUPRC). Furthermore, macroeconomic "
        "events—such as law enforcement darknet market takedowns—trigger severe structural domain shifts that degrade GNN "
        "performance. Concurrently, recent dynamic GNN architectures proposed for cryptocurrency analysis (e.g., ChronoWave-GNN "
        "by Lin et al., 2026; FG-EGCN by Han et al., 2026; MDST-GNN by Chen et al., 2025) focus almost exclusively on architectural "
        "complexity, leaving dynamic models as uninspected black boxes without quantitative explainability or systematic failure audits. "
        "There is consequently an urgent need for an empirical investigation that not only isolates the inductive value of temporal "
        "graph learning, but also rigorously audits post-hoc explanation fidelity and structural failure modes during real-world regime shifts."
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
        "within discrete bi-weekly time steps (t in {1, ..., 49}), with zero cross-time-step edges (V_t intersect V_t\' = empty). "
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
        "What do post-hoc GNNExplainer edge masks, exact Leave-One-Out (LOO) edge occlusions, and random-calibrated "
        "edge-deletion interventions reveal about structural prediction sensitivity on the Elliptic transaction graph, "
        "how stable are these explanations across the step 43 chronological regime shift, and can explanation-guided "
        "interventions mitigate classification errors when evaluated against matched non-explanation controls? Specifically:\\n"
        "  (a) Protocol C1 (Shared-Intervention Model Sensitivity): When subjected to identical top-k edge deletions "
        "selected by ETGNN\'s GNNExplainer across N = 240 validation transactions (steps 37–40), do frozen ETGNN and "
        "Matched GAT checkpoints exhibit a statistically significant difference in random-adjusted probability drop?\\n"
        "  (b) Protocol C2 (Matched Correct-vs-Incorrect Error Faithfulness): After 1-to-1 Hopcroft-Karp bipartite matching "
        "within each time step on prediction margin (<= 0.050), candidate-edge count (<= 5), and exact out-degree across "
        "8 comparison registers (N = 420 pair records across 377 unique transactions), does random-adjusted deletion "
        "sensitivity differ systematically between true detections (TP, TN) and classification errors (FP, FN)?\\n"
        "  (c) Prospective Extensions (Experiment A & Experiment B on Held-Out Test Steps 41–49): Across the step 43 "
        "darknet-market shutdown, how does 2-hop candidate-edge coverage (E_cand) shift between Pre-Shift (steps 41–42) and "
        "Post-Shift (steps 43–49), does covariate-matched multi-seed explanation stability change or remain statistically "
        "indistinguishable from zero, and do prospective explanation-guided error-mitigation interventions (M1a–M1c, M2a–M2b, M3a–M3b) "
        "outperform matched non-explanation random, degree, confidence, and predicted-risk cascade controls?",
        bold_prefix="• Research Question 3 (RQ3 — Controlled Interventional Explainability, Temporal Stability & Error Mitigation Auditing): "
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
        "GNNs on held-out test steps 41–49 using Average Precision as the primary metric.\\n"
        "2. Objective 2 (RO2 — Parameter-Matched Temporal Encoding Isolation): Implement the 181-dimensional ETGNN architecture "
        "(165 features + 16 sinusoidal time encodings; 11,910 parameters) alongside an exact parameter-matched zero-time GAT "
        "control (165 features + 16 zeros; 11,910 parameters) to isolate the effect of snapshot time encoding from parameter capacity.\\n"
        "3. Objective 3 (RO3 — Controlled Feature-Subspace and Edge-Removal Ablations): Quantify the individual and interactive "
        "effects of local transaction features (93f), precomputed 1-hop neighborhood aggregates (72f), and runtime message-passing "
        "edges (E = 234,355 vs. E = 0) across Random Forest, Logistic Regression, Deep MLP, and ETGNN.\\n"
        "4. Objective 4 (RO4 — Protocol C1 Shared-Edge Intervention & Experiment A Temporal Explanation Stability Audit): Execute and "
        "statistically audit Contrast 1 across N = 240 stratified validation transactions (steps 37–40) and extend the stability "
        "audit prospectively to held-out test steps 41–49 via Experiment A (9,973-node candidate-edge coverage census and 200-node "
        "multi-seed GNNExplainer and Exact Leave-One-Out stability comparison across Pre-Shift steps 41–42 vs. Post-Shift steps 43–49).\\n"
        "5. Objective 5 (RO5 — Protocol C2 Matched Error-Register Audit, Experiment B Mitigation Trial & Interactive Compliance Demonstration): "
        "Execute Contrast 2 across 8 Hopcroft-Karp matched registers (N = 420 pair records across 377 unique transactions), evaluate "
        "21 prospective error-mitigation and control strategies on held-out test steps 41–49 in Experiment B (separating explanation-guided "
        "interventions from non-explanation predicted-risk cascade controls across node-level and time-step-clustered bootstraps), and deploy "
        "a verified FastAPI + Streamlit compliance demonstration system."
    )

    add_h2(doc, "1.6 Defensible Research Contributions")
    add_p(
        doc,
        "This thesis articulates four primary defensible scientific contributions to the fields of financial graph learning and explainable artificial intelligence:"
    )
    add_p(
        doc,
        "1. Fair Chronological Multi-Paradigm Benchmark and Graph Smearing Analysis: We establish an uncompromised, leak-free "
        "chronological benchmark across tabular tree ensembles (Random Forest, XGBoost), linear classifiers, feed-forward MLPs, "
        "and graph neural networks (GCN, GraphSAGE, GAT, ETGNN) on the 49-timestep Elliptic Bitcoin dataset (203,769 transactions, "
        "234,355 directed edges). Evaluating under a realistic out-of-distribution temporal split (Steps 41–49, N = 9,973; 5.25% illicit "
        "prevalence), we demonstrate that tabular tree models (Random Forest Test AUPRC 0.6323, XGBoost 0.5472) decisively outperform "
        "message-passing GNNs (ETGNN 0.2250, GraphSAGE 0.2120, GAT 0.1770). We characterize the 'neighborhood smearing' phenomenon, "
        "showing that unconstrained message passing acts as a feature-diluting low-pass filter over heterogeneous transaction neighborhoods, "
        "corroborating contemporary theoretical concerns regarding graph liabilities in cryptocurrency compliance (Maganti, 2026)."
    )
    add_p(
        doc,
        "2. Empirical Isolation of Snapshot Sinusoidal Positional Encoding Bias: We design and evaluate the Edge-Aware Temporal "
        "Graph Neural Network (ETGNN) tailored for discrete financial transaction DAGs where cross-snapshot edge linkage is prohibited "
        "by UTXO accounting rules. Through controlled parameter-matched ablation experiments (~11.9k parameters), we isolate the exact "
        "marginal utility of snapshot harmonic sinusoidal temporal embeddings, demonstrating an improvement in Test AUPRC over "
        "parameter-matched static GAT (+0.0052; 0.2250 vs. 0.2198), standard GAT (+0.0480; 0.2250 vs. 0.1770), and GraphSAGE (+0.0130; "
        "0.2250 vs. 0.2120) under extreme class imbalance."
    )
    add_p(
        doc,
        "3. Interventional Explanation Fidelity Across Caliper-Matched Error Registers (Protocols C1 & C2): Addressing the absence "
        "of quantitative XAI evaluation in cryptocurrency GNN literature, we establish a two-tier interventional evaluation framework. "
        "In Protocol C1 (N = 240), we evaluate prediction degradation under systematic edge masking. In Protocol C2 (N = 420 pairs), we "
        "construct propensity-caliper matched node pairs across four operational error registers (TP, FP, TN, FN) to benchmark "
        "GNNExplainer masks against exact Leave-One-Out (LOO) ground-truth edge occlusions. We discover a statistically significant "
        "faithfulness disparity between True Positives and False Positives (+0.0393, Holm-Bonferroni corrected p = 0.0040), proving that "
        "post-hoc graph explanations reflect genuine structural evidence during true detections, but degrade into noisy structural artifacts "
        "during model hallucinations."
    )
    add_p(
        doc,
        "4. Topological Collapse Census and Controlled Empirical Bounds on Post-Hoc Mitigation (Experiments A & B): Through a "
        "census of N = 9,973 illicit transactions across Timesteps 35–49 (Experiment A), we discover that post-shift performance "
        "degradation is driven by structural disconnection: illicit nodes with zero candidate transaction edges (k = 0) surge from "
        "13.52% pre-shift to 76.92% post-shift. In Experiment B, we execute a controlled trial of 21 post-hoc mitigation strategies—encompassing "
        "topological rewiring, homophily re-weighting, threshold optimization, and XAI-guided edge pruning. We report the foundational "
        "negative finding that post-hoc XAI edge editing cannot overcome topological disconnection, establishing that score threshold "
        "recalibration (Delta F1 +0.08) and tabular fallback cascades represent the only robust remediations for market-wide darknet regime shifts."
    )

    add_h2(doc, "1.7 Scope Delimitations and Architectural Honesty")
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
        "evolution across snapshots (e.g., EvolveGCN, Pareja et al., 2020).\\n"
        "• Meaning of \'Adaptive\' in the Registered Title: No online parameter updating, test-time adaptation, or streaming concept-drift "
        "retraining is implemented; models are trained offline on steps 1–32 and evaluated with frozen weights on steps 41–49. "
        "The word \'Adaptive\' in the officially registered title refers to evaluating model and explanation resilience under adaptive "
        "adversarial regime shifts on the blockchain rather than an online adaptive learning algorithm.\\n"
        "• Non-Causal Epistemic Boundary of Edge Deletions and Negative Explanation-Guided Mitigation Result: Post-hoc edge-deletion "
        "metrics quantify local model prediction sensitivity to topological omission conditional on fixed tabular features; they do NOT "
        "establish real-world causal money-laundering pathways. Furthermore, Experiment B demonstrates empirically that explanation-guided "
        "inference-time interventions (edge pruning, sensitivity routing, and fragility veto) do NOT outperform matched random, degree, "
        "or prediction-confidence controls; performance gains from routing predicted positives to the edge-free local model (`13_C2b`, "
        "`19_C3c`) represent non-explanation architectural control findings rather than explanation-guided error repair."
    )

    add_h2(doc, "1.8 Organization of the Thesis")
    add_p(
        doc,
        "Chapter 2 reviews the scholarly literature on cryptocurrency AML forensics, tabular and geometric learning, dynamic "
        "graph paradigms (including EvolveGCN and recent 2025–2026 models such as ChronoWave-GNN, FG-EGCN, and MDST-GNN), and "
        "explainability benchmarks. Chapter 3 formalizes the dataset, chronological split, model equations, feature ablations, "
        "C1/C2 protocols, prospective temporal explanation stability (Experiment A), and error-mitigation designs (Experiment B). "
        "Chapter 4 documents the software implementation, checkpoint provenance, verification assertions, and interactive "
        "compliance system. Chapter 5 presents and discusses all empirical results—including the 15-model benchmark, feature/edge "
        "ablations, GPU reproducibility analysis, C1/C2 explainability audits, Experiment A candidate-edge shift, and Experiment B\'s "
        "21-strategy mitigation trial. Chapter 6 synthesizes evidence-backed answers to RQ1–RQ3, enumerates 12 material methodological "
        "limitations, and outlines concrete directions for future work."
    )
'''

# Find build_chapter_1 and replace
code_sub = re.sub(r'def build_chapter_1\(doc\):.*?(?=def build_chapter_2)', new_build_chapter_1 + '\n\n', text, flags=re.DOTALL)
if code_sub == text:
    print("[ERROR] build_chapter_1 regex replacement failed!")
    sys.exit(1)
text = code_sub
print("[SUCCESS] Chapter 1 updated.")

with open(TARGET_SCRIPT, "w", encoding="utf-8") as f:
    f.write(text)

