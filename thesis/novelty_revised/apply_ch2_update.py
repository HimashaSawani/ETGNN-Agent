"""
apply_ch2_update.py
Applies verified contemporary literature and Table 2.1 updates to Chapter 2 in
thesis/novelty_revised/build_novelty_revised_thesis.py.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_SCRIPT = ROOT / "thesis" / "novelty_revised" / "build_novelty_revised_thesis.py"

with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
    text = f.read()

new_build_chapter_2 = '''def build_chapter_2(doc):
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
        "AlphaBay darknet market. Subsequently, Elmougy and Liu (2023) extended the Elliptic transaction benchmark into the "
        "Elliptic++ transaction-and-actor forensic dataset (KDD \'23), demonstrating the continued dominance of tree-based ensembles "
        "while exposing how unguarded lifetime wallet summary statistics can introduce look-ahead leakage if not strictly filtered "
        "by transaction timestamp."
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
        "transactions\' local features). Therefore, any classifier trained on the full 165-feature matrix already ingests "
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
        "the ego-node\'s features compete inside the same softmax normalization against all incoming neighbors; in heterophilic "
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
        "(V_t intersect V_t\' = empty for t != t\') with zero cross-snapshot edges, persistent node-state recurrence is "
        "structurally inapplicable on the transaction-level Elliptic DAG.",
        bold_prefix="1. Persistent Node-State Recurrence (Inapplicable to Disjoint Transaction Sets): "
    )
    add_p(
        doc,
        "To handle dynamic graphs where node sets change or are completely disjoint across snapshots, Pareja, Domeniconi, "
        "Chen, Ma, Suzumura, Kanezashi, Kaler, Schardl, and Leiserson (2020) introduced EvolveGCN at AAAI 2020—and benchmarked "
        "it directly on the Elliptic Bitcoin dataset. Rather than evolving node states, EvolveGCN uses a recurrent neural "
        "network to evolve the fixed-dimension GCN weight matrices W_t^(l) in R^(d^(l) x d^(l+1)) across snapshots t = 1..49:\\n"
        "  • EvolveGCN-O evolves weight matrices via an LSTM conditioned purely on prior weight matrices: "
        "W_t^(l) = LSTM(W_(t-1)^(l)). This recurrence is completely independent of whether nodes persist across snapshots.\\n"
        "  • EvolveGCN-H evolves weight matrices via a GRU conditioned on a fixed-size Top-k pooling summary of the current "
        "snapshot\'s node embeddings: W_t^(l) = GRU(TopK(H_t^(l), k), W_(t-1)^(l)). Because Top-k pooling projects variable-sized "
        "node sets |V_t| into a fixed k x d^(l+1) matrix matching the column dimension of W_(t-1)^(l), EvolveGCN-H is also "
        "mathematically well-defined on disjoint node sets, although pooling across disjoint transaction cohorts captures "
        "macro-level snapshot feature distributions rather than persistent entity trajectories.\\n"
        "Crucially, in both Weber et al. (2019) and Pareja et al. (2020), EvolveGCN remained substantially below Random "
        "Forest on the Elliptic benchmark, demonstrating that evolving GCN weights across snapshots does not overcome the "
        "underlying dilution of local transaction features by GCN message passing. In contrast, our implemented ETGNN "
        "adopts the deterministic sinusoidal positional encoding of Vaswani et al. (2017) concatenated to GAT node inputs, "
        "allowing controlled isolation of static snapshot time encoding against a parameter-matched zero-time GAT baseline.",
        bold_prefix="2. Weight-Matrix Evolution Across Disjoint Snapshots (EvolveGCN-O and EvolveGCN-H; Pareja et al., 2020): "
    )

    add_h2(doc, "2.5 Contemporary Advancements in Dynamic Financial GNNs (2025–2026)")
    add_p(
        doc,
        "Between 2025 and 2026, research in cryptocurrency fraud detection expanded across three distinct trajectories: "
        "complex dynamic representation learning, adversarial stress testing, and benchmark skepticism."
    )
    add_p(
        doc,
        "Lin et al. (2026) introduced ChronoWave-GNN (Scientific Reports, DOI: 10.1038/s41598-025-23901-3), utilizing graph "
        "wavelets and Chebyshev spectral filters to capture multi-frequency temporal patterns across transaction streams. "
        "Concurrently, Han et al. (2026) proposed FG-EGCN (Scientific Reports, DOI: 10.1038/s41598-026-53783-y), integrating dynamic "
        "feature-gating mechanisms to suppress feature dilution during GCN aggregation. Chen et al. (2025) presented MDST-GNN "
        "(Advanced Intelligent Systems, DOI: 10.1002/aisy.202400898), which employs multiscale spatial-temporal multi-hop attention. "
        "While these models advance architectural sophistication, they treat dynamic GNNs as uninspected black boxes, reporting "
        "zero quantitative evaluation of explanation fidelity, feature attribution stability, or interventional validation.",
        bold_prefix="1. Dynamic Representation Models (Lin et al., 2026; Han et al., 2026; Chen et al., 2025): "
    )
    add_p(
        doc,
        "Critical evaluations have challenged the prevailing assumption that graph neural architectures are superior to tabular "
        "models in financial forensics. Maganti (2026, arXiv:2604.19514), in an empirical investigation titled \'When Graph Structure "
        "Becomes a Liability\', showed that on chronological splits of the Elliptic dataset, Random Forest achieves higher F1 scores "
        "than standard GCN and GAT. Maganti argued that recursive message passing dilutes high-signal local transaction features "
        "(e.g., fee rates, input/output counts) when aggregating across heterogeneous counter-parties. This finding is further "
        "substantiated by the extensive benchmark survey of Chakraborty et al. (2026, IEEE ICBC), which demonstrated that GNNs "
        "exhibit severe precision trade-offs in highly imbalanced cryptocurrency environments.",
        bold_prefix="2. Benchmark Skepticism and Tabular Superiority (Maganti, 2026; Chakraborty et al., 2026): "
    )
    add_p(
        doc,
        "Akter et al. (2025/2026, EAI Endorsed Transactions on AI and Robotics, DOI: 10.4108/airo.13658) conducted temporal-structural "
        "stress testing of GNNs across multi-granularity blockchain graphs, evaluating degradation under synthetic adversarial topological "
        "perturbations. However, their analysis did not examine macroeconomic regime shifts (such as darknet market closures) or "
        "assess whether post-hoc explanation masks can guide model recovery under distribution collapse.",
        bold_prefix="3. Adversarial Stress Testing (Akter et al., 2025/2026): "
    )

    add_h2(doc, "2.6 Explainable Graph Neural Networks and Faithfulness Benchmarks")
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

    add_h2(doc, "2.7 Precision–Recall Evaluation Under Severe Class Imbalance")
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

    add_h2(doc, "2.8 Comparative Literature Matrix and Research Gaps")
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
        ["Davis & Goadrich (2006) & Saito & Rehmsmeier (2015)", "Imbalanced Statistical Evaluation", "Proved PR superiority over ROC in imbalance and step-function AP superiority over trapezoidal PR-AUC.", "Establishes Average Precision (`average_precision_score`) as our primary standardized metric."],
        ["Lin et al. (2026)", "Dynamic GNN (ChronoWave-GNN)", "Graph wavelets and Chebyshev spectral filters for multi-frequency transaction patterns in Scientific Reports.", "Contemporary dynamic GNN benchmark; highlights the black-box gap with zero XAI evaluation."],
        ["Han et al. (2026)", "Dynamic Feature-Gated GCN (FG-EGCN)", "Dynamic feature-gating mechanism to suppress feature dilution during GCN aggregation in Scientific Reports.", "Addresses feature dilution inside GCN; motivates our ego-residual and neighborhood smearing analysis."],
        ["Chen et al. (2025)", "Multiscale Spatial-Temporal GNN (MDST-GNN)", "Multiscale spatial-temporal multi-hop attention for transaction fraud in Advanced Intelligent Systems.", "Contemporary complex architecture; provides comparative baseline for dynamic attention models."],
        ["Akter et al. (2025/2026)", "Graph Stress Testing", "Temporal-structural stress testing across multi-granularity blockchain graphs in EAI Trans. AI & Robotics.", "Validates structural vulnerability; our Exp A/B extends to real-world darknet regime collapse."],
        ["Chakraborty et al. (2026)", "GNN Blockchain Benchmark", "Comprehensive evaluation of GNNs in blockchain transactions at IEEE ICBC 2026.", "Corroborates severe precision-recall trade-offs and tabular competitiveness on cryptocurrency graphs."],
        ["Maganti (2026)", "Graph Liability Critique", "Empirical study in arXiv:2604.19514 showing graph structure becomes a liability on Elliptic under inductive splits.", "Theoretical corroboration of our finding that tabular tree models outperform message-passing GNNs."]
    ]
    add_tbl(doc, headers_lit, rows_lit, col_widths=[1.3, 1.3, 2.0, 1.9],
            caption="Table 2.1: Verified Comparative Literature Matrix and Methodological Positioning.", font_size_pt=6.8)

    add_p(
        doc,
        "As synthesized in Table 2.1, while contemporary 2025–2026 literature (Lin et al., 2026; Han et al., 2026; Chen et al., 2025) "
        "has focused heavily on introducing increasingly intricate dynamic GNN architectures, existing research leaves three decisive "
        "methodological voids:\\n"
        "1. The XAI Quantitative Blind Spot: Contemporary dynamic GNNs operate entirely as black boxes, providing zero quantitative "
        "interventional evaluation of post-hoc explanation fidelity or stability across error registers.\\n"
        "2. The Mechanism of Darknet Regime Collapse: While studies note performance drops post-step 43 (Weber et al., 2019; "
        "Akter et al., 2025), the underlying topological mechanism has not been characterized at the graph census level.\\n"
        "3. Empirical Bounds on XAI-Guided Mitigation: Prior work has not tested whether post-hoc explanation masks can guide "
        "inference-time model repair under domain shift.\\n"
        "Addressing these three critical gaps forms the primary empirical contribution of this thesis."
    )
'''

code_sub = re.sub(r'def build_chapter_2\(doc\):.*?(?=def build_chapter_3)', new_build_chapter_2 + '\n\n', text, flags=re.DOTALL)
if code_sub == text:
    print("[ERROR] build_chapter_2 regex replacement failed!")
    sys.exit(1)
text = code_sub
print("[SUCCESS] Chapter 2 updated.")

with open(TARGET_SCRIPT, "w", encoding="utf-8") as f:
    f.write(text)

