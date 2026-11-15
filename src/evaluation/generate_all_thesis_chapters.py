"""
Complete Thesis Document Generator for ETGNN
Builds Chapter 1 to 6 Word documents, the compiled complete thesis draft,
and academic verification markdown files.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from thesis_generator_core import (
    create_styled_document, add_h1, add_h2, add_h3, add_p,
    add_callout, add_equation, add_image_if_exists, add_table_data,
    OUTPUT_DIR
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# CHAPTER 1: INTRODUCTION
# ==============================================================================
def build_chapter_1(doc=None, is_standalone=True):
    if doc is None:
        doc = create_styled_document()
    
    if is_standalone:
        add_callout(doc, 
            "OFFICIAL PROPOSED TITLE: Explainable Temporal Graph Neural Networks for Adaptive Financial Fraud Detection\n"
            "SUPERVISOR RECOMMENDATION: The implemented architecture is a static time-encoded Graph Attention Network (GAT) without "
            "runtime parameter adaptation, continuous-time event memory, or dynamic rewiring. We formally recommend revising "
            "the thesis title to: 'Explainable Time-Encoded Graph Attention Networks for Bitcoin Illicit Transaction Classification' "
            "or explicitly bounding 'Adaptive' as prospective/aspirational framing in Section 1.6.",
            title="SUPERVISOR TITLE GUIDANCE")
    
    add_h1(doc, "Chapter 1: Introduction")
    
    add_h2(doc, "1.1 Background and Context")
    add_p(doc, 
        "The rapid emergence of decentralized cryptographic ledger technologies over the past decade has transformed the global financial "
        "ecosystem. Bitcoin, introduced by Satoshi Nakamoto in 2008, established a peer-to-peer electronic cash system operating without "
        "intermediary financial institutions. While blockchain technology offers unprecedented settlement transparency and censorship resistance, "
        "its pseudonymous architecture—where transactions are publicly recorded under alphanumeric public keys rather than verified real-world "
        "identities—has concurrently created fertile ground for illicit activities. Financial crimes, including ransomware extortions, darknet "
        "marketplace vendor payments, terrorist financing, high-yield investment fraud, and money laundering syndicates, exploit this pseudonymity "
        "to obfuscate transactional audit trails across decentralized networks.")
    add_p(doc, 
        "To mitigate systemic systemic risk and comply with stringent international Anti-Money Laundering (AML) and Counter-Terrorist Financing (CTF) "
        "regulations (such as the Financial Action Task Force recommendations), regulatory agencies and virtual asset service providers (VASPs) "
        "require automated, high-precision forensic analytics systems. Historically, compliance screening relied on rule-based heuristics and static "
        "address blacklists. However, modern money laundering operations employ sophisticated evasion strategies, including mixing services, "
        "peeling chains, and multi-hop transactional layering, rendering signature-based detection largely ineffective against novel threat typologies.")

    add_h2(doc, "1.2 The Financial Fraud Detection Challenge in Transaction Graphs")
    add_p(doc, 
        "Financial transaction networks are intrinsically relational and dynamic. Bitcoin transactions naturally form directed payment-flow graphs, "
        "where vertices represent transaction events and directed edges represent the movement of funds from input unspent transaction outputs (UTXOs) "
        "to recipient addresses. Detecting illicit transactions within such networks presents three fundamental technical hurdles:")
    add_p(doc, 
        "1. Extreme Class Imbalance: Illicit transactions constitute an extreme minority of overall network traffic. In real-world forensic datasets, "
        "illicit entities frequently represent less than two to five percent of all transactions. Standard machine learning objectives optimized for "
        "overall accuracy produce deceptive classifiers that trivially predict the licit majority class while entirely failing to identify financial crimes.")
    add_p(doc, 
        "2. Topological and Structural Dependencies: Transactions do not exist as independent and identically distributed (i.i.d.) observations. "
        "A transaction's illicit status is intimately linked to its structural position within the payment graph, including the structural properties "
        "of its sender and recipient neighborhoods. Standard tabular models evaluate transactions in isolation unless supplied with hand-engineered "
        "topological aggregations.")
    add_p(doc, 
        "3. Temporal Concept Drift and Network Evolution: Illicit techniques, cryptocurrency valuations, regulatory enforcement actions, and darknet "
        "market seizures evolve dynamically over time. Graph representations that ignore temporal boundaries risk temporal data leakage, while models "
        "that fail to capture time may overfit to transient historical patterns that rapidly decay in forward-looking deployments.")

    add_h2(doc, "1.3 Problem Statement")
    add_p(doc, 
        "Recent advances in geometric deep learning have established Graph Neural Networks (GNNs)—including Graph Convolutional Networks (GCN), "
        "GraphSAGE, and Graph Attention Networks (GAT)—as powerful tools for structural anomaly detection. However, their application to forensic "
        "cryptocurrency classification encounters severe methodological challenges. First, transaction datasets such as the benchmark Elliptic Bitcoin "
        "dataset do not provide continuous cross-time-step edges; instead, all recorded payment flows exist exclusively within discrete, completed "
        "time snapshots. Standard GNNs operating purely on static graph topology cannot exploit the chronological ordering of transactions without "
        "explicit temporal representations.")
    add_p(doc, 
        "Second, deep graph architectures operate as opaque 'black boxes.' In highly regulated financial compliance and criminal forensics, "
        "a binary classification score alone is legally and operationally insufficient. Human compliance officers require faithful, verifiable "
        "explanations justifying why a particular transaction was flagged as suspicious. While post-hoc explainers such as GNNExplainer generate "
        "subgraph importance masks, conventional evaluations rely on anecdotal case studies or synthetic motifs without rigorous, controlled "
        "interventional benchmarking. Most critically, previous literature has largely ignored how explanation sensitivity differs between correct "
        "classifications and systematic model errors (False Positives and False Negatives).")

    add_h2(doc, "1.4 Research Questions")
    add_p(doc, 
        "To rigorously address these challenges without overstating architectural capabilities, this undergraduate thesis investigates four "
        "formal, evidence-bounded research questions:")
    add_p(doc, 
        "• Research Question 1 (RQ1 - Baseline Classification): How does a Graph Attention Network augmented with fixed sinusoidal time-step "
        "encodings (ETGNN) perform relative to established tabular baselines (Random Forest, XGBoost) and standard graph neural networks "
        "(GCN, GraphSAGE, static GAT) on held-out future Elliptic time steps when evaluated using illicit-class Area Under the Precision-Recall "
        "Curve (AUPRC)?")
    add_p(doc, 
        "• Research Question 2 (RQ2 - Temporal Encoding Contribution): On validation data, does the addition of a fixed 16-dimensional sinusoidal "
        "time encoding yield a statistically discernible improvement in AUPRC over a parameter-matched zero-time GAT control across multiple random "
        "initialization seeds?")
    add_p(doc, 
        "• Research Question 3 (RQ3 - Shared Interventional Contrast C1): When subjected to identical edge deletions selected by GNNExplainer on "
        "ETGNN under a common nominal budget, do ETGNN and its matched zero-time GAT comparator exhibit a differential prediction-probability response "
        "after calibrating against random edge deletion baselines?")
    add_p(doc, 
        "• Research Question 4 (RQ4 - Matched Error-vs-Correct Contrast C2): After strictly matching transaction pairs on prediction margin and "
        "computation-neighborhood candidate edge counts, does the explanation-minus-random deletion advantage differ systematically between erroneous "
        "predictions (False Positives, False Negatives) and correct predictions (True Positives, True Negatives) across primary and degree-balanced registers?")

    add_h2(doc, "1.5 Research Aim and Objectives")
    add_p(doc, 
        "The overarching aim of this research is to design, implement, and rigorously audit an explainable, time-encoded graph attention framework "
        "for illicit transaction classification on the Elliptic Bitcoin network, establishing defensible empirical bounds on both its predictive "
        "accuracy and the sensitivity of its post-hoc graph explanations.")
    add_p(doc, "To achieve this aim, the following specific technical objectives were executed:")
    add_p(doc, 
        "1. Construct a leak-free chronological data engineering pipeline on the Elliptic dataset, fitting feature scalers strictly on training time "
        "steps (1–32) and structuring non-overlapping validation (33–40) and held-out testing (41–49) graph snapshots.\n"
        "2. Implement and evaluate competitive tabular baselines (Random Forest, XGBoost) and deep graph architectures (GCN, GraphSAGE, static GAT) "
        "alongside a parameter-matched zero-time GAT comparator.\n"
        "3. Design the ETGNN architecture by coupling a deterministic 16-dimensional sinusoidal temporal projection to node feature representations "
        "prior to multi-head graph attention message passing.\n"
        "4. Conduct comprehensive ablation experiments isolating the predictive contributions of temporal encodings, local versus graph-aggregated "
        "features, and graph topological edges.\n"
        "5. Formulate and execute Contrast 1 (C1): a frozen interventional audit across 240 validation transactions evaluating random-adjusted "
        "response differences under shared edge deletions.\n"
        "6. Formulate and execute Contrast 2 (C2): a multi-register matched-pair cluster bootstrap audit across 420 pairs assessing explanation-minus-random "
        "advantages between correct and incorrect classifications under primary caliper and exact out-degree sensitivity matching.\n"
        "7. Deploy an interactive demonstration system integrating a FastAPI prediction and explanation backend with a Streamlit compliance dashboard.")

    add_h2(doc, "1.6 Scope and Delimitations")
    add_p(doc, 
        "To ensure absolute scientific integrity, the scope of this research is explicitly delimited as follows:")
    add_p(doc, 
        "• Static Time Encoding vs. Continuous-Time Memory: The implemented ETGNN is a static Graph Attention Network supplied with fixed sinusoidal "
        "time-step features. It does NOT implement recurrent state tracking, continuous-time event memory (such as TGN or JODIE), or dynamic edge "
        "prediction across time steps. Furthermore, because all Elliptic edges exist strictly within the same time step, the model processes discrete "
        "snapshots.\n"
        "• Absence of Online Parameter Adaptation: The research does NOT implement online continual learning, streaming drift adaptation, or runtime "
        "graph rewiring. The term 'Adaptive' in the approved project title represents the broad motivating compliance domain rather than an implemented "
        "algorithmic mechanism.\n"
        "• Model Retraining and Interventions: Edge-deletion perturbations evaluate model sensitivity under structural omission. The model weights "
        "remain strictly frozen; no retraining or online weight adjustment is performed during explanation audits.\n"
        "• Transductive Snapshot Setting: The experimental pipeline assumes that transaction features and within-step edges are observed at the end "
        "of each discrete time step. It does not evaluate sub-second screening at immediate mempool arrival.")

    add_h2(doc, "1.7 Structure of the Thesis")
    add_p(doc, 
        "The remainder of this thesis is structured as follows:\n"
        "• Chapter 2 (Literature Review) synthesizes foundational literature in blockchain AML, tabular machine learning, graph representation learning, "
        "temporal GNN paradigms, and post-hoc explanation evaluation, culminating in a structured research gap.\n"
        "• Chapter 3 (Research Methodology) details the dataset characteristics, chronological partition design, model mathematical formulations, "
        "training protocols, and the formal statistical specifications of Contrasts C1 and C2.\n"
        "• Chapter 4 (Classification Results and Analysis) presents the empirical classification findings on held-out test data, documents the "
        "superiority of tabular tree baselines, and analyzes topological and temporal feature ablations.\n"
        "• Chapter 5 (Explainability Results and Discussion) details the findings of the C1 and C2 benchmarks, examines deletion budget discreteness, "
        "presents the interactive demonstration architecture, and establishes complete epistemic limitations.\n"
        "• Chapter 6 (Conclusion and Future Work) synthesizes direct answers to the four research questions, summarizes verified contributions, "
        "and outlines concrete directions for future dynamic fraud detection research.")

    if is_standalone:
        doc.save(os.path.join(OUTPUT_DIR, "01_Introduction.docx"))
        print("[SUCCESS] Chapter 1: 01_Introduction.docx generated.")
    return doc

# ==============================================================================
# CHAPTER 2: LITERATURE REVIEW
# ==============================================================================
def build_chapter_2(doc=None, is_standalone=True):
    if doc is None:
        doc = create_styled_document()
        
    add_h1(doc, "Chapter 2: Literature Review")
    
    add_h2(doc, "2.1 Anti-Money Laundering and Forensics in Blockchain Networks")
    add_p(doc, 
        "Cryptocurrency ecosystems, underpinned by public distributed ledgers such as Bitcoin, maintain transparent, append-only records of all historical "
        "transactions. However, this transparency is decoupled from real-world legal identities, creating a pseudonymity barrier that enables criminal "
        "actors to execute illicit transfers with perceived impunity. Early forensic tracking methodologies relied on deterministic clustering heuristics "
        "(e.g., common-input ownership heuristics and change address detection algorithms) to aggregate individual addresses into entity-level clusters. "
        "While foundational, these rule-based heuristics suffer severe limitations when confronted with privacy-enhancing overlays, including CoinJoin "
        "mixers, decentralized cross-chain bridges, and peeling chains where small licit amounts are shaved off during continuous multi-hop laundering chains.")
    add_p(doc, 
        "The release of the Elliptic Bitcoin dataset by Weber et al. (2019) marked a seminal milestone in cryptocurrency AML research. For the first time, "
        "a large-scale, real-world forensic dataset comprising 203,769 transactions and 234,355 payment-flow edges was made publicly available to the academic "
        "community. The authors demonstrated that supervised machine learning could successfully discriminate between illicit entities (associated with "
        "ransomware, black-market bazaars, and sanctioned actors) and licit entities (such as regulated exchanges, miners, and commercial payment processors). "
        "Significantly, their initial benchmarking demonstrated that standard Random Forest classifiers outperformed basic Graph Convolutional Networks, "
        "establishing an important baseline precedent that underscores the necessity of rigorous tabular comparisons in financial graph analytics.")

    add_h2(doc, "2.2 Tabular Machine Learning Baselines")
    add_p(doc, 
        "Tree-based ensemble architectures remain the gold standard in tabular data modeling. Breiman (2001) formalized Random Forests, which construct "
        "uncorrelated ensembles of decision trees via bootstrap aggregation (bagging) and randomized feature subspace selection. Random Forests exhibit "
        "exceptional resilience to hyperparameter misspecification, handle mixed feature scales naturally, and are robust against extreme outlier values "
        "frequently observed in financial transaction amounts. Chen and Guestrin (2016) introduced XGBoost, an optimized end-to-end gradient tree boosting "
        "framework utilizing second-order Taylor expansion approximations of the loss function, column subsampling, and exact greedy tree-pruning algorithms. "
        "In imbalanced fraud detection settings, XGBoost allows direct tuning of scale-position weights to penalize false negatives on the minority class.")
    add_p(doc, 
        "A critical, frequently overlooked methodological nuance in the Elliptic literature concerns feature provenance. The 166 features published by "
        "Weber et al. (2019) comprise 94 local transaction attributes (including the discrete time step) and 72 precomputed one-hop neighborhood aggregations "
        "(capturing the minimum, maximum, mean, and standard deviation of transacting neighbors' features). Consequently, when tabular models such as "
        "Random Forest and XGBoost are trained on the full 165 numeric feature matrix, they already receive extensive graph-derived topological information. "
        "They cannot be legitimately characterized as 'graph-information-free' baselines—an empirical reality that directly informs our ablation design.")

    add_h2(doc, "2.3 Graph Representation Learning")
    add_p(doc, 
        "Graph Neural Networks (GNNs) extend deep learning primitives to non-Euclidean graph domains by implementing recursive neighborhood aggregation "
        "protocols, often formalized under the Message Passing Neural Network (MPNN) framework. Kipf and Welling (2017) introduced Graph Convolutional "
        "Networks (GCN), which approximate spectral graph convolutions via a localized, first-order Chebyshev polynomial expansion. GCN computes normalized "
        "symmetric adjacency aggregations: H^(l+1) = sigma(D~^(-1/2) A~ D~^(-1/2) H^(l) W^(l)), effectively propagating feature representations across "
        "immediate network neighbors.")
    add_p(doc, 
        "Hamilton, Ying, and Leskovec (2017) recognized the computational bottleneck of full-graph Laplacian operations in expanding networks and proposed "
        "GraphSAGE. GraphSAGE operates inductively by sampling uniform localized neighborhoods and learning parameterizable aggregation operators (such "
        "as mean, pooling, or LSTM aggregators), enabling inductive generalization to previously unseen nodes. Veličković et al. (2018) addressed the rigid, "
        "isotropic weighting of standard GCNs by introducing Graph Attention Networks (GAT). GAT incorporates self-attention mechanisms that assign learned, "
        "anisotropic attention coefficients alpha_ij to neighboring nodes based on paired feature projections. This attention mechanism allows the network "
        "to dynamically prioritize suspicious or informative transaction flows over high-volume benign payment conduits.")

    add_h2(doc, "2.4 Temporal Graph Learning Paradigms")
    add_p(doc, 
        "Financial networks evolve continuously, necessitating models capable of processing temporal dynamics. In the broader machine learning literature, "
        "temporal graph learning spans two primary structural paradigms:")
    add_p(doc, 
        "1. Discrete Dynamic Graph Snapshots: Methods such as EvolveGCN (Pareja et al., 2020) model temporal graphs as a sequence of discrete static "
        "graph snapshots, using recurrent neural network units (LSTM or GRU) to evolve the weight matrices of GCN layers over successive time periods. "
        "This captures macroeconomic weight evolution but assumes structural continuity across snapshots.")
    add_p(doc, 
        "2. Continuous-Time Dynamic Event Streams: Architectures such as Temporal Graph Networks (TGN; Rossi et al., 2020) and JODIE model networks at "
        "the granularity of individual timestamped events. TGN maintains continuous node memory states that update dynamically whenever a node participates "
        "in a transaction, coupled with temporal embedding modules that process time intervals via continuous Fourier or sinusoidal embeddings.")
    add_p(doc, 
        "Crucially, our inspection of the Elliptic dataset demonstrates that all 234,355 directed edges connect transaction nodes occurring strictly "
        "within the identical discrete time step (ranging from 1 to 49). There are zero cross-time-step edges. Consequently, the implemented ETGNN "
        "architecture does NOT employ continuous-time event memory or recurrent parameter evolution. Instead, it operates within completed time-step "
        "snapshots, augmenting the node feature representation with a deterministic sinusoidal time encoding vector prior to graph attention message passing.")

    add_h2(doc, "2.5 Explainability in Graph Neural Networks and Evaluation Benchmarks")
    add_p(doc, 
        "As deep graph models achieve prominence in high-stakes domains, Explainable AI (XAI) has emerged as an imperative requirement. Ying et al. (2019) "
        "introduced GNNExplainer, a model-agnostic, post-hoc optimization framework that identifies compact computation subgraphs and feature subsets "
        "most responsible for a target node prediction. GNNExplainer formulates a mutual information maximization objective, optimizing continuous edge "
        "masks M in [0, 1] through entropy regularization: min_M -sum_c 1_(y^=c) log P(Y = c | G_sub * sigma(M), X_sub). While widely adopted, standard "
        "GNNExplainer implementations produce continuous attribution masks that require subjective thresholding for visual inspection.")
    add_p(doc, 
        "A foundational challenge in XAI is the absence of ground-truth explanation labels. In real-world financial fraud, investigators possess transaction "
        "labels (illicit vs. licit) but lack verified annotations detailing which specific payment edges constituted the illegal conspiracy. To evaluate "
        "explanations objectively without ground truth, Hooker et al. (2019) established the Remove and Retrain (ROAR) framework in computer vision, proving "
        "that uncalibrated feature deletion often induces non-specific model degradation unrelated to true explanation fidelity. In the graph domain, "
        "subsequent benchmarking frameworks such as GraphFramEx (Amara et al., 2022) formalized deletion fidelity, evaluating whether omitting top-ranked "
        "explanation edges causes steeper predictive degradation than omitting unranked or random edges.")

    add_h2(doc, "2.6 Comparative Literature Matrix")
    add_p(doc, "Table 2.1 provides a structured synthesis of foundational studies directly informing this research:")
    
    headers_lit = ["Study", "Domain / Method", "Core Verified Contribution", "Relevance to ETGNN Implementation"]
    rows_lit = [
        ["Weber et al. (2019)", "Bitcoin AML / Elliptic Dataset", "Released Elliptic benchmark (203k nodes, 234k edges, 49 steps); established Random Forest baseline outperforming GCN.", "Defines our primary evaluation dataset and establishes strong tree baseline precedent. Proves features contain 72 neighborhood aggregates."],
        ["Breiman (2001)", "Ensemble Learning / Random Forest", "Formalized bootstrap aggregation with randomized feature subspaces for variance reduction.", "Forms our primary tabular benchmark; evaluated using balanced class weighting across 100 trees."],
        ["Chen & Guestrin (2016)", "Gradient Boosting / XGBoost", "Scalable tree boosting with second-order gradient optimization and sparsity-aware split finding.", "Forms our secondary boosted tabular baseline with scale_pos_weight optimization."],
        ["Kipf & Welling (2017)", "Geometric DL / GCN", "Localized first-order spectral graph convolutions over node neighborhoods.", "Direct foundation for our homogeneous graph convolution baseline on chronological snapshots."],
        ["Hamilton et al. (2017)", "Inductive DL / GraphSAGE", "Inductive neighborhood sampling and aggregation for large-scale graphs.", "Direct foundation for our inductive neighborhood aggregation baseline using mean pooling."],
        ["Veličković et al. (2018)", "Attention DL / GAT", "Masked self-attention mechanisms over graph neighborhoods.", "Direct architectural backbone for our standard GAT baseline, Matched GAT control, and ETGNN."],
        ["Ying et al. (2019)", "XAI / GNNExplainer", "Mutual information maximization over continuous edge and feature masks.", "Forms our core explanation generation algorithm across 2-hop computation subgraphs."],
        ["Hooker et al. (2019)", "XAI Benchmark / ROAR", "Demonstrated that explanation attribution must be evaluated against explicit random degradation baselines.", "Direct theoretical motivation for calibrating GNNExplainer deletion against 30 random edge deletion draws in C1 and C2."],
        ["Amara et al. (2022)", "Graph XAI / GraphFramEx", "Formalized evaluation frameworks (GraphFramEx) for systematic graph explanation benchmarking and deletion fidelity.", "Directly informs our structural edge-deletion protocol and fidelity advantage metrics."],
        ["Saito & Rehmsmeier (2015)", "Statistical Evaluation / PR Curves", "Demonstrated that Precision-Recall curves and AUPRC are substantially more informative than ROC-AUC in extreme class imbalance.", "Establishes illicit-class AUPRC (Average Precision) as the primary decision metric across all experiments."]
    ]
    add_table_data(doc, headers_lit, rows_lit, col_widths=[1.4, 1.4, 2.0, 1.7])

    add_h2(doc, "2.7 Critical Synthesis and Research Gap")
    add_p(doc, 
        "The synthesis of existing literature exposes an acute, unresolved methodological gap at the intersection of temporal graph learning and "
        "explainability evaluation. Prior graph explanation literature predominantly suffers from three structural constraints:")
    add_p(doc, 
        "1. Restriction to Synthetic Motifs or Static Graphs: The vast majority of XAI benchmarks evaluate explainers on synthetic graphs with planted "
        "motifs (such as BA-Shapes or Tree-Grid) or static citation networks. Real-world financial graphs—characterized by extreme class imbalance, "
        "anonymized numeric features, discrete temporal snapshots, and severe graph sparsity—are rarely subjected to rigorous explanation benchmarking.\n"
        "2. Absence of Parameter-Matched Shared Interventions (Contrast 1 Gap): Studies claiming that temporal graph models yield 'better explanations' "
        "almost universally evaluate each model against its own uncalibrated attributions. No existing study subjects a time-augmented GAT and a parameter-matched "
        "zero-time GAT control to the exact same shared structural interventions under a common target class while calibrating against random deletion baselines.\n"
        "3. Neglect of Systematic Error Contrasts (Contrast 2 Gap): Existing explanation benchmarks focus almost exclusively on explaining correct "
        "classifications (True Positives). In real-world financial compliance, false alarms (False Positives) consume massive institutional resources, "
        "while missed illicit operations (False Negatives) represent catastrophic compliance failures. Crucially, the literature lacks controlled, "
        "margin-matched and degree-balanced comparisons investigating whether explanation masks behave differently on model errors than on correct predictions.")
    add_p(doc, 
        "This thesis directly bridges these gaps through an evidence-bounded empirical investigation, implementing rigorous shared-intervention (C1) "
        "and matched-pair error-register (C2) audits on the Elliptic Bitcoin network.")

    if is_standalone:
        doc.save(os.path.join(OUTPUT_DIR, "02_Literature_Review.docx"))
        print("[SUCCESS] Chapter 2: 02_Literature_Review.docx generated.")
    return doc

# ==============================================================================
# CHAPTER 3: METHODOLOGY
# ==============================================================================
def build_chapter_3(doc=None, is_standalone=True):
    if doc is None:
        doc = create_styled_document()
        
    add_h1(doc, "Chapter 3: Research Methodology and Implementation")
    
    add_h2(doc, "3.1 Research Design Overview")
    add_p(doc, 
        "This research implements a dual-track empirical design engineered to provide rigorous, scientifically defensible answers to the stated "
        "research questions. Part A conducts a strict chronological benchmark evaluating predictive classification performance across tabular tree ensembles "
        "and deep graph architectures. Part B establishes a frozen, interventional explainability audit utilizing GNNExplainer, executing two controlled "
        "experiments: Contrast 1 (between-model shared edge deletions) and Contrast 2 (within-model matched error-versus-correct comparisons).")

    add_h2(doc, "3.2 Dataset Architecture and Class Imbalance")
    add_p(doc, 
        "All experiments are conducted on the official Elliptic Bitcoin Transaction Dataset. The dataset comprises 203,769 transaction vertices, "
        "234,355 directed edges representing Bitcoin payment flows, and spans 49 discrete time steps sampled at approximately two-week intervals.")
    add_p(doc, "The ground-truth label distribution exhibits severe class imbalance:")
    add_p(doc, 
        "• Class 1 (Illicit Transactions): 4,545 entities (~2.23% of total nodes). These represent verified malicious entities, including ransomware "
        "wallets, darknet market vendors, mixers, and terrorist financing conduits.\n"
        "• Class 2 (Licit Transactions): 42,019 entities (~20.62% of total nodes). These correspond to regulated exchanges, wallet providers, "
        "licensed miners, and legitimate merchant services.\n"
        "• Unknown (Unlabelled Transactions): 157,205 entities (~77.15% of total nodes). These represent unclassified background financial activity.")
    add_p(doc, 
        "For binary supervised classification, Class 1 is mapped to positive target label y = 1 and Class 2 is mapped to negative target label y = 0. "
        "Critically, unknown transactions (y = -1) participate fully in the graph structure—supplying node feature representations and propagating "
        "messages across directed edges during GNN convolution—but are strictly masked out of supervised cross-entropy loss functions and downstream "
        "evaluation metrics.")

    add_h2(doc, "3.3 Feature Decomposition and Provenance")
    add_p(doc, 
        "The published Elliptic feature matrix contains 167 raw columns without headers, comprising the transaction ID, the discrete time step index "
        "(t in [1, 49]), and 165 continuous numeric features. As established in Section 2.2, these 165 numeric features are cleanly partitioned into "
        "two distinct sub-matrices:")
    add_p(doc, 
        "1. Local Features (Columns 1–93): 93 features describing intrinsic transaction properties, including transaction fees, input/output counts, "
        "transacted BTC volume, and script execution metadata.\n"
        "2. Precomputed Neighborhood Aggregates (Columns 94–165): 72 features representing the minimum, maximum, mean, and standard deviation of "
        "local properties aggregated across one-hop transaction neighbors within the original Bitcoin graph.")
    add_p(doc, 
        "This structural decomposition has profound implications: full-feature tabular models (Random Forest and XGBoost) directly ingest 72 graph-derived "
        "features. To isolate the genuine contribution of runtime graph convolution from static tabular aggregates, our experimental protocol incorporates "
        "dedicated local-feature-only ablations.")

    add_h2(doc, "3.4 Chronological Splitting and Leakage Prevention")
    add_p(doc, 
        "In financial fraud detection, random k-fold cross-validation introduces severe temporal data leakage, as models evaluate past events using "
        "patterns learned from future transactions. To enforce temporal realism, the 49 time steps are partitioned chronologically:")
    add_p(doc, 
        "• Training Period: Time steps 1 to 32 (130,628 total transactions; 2,987 illicit, 26,734 licit). Used for parameter optimization.\n"
        "• Validation Period: Time steps 33 to 40 (Used for early stopping, hyperparameter checkpoint selection, and post-hoc explainability audits).\n"
        "• Held-Out Test Period: Time steps 41 to 49 (9,973 labelled transactions; 524 illicit [5.25%], 9,449 licit). Evaluated once under frozen model checkpoints.")
    add_p(doc, 
        "Data Leakage Safeguards: To strictly avoid future information leakage, the Scikit-learn StandardScaler was fitted exclusively on the 130,628 nodes "
        "of the training period (steps 1–32). The fitted mean and variance parameters were frozen and applied forward to transform validation and test features. "
        "Furthermore, stage-specific edge lists were constructed such that message-passing convolutions evaluated at stage T cannot observe edges from "
        "future stages T' > T.")
    add_callout(doc, 
        "DISCLOSURE OF VALIDATION-PERIOD REUSE: The downstream explainability benchmark (Contrasts C1 and C2) was conducted across 377 completed "
        "transactions from validation time steps 37–40 (following initial pipeline development on steps 33–36). Because these transactions reside within "
        "the validation period where model checkpoints and early-stopping criteria were selected, these experiments represent structured retrospective "
        "audits rather than untouched prospective trials. Furthermore, historical test split performance (steps 41–49) was inspected during preliminary "
        "research phases and is reported under frozen checkpoints without subsequent tuning.",
        title="METHODOLOGICAL DISCLOSURE")

    add_h2(doc, "3.5 Implemented Model Architectures")
    add_p(doc, "To provide a rigorous comparative foundation, seven distinct model architectures were implemented and frozen:")
    
    add_h3(doc, "3.5.1 Tabular Machine Learning Baselines")
    add_p(doc, 
        "• Random Forest (Scikit-learn): Configured with 100 decision trees, balanced class sub-sampling (class_weight='balanced'), Gini impurity split "
        "criteria, and unlimited maximum depth. Evaluated on full features (165), local features (93), and features augmented with time encoding.\n"
        "• XGBoost: Implemented using the exact greedy booster with scale_pos_weight set to the training class ratio (~8.95), maximum tree depth 6, "
        "learning rate 0.05, subsample ratio 0.8, and early stopping on validation AUPRC.")

    add_h3(doc, "3.5.2 Graph Neural Network Baselines")
    add_p(doc, 
        "• Graph Convolutional Network (GCN): 2-layer GCN architecture utilizing PyTorch Geometric GCNConv layers with hidden dimension 64, ReLU activation, "
        "and dropout 0.1.\n"
        "• GraphSAGE: 2-layer inductive architecture utilizing SAGEConv layers with mean neighborhood aggregation, hidden dimension 64, ReLU activation, "
        "and dropout 0.1.\n"
        "• Standard Graph Attention Network (GAT): 2-layer GAT architecture utilizing GATConv layers with 4 attention heads per layer (hidden dimension "
        "64 per head, concatenated to 256 in layer 1; 1 head in output layer), ELU non-linearity, and dropout 0.1.")

    add_h3(doc, "3.5.3 Parameter-Matched Zero-Time GAT Comparator")
    add_p(doc, 
        "To rigorously test whether temporal improvements stem from temporal information rather than expanded parameter capacity, a Parameter-Matched "
        "GAT was constructed. The model takes a 181-dimensional input, formed by concatenating the 165 node features with a 16-dimensional zero-vector "
        "[0, 0, ..., 0]. The weight matrices and layer dimensions are mathematically identical to ETGNN, ensuring that any performance disparity is "
        "attributable solely to the temporal encoding values.")

    add_h3(doc, "3.5.4 Proposed ETGNN (Time-Encoded GAT)")
    add_p(doc, 
        "The ETGNN architecture incorporates a deterministic 16-dimensional sinusoidal time-step projection. For a transaction node v occurring at discrete "
        "time step t in {1, ..., 49}, its temporal positional encoding PE(t) in R^16 is computed as:")
    add_equation(doc, "PE(t, 2i) = sin( t / (10000^(2i / 16)) ),    PE(t, 2i+1) = cos( t / (10000^(2i / 16)) )")
    add_p(doc, 
        "where i in {0, ..., 7}. The resulting temporal vector is concatenated with the standardized 165-dimensional node feature vector x_v, forming "
        "an augmented input representation h_v^(0) = [x_v || PE(t)] in R^181. This joint representation is processed by two GATConv layers operating "
        "over within-step edges with multi-head self-attention. The model produces unnormalized binary class logits, transformed via softmax into predicted "
        "class probabilities p(illicit) and p(licit).")

    add_h2(doc, "3.6 Training Protocols and Hyperparameter Management")
    add_p(doc, 
        "All GNN models were trained using PyTorch 2.6 and PyTorch Geometric 2.6 on a Windows workstation. Training utilized the Adam optimizer "
        "(learning rate 0.001, weight decay 1e-5) minimizing class-weighted binary cross-entropy loss, where positive illicit errors received a penalty "
        "weight proportional to the inverse training class frequency (~8.95). Training executed for up to 100 epochs with an early stopping patience "
        "of 15 epochs monitoring illicit-class AUPRC on validation time steps 33–40. A single optimal checkpoint per architecture was frozen for final testing.")

    add_h2(doc, "3.7 Evaluation Metrics Formulation")
    add_p(doc, 
        "In forensic classification with severe class imbalance (5.25% illicit prevalence in test data), traditional accuracy is fundamentally misleading. "
        "Accordingly, the primary evaluation metric is the Area Under the Precision-Recall Curve (AUPRC), computed as Average Precision (AP):")
    add_equation(doc, "AUPRC = sum_n (R_n - R_(n-1)) * P_n")
    add_p(doc, 
        "Secondary metrics include Area Under the Receiver Operating Characteristic (ROC-AUC), Precision, Recall, and F1-score evaluated at the canonical "
        "decision threshold tau = 0.50:")
    add_equation(doc, "y^ = 1  if  p(illicit) > 0.50,    y^ = 0  if  p(illicit) <= 0.50")

    add_h2(doc, "3.8 Post-Hoc Explainability Protocol (GNNExplainer)")
    add_p(doc, 
        "Explainability was evaluated using GNNExplainer on extracted directed two-hop computation subgraphs. For each evaluated transaction node v, "
        "GNNExplainer optimizes continuous edge masks M in [0, 1] over 50 epochs using Adam (learning rate 0.01) across frozen model checkpoints under "
        "three fixed random seeds: [42, 43, 44].")
    add_p(doc, 
        "Runtime Subgraph Equivalence Assertion: During execution, runtime assertions enforced that two-hop computation subgraph forward passes reproduced "
        "full-graph prediction probabilities within a strict tolerance of 1e-5 across all 377 completed nodes for both ETGNN and Matched GAT. An independent "
        "stratified check across candidate neighborhood sizes (E_cand in {2..7}) observed maximum discrepancies bounded by 3.58e-7 (ETGNN) and 8.94e-8 (Matched GAT).")

    add_h3(doc, "3.8.1 Discrete Deletion Budget Rule and Sparsity Inflation")
    add_p(doc, 
        "To evaluate structural explanation fidelity, the top-k edges prioritized by GNNExplainer were deleted from the computation subgraph. "
        "The frozen protocol enforced the discrete ceiling rule:")
    add_equation(doc, "k = max( 1, ceil( 0.25 * E_cand ) )")
    add_p(doc, 
        "where E_cand denotes the count of eligible physical candidate edges in the defined two-hop computation neighborhood (excluding self-loops). "
        "Because financial transaction neighborhoods in Elliptic are highly sparse, this discrete ceiling rule creates substantial deletion fraction inflation:")
    add_p(doc, 
        "• E_cand = 2 (N = 317 transactions; 84.1% of evaluated sample): k = max(1, ceil(0.50)) = 1 edge deleted. The actual deletion fraction is 1 / 2 = 50.0% "
        "(a 2.0x inflation over the nominal 25% target).\n"
        "• E_cand = 3 (N = 26; 6.9%): k = 1 (actual deletion fraction 33.3%).\n"
        "• E_cand = 4 (N = 18; 4.8%): k = 1 (actual deletion fraction 25.0%).\n"
        "• E_cand = 5 (N = 10; 2.7%): k = 2 (actual deletion fraction 40.0%).\n"
        "• E_cand = 6 (N = 2; 0.5%): k = 2 (actual deletion fraction 33.3%).\n"
        "• E_cand = 7 (N = 4; 1.1%): k = 2 (actual deletion fraction 28.6%).\n"
        "Across all 377 completed transactions, the mean actual deletion fraction was 47.08% (median 50.0%). Deletion perturbations therefore represent "
        "discrete topological amputations rather than marginal sensitivity gradients.")

    add_h2(doc, "3.9 Contrast 1 (C1) Experimental Specification")
    add_p(doc, 
        "Contrast 1 (C1) evaluates whether ETGNN and Matched GAT exhibit differential prediction drops when subjected to identical structural edge deletions. "
        "The experiment evaluated N = 240 transactions from validation time steps 37–40 under the following protocol controls:\n"
        "1. Shared Deletion Sets: Both models are subjected to the identical deletion of top-k edges identified by GNNExplainer on ETGNN.\n"
        "2. Common Target Class: Deletion drops for both models are evaluated on the identical target class (ETGNN's predicted class c = y^_ETGNN).\n"
        "3. Random Deletion Calibration: For each model M in {ETGNN, Matched_GAT}, the raw drop D_GE,M is calibrated against the mean drop across 30 seeded "
        "random edge draws under identical budget k, establishing the within-model advantage:")
    add_equation(doc, "Adv_M(v) = D_(GE, M)(v) - (1 / 30) * sum_(d=1)^30 D_(Rnd, M)^(d)(v)")
    add_p(doc, "The primary C1 estimator is the random-adjusted difference:")
    add_equation(doc, "Delta_adj(v) = Adv_ETGNN(v) - Adv_Matched_GAT(v)")
    add_p(doc, "with unadjusted raw drop difference Delta_raw(v) = D_(GE, ETGNN)(v) - D_(GE, Matched_GAT)(v) serving as a secondary diagnostic.")

    add_h2(doc, "3.10 Contrast 2 (C2) Matched-Pair Specification")
    add_p(doc, 
        "Contrast 2 (C2) assesses whether GNNExplainer edge masks yield larger deletion responses on erroneous classifications than on correct "
        "classifications within each model. Unlike C1, each model in C2 is evaluated using its own GNNExplainer edge ranking and its own predicted target class.")
    add_p(doc, 
        "Pairing and Caliper Constraints: Candidate pairs were matched within the same time step using Hopcroft-Karp maximum bipartite matching under "
        "strict eligibility calipers:\n"
        "• Prediction Margin: For any node v, margin(v) = |p(illicit, v) - 0.50|. The matching eligibility condition is:")
    add_equation(doc, "| margin(incorrect) - margin(correct) | <= 0.050")
    add_p(doc, "• Candidate Edge Caliper: The protocol-configured candidate-edge caliper is:")
    add_equation(doc, "| E_cand(incorrect) - E_cand(correct) | <= 5")
    add_p(doc, 
        "Two matching specifications were established across 8 registers (N = 420 total pair records):\n"
        "1. Primary Caliper Matching: Enforces margin caliper <= 0.050 and candidate-edge caliper <= 5.\n"
        "2. Degree-Balanced Sensitivity Matching: Enforces exact out-degree matching (out_degree_treat == out_degree_ctrl for 100% of pairs) under the identical calipers.")
    add_p(doc, "The paired difference contrast is defined as:")
    add_equation(doc, "Delta_pair = Adv(incorrect) - Adv(correct)")
    add_p(doc, 
        "Specifically, for FP–TP pairs: Delta_pair = Adv(FP) - Adv(TP); for FN–TN pairs: Delta_pair = Adv(FN) - Adv(TN). Uncertainty was estimated via "
        "matched-pair cluster bootstrap (B = 2,000 iterations, seed 20261020), resampling pairs as clusters to account for within-pair correlation.")

    if is_standalone:
        doc.save(os.path.join(OUTPUT_DIR, "03_Methodology.docx"))
        print("[SUCCESS] Chapter 3: 03_Methodology.docx generated.")
    return doc

# ==============================================================================
# CHAPTER 4: CLASSIFICATION RESULTS
# ==============================================================================
def build_chapter_4(doc=None, is_standalone=True):
    if doc is None:
        doc = create_styled_document()
        
    add_h1(doc, "Chapter 4: Classification Results and Analysis")
    
    add_h2(doc, "4.1 Held-Out Test Evaluation Performance")
    add_p(doc, 
        "The primary empirical evaluation of classification performance was conducted on the held-out test split (time steps 41–49). The test set "
        "comprises 9,973 labelled Bitcoin transactions, including 524 illicit transactions (5.25% positive prevalence). Models were evaluated at the "
        "canonical decision threshold tau = 0.50 under frozen checkpoints. Table 4.1 summarizes the complete performance metrics across all models.")
    
    headers_test = ["Model Architecture", "Input Feature Specification", "Precision", "Recall", "F1-Score", "ROC-AUC", "Test AUPRC"]
    rows_test = [
        ["Random Forest (Full)", "Full 165 features (93 local + 72 agg)", "0.6966", "0.5782", "0.6319", "0.8735", "0.6323"],
        ["Random Forest (Local Only)", "Local 93 numeric features only", "0.5635", "0.5840", "0.5736", "0.8515", "0.6301"],
        ["Random Forest (+ Time)", "165 features + 16-d sinusoidal time", "0.6594", "0.5802", "0.6173", "0.8672", "0.6295"],
        ["XGBoost", "Full 165 features (boosted trees)", "0.2823", "0.6050", "0.3849", "0.8074", "0.5472"],
        ["ETGNN (No-Edge Control)", "93 local + 16-d time, NO EDGES", "0.2445", "0.6164", "0.3501", "0.8107", "0.4812"],
        ["GraphSAGE", "165 features + 2-layer mean aggregation", "0.1238", "0.7481", "0.2125", "0.8205", "0.2427"],
        ["ETGNN (Local Only)", "93 local + 16-d time + edges", "0.2225", "0.5286", "0.3132", "0.7894", "0.2344"],
        ["ETGNN (Full Proposed)", "165 features + 16-d time + edges", "0.2614", "0.4504", "0.3308", "0.8062", "0.2250"],
        ["Matched GAT Control", "165 features + 16 zero-pad + edges", "0.1205", "0.7424", "0.2074", "0.8175", "0.2099"],
        ["GCN", "165 features + 2-layer convolution", "0.1009", "0.8645", "0.1808", "0.7732", "0.1306"],
        ["Standard GAT", "165 features + 2-layer attention", "0.0893", "0.9141", "0.1627", "0.7548", "0.1155"]
    ]
    add_table_data(doc, headers_test, rows_test, col_widths=[1.5, 1.8, 0.7, 0.7, 0.7, 0.7, 0.8])

    add_h2(doc, "4.2 Analysis of Hypothesis H1: Tabular vs. Graph Performance Gap")
    add_p(doc, 
        "The empirical findings deliver a definitive, evidence-bounded answer to Research Question 1 and Hypothesis H1: Hypothesis H1 (that ETGNN would "
        "outperform the strongest baseline) is NOT supported. Tabular Random Forest achieved an AUPRC of 0.6323, dramatically outperforming all deep graph "
        "architectures. The proposed ETGNN achieved an AUPRC of 0.2250, trailing Random Forest by 0.4073 AUPRC points. XGBoost also surpassed all GNNs, "
        "achieving an AUPRC of 0.5472.")
    add_p(doc, 
        "Detailed inspection of the precision-recall trade-offs explains this disparity. Standard GNNs achieved high recall (GAT reached 91.41% recall; "
        "GCN reached 86.45%) but suffered catastrophic precision collapse (GAT precision was 8.93%; GCN was 10.09%). In practical terms, to capture illicit "
        "nodes, standard GNNs generated massive false-alarm volumes, flagging over nine licit transactions for every genuine illicit transaction detected. "
        "Random Forest maintained a balanced precision of 69.66% and recall of 57.82%, yielding an F1-score of 0.6319.")

    add_h2(doc, "4.3 Topological Message-Passing Ablation")
    add_p(doc, 
        "To investigate why graph message passing underperformed, a controlled topological edge ablation was evaluated. When ETGNN was trained on "
        "the 93 local numeric features with full graph edges, it achieved a held-out test AUPRC of 0.2344. When evaluated with the exact same architecture, "
        "local features, and time encodings but with all graph edges removed (the 'No-Edge Control'), AUPRC surged to 0.4812 (+0.2468 increase; Table 4.1).")
    add_p(doc, 
        "This critical ablation establishes that within-step payment edges did not assist fraud detection; rather, message passing across these edges actively "
        "degraded classification precision. In transaction networks where money launderers mix licit and illicit inputs across high-degree aggregation nodes, "
        "unconstrained neighborhood convolution propagates noise, causing oversmoothing and obscuring localized transaction features.")
    
    add_image_if_exists(doc, "ablation_graph_vs_features.png", 
        "Figure 4.1: Feature and Edge Ablation Comparison. Demonstrates the performance jump when graph edges are removed in the local-feature ETGNN.", 
        width_in_inches=5.2)

    add_h2(doc, "4.4 Multi-Seed Temporal Encoding Validation Ablation")
    add_p(doc, 
        "To address Research Question 2 and Hypothesis H2, the contribution of sinusoidal time encoding was evaluated against the Parameter-Matched "
        "Zero-Time GAT control across three identical random seeds on validation time steps 33–40:")
    add_p(doc, 
        "• Seed 7: ETGNN AUPRC = 0.5096 vs. Matched GAT AUPRC = 0.5572 (Matched GAT higher by +0.0476)\n"
        "• Seed 42: ETGNN AUPRC = 0.5488 vs. Matched GAT AUPRC = 0.4836 (ETGNN higher by +0.0652)\n"
        "• Seed 123: ETGNN AUPRC = 0.5293 vs. Matched GAT AUPRC = 0.3934 (ETGNN higher by +0.1359)\n"
        "• Three-Seed Mean: ETGNN Mean = 0.5292 vs. Matched GAT Mean = 0.4781 (+0.0511 mean advantage for ETGNN).")
    add_p(doc, 
        "Statistical Interpretation: While ETGNN held a higher three-seed mean on validation data, Matched GAT scored higher under Seed 7. "
        "Furthermore, on the held-out test period, the difference between frozen checkpoints narrowed to 0.0151 (ETGNN 0.2250 vs. Matched GAT 0.2099). "
        "Consequently, Hypothesis H2 is only partially supported: sinusoidal time encoding provides indicative validation benefit in certain runs, "
        "but does not establish a statistically significant or generalizable temporal effect.")

    add_h2(doc, "4.5 Temporal Concept Drift and Illicit Prevalence Analysis")
    add_p(doc, 
        "Figure 4.2 plots model performance across individual test time steps (41 to 49) alongside the illicit class prevalence. The empirical evidence "
        "demonstrates severe temporal drift: at time steps 41 and 42, Random Forest achieved strong AUPRC (>0.75), but performance degraded sharply at "
        "time steps 45–48. Crucially, illicit prevalence fluctuates dramatically across the test period: time step 46 contains only two illicit transactions "
        "(0.28% prevalence), causing AUPRC scores to fluctuate due to finite-sample volatility rather than intrinsic model failure.")
    
    add_image_if_exists(doc, "test_auprc_by_time_step.png", 
        "Figure 4.2: Held-out Test AUPRC by Time Step alongside Illicit Prevalence Drift across Steps 41 to 49.", 
        width_in_inches=5.6)

    add_image_if_exists(doc, "pr_curves.png", 
        "Figure 4.3: Precision-Recall Curves across Models on Held-Out Test Time Steps (41–49).", 
        width_in_inches=5.4)

    if is_standalone:
        doc.save(os.path.join(OUTPUT_DIR, "04_Classification_Results.docx"))
        print("[SUCCESS] Chapter 4: 04_Classification_Results.docx generated.")
    return doc

# ==============================================================================
# CHAPTER 5: EXPLAINABILITY RESULTS
# ==============================================================================
def build_chapter_5(doc=None, is_standalone=True):
    if doc is None:
        doc = create_styled_document()
        
    add_h1(doc, "Chapter 5: Explainability Results and Discussion")
    
    add_h2(doc, "5.1 Computation Subgraph Sparsity and Budget Discreteness")
    add_p(doc, 
        "Before evaluating structural deletions, the topological structure of the 377 completed validation transactions (steps 37–40) was audited. "
        "As established in Section 3.8.1, the nominal deletion rule is k = max(1, ceil(0.25 * E_cand)). Analysis reveals an acute structural reality: "
        "317 out of 377 transactions (84.1%) possess exactly E_cand = 2 eligible candidate edges. For these transactions, deleting one edge (k = 1) "
        "imposes an actual deletion fraction of 50.0%—a two-fold inflation over the nominal 25% target.")
    add_p(doc, 
        "Figure 5.1 illustrates the candidate-edge distribution and deletion fraction inflation. Across all evaluated transactions, the mean actual "
        "deletion fraction was 47.08% (median 50.0%). Researchers and compliance officers must recognize that GNNExplainer edge removals on Elliptic "
        "do not evaluate infinitesimal marginal attributions; they evaluate macroscopic topological amputations on ultra-sparse computation subgraphs.")
    
    add_image_if_exists(doc, "candidate_edges_deletion_budget.png", 
        "Figure 5.1: Distribution of Eligible Candidate Edges and Actual Deletion Fraction Inflation under the Ceiling Budget Rule.", 
        width_in_inches=5.6)

    add_h2(doc, "5.2 Contrast 1 (C1): Shared-Intervention Between-Model Audit")
    add_p(doc, 
        "Contrast 1 evaluates whether ETGNN and Matched GAT respond differently when subjected to the identical deletion of top edges selected by "
        "GNNExplainer on ETGNN across N = 240 validation transactions (steps 37–40). Table 5.1 summarizes the C1 empirical response statistics.")
    
    headers_c1 = ["Cohort Register", "N", "Mean Delta_adj", "Sample SD (s)", "Median Delta_adj", "IQR", "Mean Delta_raw", "ETGNN Adv", "GAT Adv"]
    rows_c1 = [
        ["Overall C1 Cohort", "240", "+0.002903", "+/-0.096180", "-0.001534", "0.045904", "+0.038491", "0.020239", "0.017336"],
        ["True Positives (TP)", "95", "-0.005973", "+/-0.132491", "-0.001400", "0.044846", "+0.040412", "-0.003310", "0.002663"],
        ["False Positives (FP)", "95", "+0.011367", "+/-0.051097", "-0.000205", "0.041682", "+0.040085", "0.036034", "0.024667"],
        ["True Negatives (TN)", "25", "+0.013444", "+/-0.085633", "0.000000", "0.052547", "+0.044734", "0.063984", "0.050539"],
        ["False Negatives (FN)", "25", "-0.006073", "+/-0.070797", "-0.017843", "0.065701", "+0.018893", "0.005957", "0.012030"]
    ]
    add_table_data(doc, headers_c1, rows_c1, col_widths=[1.5, 0.5, 1.0, 1.0, 0.9, 0.7, 0.9, 0.7, 0.7])

    add_p(doc, 
        "Empirical Findings and Interpretation for RQ3:\n"
        "1. Magnitude of Adjusted Difference: Across all 240 transactions, the observed mean random-adjusted difference was very small: "
        "mean Delta_adj = +0.002903. The sample standard deviation across transactions is s = +/-0.096180 (reflecting node-level variance, not standard error). "
        "The median difference was -0.001534 with an IQR of 0.045904.\n"
        "2. Advantage Over Random Comparator: Both models exhibited positive mean advantage over 30 random deletion draws (0.0202 for ETGNN; 0.0173 for Matched GAT), "
        "confirming that GNNExplainer edge rankings induce slightly larger probability drops than unguided random edge removals.\n"
        "3. Epistemic Limitation: A small observed mean difference (+0.0029) does NOT prove that ETGNN and Matched GAT are functionally equivalent, nor does "
        "it prove that sinusoidal time encoding has zero impact. It indicates that under frozen checkpoints, the two models exhibit broadly comparable sensitivity "
        "to shared edge omissions.")
    
    add_image_if_exists(doc, "c1_response_distributions.png", 
        "Figure 5.2: Contrast 1 Response Distributions across 240 Validation Transactions showing Adjusted (Delta_adj) and Raw (Delta_raw) Differences.", 
        width_in_inches=5.6)

    add_h2(doc, "5.3 Contrast 2 (C2): Matched-Pair Error-Register Analysis")
    add_p(doc, 
        "Contrast 2 tests whether GNNExplainer edge masks yield larger deletion responses on erroneous classifications than on correct classifications "
        "within each model. Table 5.2 presents the complete results across all 8 registers, evaluated via matched-pair cluster bootstrap (B = 2,000 iterations).")
    
    headers_c2 = ["Register Name", "Model", "Match Specification", "Pairs", "Mean Delta_pair", "Bootstrap SE", "95% Marginal CI", "Zero Crossed?"]
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
    add_table_data(doc, headers_c2, rows_c2, col_widths=[1.5, 0.9, 1.1, 0.5, 1.0, 0.9, 1.3, 0.6])

    add_p(doc, 
        "Key Scientific Findings for RQ4:\n"
        "1. Direction and Persistence of ETGNN FP–TP Difference: Within the matched sample, the explanation-minus-random deletion response was larger "
        "for false positives than true positives (mean paired difference +0.039343 in primary matching; +0.043455 in degree-sensitivity matching). Neither "
        "95% bootstrap confidence interval crosses zero. This demonstrates that removing top-ranked explanation edges reduced predicted-class probabilities "
        "more sharply for false alarms than for true illicit classifications. Crucially, this empirical response pattern must NOT be described as proof of "
        "explanation faithfulness, identification of harmful edges, or automated error correction.\n"
        "2. Matching Specification Sensitivity in Matched GAT: In Matched GAT, primary caliper matching yielded a positive estimate (+0.020315, 95% CI "
        "[+0.003917, +0.038684]), but under exact out-degree matching the estimate attenuated to +0.010547 and its 95% CI crossed zero ([-0.013240, +0.032770]). "
        "Because primary and degree-sensitivity registers represent overlapping matched cohorts, this shift demonstrates that GAT explanation sensitivity is "
        "vulnerable to neighborhood degree matching specifications.\n"
        "3. Pervasive Imprecision in False Negative (FN) Registers: In all four TN–FN registers, the 95% bootstrap confidence intervals are exceptionally wide "
        "and span zero. This imprecision stems directly from finite-sample limitations: only 25 false negatives were available for ETGNN and 21 for Matched GAT. "
        "The inclusion of zero must NOT be interpreted as evidence of functional equality between TN and FN explanations; it reflects statistical imprecision.")

    add_image_if_exists(doc, "c2_paired_differences_ci.png", 
        "Figure 5.3: Contrast 2 Forest Plot of Matched-Pair Differences (Delta_pair) and 95% Marginal Bootstrap Confidence Intervals across All 8 Registers.", 
        width_in_inches=5.8)

    add_image_if_exists(doc, "c2_primary_vs_degree_sensitivity.png", 
        "Figure 5.4: Comparative Analysis of Primary Caliper vs. Degree-Balanced Sensitivity Matching across Error Registers.", 
        width_in_inches=5.6)

    add_h2(doc, "5.4 Qualitative Subgraph Visualizations")
    add_p(doc, 
        "Figures 5.5 and 5.6 present representative two-hop computation subgraphs extracted for individual True Positive (TP) and False Positive (FP) "
        "transactions. Visual inspection confirms that while GNNExplainer highlights high-weight payment conduits, the continuous masks require subjective "
        "thresholding, reinforcing the necessity of our quantitative edge-deletion benchmark over anecdotal visualization alone.")
    
    add_image_if_exists(doc, "explanation_tp_subgraph.png", 
        "Figure 5.5: Computation Subgraph and GNNExplainer Edge Mask for Representative True Positive Transaction (ID: 288717023).", 
        width_in_inches=4.8)
    add_image_if_exists(doc, "explanation_fp_subgraph.png", 
        "Figure 5.6: Computation Subgraph and GNNExplainer Edge Mask for Representative False Positive Transaction (ID: 116898498).", 
        width_in_inches=4.8)

    add_h2(doc, "5.5 Interactive Demonstration System")
    add_p(doc, 
        "To satisfy the applied engineering objective, a decoupled demonstration system was developed in src/demo/:\n"
        "• FastAPI Backend (src/demo/backend.py): Exposes high-performance REST endpoints for health checks (/health), precomputed explanation examples "
        "(/examples), and live transaction inference (/transactions/{id}). Predictions are served using cached in-memory PyTorch graph snapshots with "
        "strict input validation (returning HTTP 404 for unknown transaction IDs).\n"
        "• Streamlit Compliance Dashboard (src/demo/frontend.py): Provides an interactive compliance interface rendering risk gauges, model comparison "
        "dials (Random Forest vs. ETGNN probability), and interactive PyVis topological subgraphs highlighting high-risk payment flows.\n"
        "• End-to-End Verification: An automated integration test (src/evaluation/test_demo_http_server.py) verifies server startup, route integrity, "
        "and error handling with 100% pass rate.")

    add_h2(doc, "5.6 Threats to Validity and Epistemic Boundaries")
    add_p(doc, "The interpretation of these results is subject to four critical epistemic constraints:")
    add_p(doc, 
        "1. Retrospective Validation Reuse: As disclosed, C1 and C2 evaluate validation time steps 37–40, where model early stopping occurred. "
        "These are structured retrospective audits, not prospective clinical trials.\n"
        "2. Confounding of Static Precomputed Features: Deleting graph edges perturbs runtime message passing during GNN forward passes but leaves "
        "the 72 precomputed tabular neighborhood aggregates static in the feature matrix.\n"
        "3. Network Autocorrelation: While matched-pair bootstrap resamples pairs as independent clusters, transactions reside in a connected financial "
        "graph where computation subgraphs partially overlap. Bootstrap intervals do not fully eliminate spatial autocorrelation.\n"
        "4. Non-Causal Nature of Explanations: Deletion perturbations index empirical model sensitivity under edge omission; they do NOT establish "
        "causal mechanisms of financial fraud, legal culpability, or automated error repair.")

    if is_standalone:
        doc.save(os.path.join(OUTPUT_DIR, "05_Explainability_Results.docx"))
        print("[SUCCESS] Chapter 5: 05_Explainability_Results.docx generated.")
    return doc

# ==============================================================================
# CHAPTER 6: CONCLUSION
# ==============================================================================
def build_chapter_6(doc=None, is_standalone=True):
    if doc is None:
        doc = create_styled_document()
        
    add_h1(doc, "Chapter 6: Conclusion and Future Work")
    
    add_h2(doc, "6.1 Synthesis and Answers to Research Questions")
    add_p(doc, 
        "This thesis conducted a rigorous, evidence-bounded empirical investigation into time-encoded Graph Attention Networks and post-hoc explanation "
        "sensitivity on the Elliptic Bitcoin transaction network. The four core research questions are answered definitively as follows:")
    add_p(doc, 
        "• Answer to Research Question 1 (Comparative Classification): Tabular Random Forest substantially outperformed all Graph Neural Networks on "
        "held-out future time steps (41–49), achieving an illicit-class AUPRC of 0.6323 compared to ETGNN's 0.2250, XGBoost's 0.5472, GraphSAGE's 0.2427, "
        "and standard GAT's 0.1155. Hypothesis H1 is decisively NOT supported. In extreme class imbalance with precomputed neighborhood aggregates, "
        "tree bagging resists distribution drift far more effectively than deep graph message passing.\n"
        "• Answer to Research Question 2 (Temporal Encoding Effect): Adding fixed 16-dimensional sinusoidal time encoding yielded an indicative improvement "
        "on validation data across a three-seed mean (0.5292 vs. 0.4781 for Matched GAT), but Matched GAT outperformed ETGNN under Seed 7 (0.5572 vs. 0.5096) "
        "and held-out test differences were small (0.2250 vs. 0.2099). Hypothesis H2 is partially supported on validation data but does not establish a "
        "statistically generalizable temporal effect.\n"
        "• Answer to Research Question 3 (C1 Shared Interventions): Under shared GNNExplainer edge deletions across 240 validation transactions, the observed "
        "mean random-adjusted difference between ETGNN and Matched GAT was very small (Delta_adj = +0.002903, s = +/-0.096180). Both models exhibited "
        "modest positive advantages over random edge removals (~0.020 vs. 0.017), indicating comparable structural sensitivity.\n"
        "• Answer to Research Question 4 (C2 Matched Error Contrasts): Within the matched sample, GNNExplainer edge deletions induced a statistically "
        "persistent, larger probability drop for false positives than true positives in ETGNN across both primary (Delta_pair = +0.039343, 95% CI "
        "[+0.020234, +0.057410]) and degree-sensitivity (+0.043455, 95% CI [+0.019882, +0.069552]) registers. In contrast, Matched GAT estimates were "
        "sensitive to out-degree matching (its 95% CI crossing zero), while all False Negative registers exhibited pervasive statistical imprecision due "
        "to limited sample sizes.")

    add_h2(doc, "6.2 Summary of Verified Contributions")
    add_p(doc, "This research delivers four verified academic contributions to the fields of financial forensics and geometric deep learning:")
    add_p(doc, 
        "1. Leak-Free Chronological Benchmarking: A fully reproducible data engineering and evaluation pipeline on Elliptic enforcing strict temporal "
        "scaling and non-overlapping masks, proving that standard GNNs underperform tabular baselines when precomputed topological aggregates are available.\n"
        "2. Topological Message-Passing Damage Isolation: Empirical demonstration via local-feature edge ablations that message passing across within-step "
        "edges degraded AUPRC from 0.4812 (no-edge control) down to 0.2344 (with edges), establishing oversmoothing and noise propagation in dense financial hubs.\n"
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

    if is_standalone:
        doc.save(os.path.join(OUTPUT_DIR, "06_Conclusion.docx"))
        print("[SUCCESS] Chapter 6: 06_Conclusion.docx generated.")
    return doc

# ==============================================================================
# COMPILED MASTER THESIS DRAFT
# ==============================================================================
def build_complete_thesis():
    doc = create_styled_document()
    
    # Title Page
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(36)
    p_title.paragraph_format.space_after = Pt(12)
    r_t = p_title.add_run("EXPLAINABLE TEMPORAL GRAPH NEURAL NETWORKS FOR ADAPTIVE FINANCIAL FRAUD DETECTION")
    r_t.font.name = 'Times New Roman'
    r_t.font.size = Pt(22)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(24)
    r_sub = p_sub.add_run("A Controlled Chronological and Interventional Explainability Audit on the Elliptic Bitcoin Transaction Network")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(14)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(48)
    p_meta.paragraph_format.space_after = Pt(48)
    r_meta = p_meta.add_run(
        "A Final Year Research Thesis Submitted in Partial Fulfillment of the\n"
        "Requirements for the Degree of Bachelor of Science (Honours) in Computer Science / Artificial Intelligence\n\n"
        "Department of Computer Science & Software Engineering\n"
        "Faculty of Computing / Information Technology\n"
        "Academic Year: 2026\n"
    )
    r_meta.font.name = 'Times New Roman'
    r_meta.font.size = Pt(11)
    r_meta.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    add_callout(doc, 
        "SUPERVISOR FORMAL TITLE RECOMMENDATION: The implemented architecture is a static time-encoded Graph Attention Network (GAT) "
        "without runtime parameter adaptation, continuous-time event memory, or dynamic rewiring. We formally recommend revising "
        "the thesis title to: 'Explainable Time-Encoded Graph Attention Networks for Bitcoin Illicit Transaction Classification' "
        "or explicitly bounding 'Adaptive' as prospective/aspirational framing in Section 1.6.",
        title="SUPERVISOR / EXAMINER TITLE NOTE")
    
    doc.add_page_break()
    
    # Abstract
    add_h1(doc, "Abstract")
    add_p(doc, 
        "The proliferation of decentralized cryptographic ledger technologies has exacerbated the challenge of detecting illicit financial flows, "
        "including money laundering, ransomware extortions, and darknet marketplace financing. Graph Neural Networks (GNNs) offer a natural paradigm "
        "for modeling transaction topologies, yet their practical deployment in compliance is hindered by extreme class imbalance, temporal distribution "
        "shifts, and the opacity of deep graph representations. This undergraduate thesis investigates an Explainable Time-Encoded Graph Attention Network "
        "(ETGNN) for illicit Bitcoin transaction classification on the benchmark Elliptic dataset (203,769 transactions, 234,355 directed edges across 49 time steps). "
        "The research executes two connected tracks: (1) a leak-free chronological classification evaluation against competitive tabular tree ensembles "
        "(Random Forest, XGBoost) and standard deep graph baselines (GCN, GraphSAGE, static GAT), and (2) a frozen post-hoc explainability benchmark "
        "utilizing GNNExplainer across 377 completed validation transactions (steps 37–40) evaluating random-adjusted deletion responses under shared "
        "interventions (Contrast 1) and matched correct-versus-incorrect prediction pairs (Contrast 2).")
    add_p(doc, 
        "Contrary to prevalent architectural assumptions, tabular Random Forest decisively achieved the highest held-out test performance (time steps 41–49) "
        "with an illicit-class Area Under the Precision-Recall Curve (AUPRC) of 0.6323, substantially surpassing XGBoost (0.5472), GraphSAGE (0.2427), "
        "and the proposed ETGNN (0.2250). A controlled topological edge ablation revealed that evaluating ETGNN without graph edges surged local-feature AUPRC "
        "from 0.2344 to 0.4812 (+0.2468), proving that message passing across within-step payment edges introduced structural noise and oversmoothing in dense "
        "transaction hubs. In post-hoc explainability audits, Contrast 1 demonstrated a small observed mean random-adjusted difference between ETGNN and a "
        "parameter-matched zero-time GAT control (+0.002903 across 240 transactions). In Contrast 2, within-model matched-pair cluster bootstrapping (2,000 iterations) "
        "demonstrated that GNNExplainer edge removals induced a larger probability drop on false positives than true positives for ETGNN across both primary "
        "(+0.039343, 95% CI [+0.020234, +0.057410]) and exact out-degree sensitivity (+0.043455, 95% CI [+0.019882, +0.069552]) registers. However, because 84.1% "
        "of evaluated transactions possess only two eligible computation edges, discrete ceiling budget rules imposed an actual 50.0% structural deletion fraction. "
        "These findings demonstrate that perturbation responses index local model sensitivity to topological omission rather than causal explanations of fraud, "
        "establishing evidence-bounded empirical constraints for accountable financial forensics.")
    add_p(doc, 
        "Keywords: Graph Neural Networks, Financial Fraud Detection, Bitcoin Forensic Analytics, Elliptic Dataset, Graph Attention Networks, "
        "GNNExplainer, Post-Hoc Explainability, Chronological Evaluation, Perturbation Auditing, Model Interpretability.")
    
    doc.add_page_break()
    
    # Table of Contents Outline
    add_h1(doc, "Table of Contents")
    toc_items = [
        ("Chapter 1: Introduction", "1.1 Background, 1.2 Challenges, 1.3 Problem Statement, 1.4 Research Questions, 1.5 Objectives, 1.6 Scope & Delimitations, 1.7 Organization"),
        ("Chapter 2: Literature Review", "2.1 Blockchain AML, 2.2 Tabular ML, 2.3 Graph Learning, 2.4 Temporal GNNs vs Time-Encoding, 2.5 Explainability & ROAR/GraphFramEx, 2.6 Literature Matrix, 2.7 Research Gap"),
        ("Chapter 3: Methodology", "3.1 Research Design, 3.2 Elliptic Dataset, 3.3 Feature Decomposition, 3.4 Chronological Split & Leakage Controls, 3.5 Models, 3.6 Training, 3.7 Metrics, 3.8 GNNExplainer & Budget Rule, 3.9 Contrast 1 Protocol, 3.10 Contrast 2 Protocol"),
        ("Chapter 4: Classification Results", "4.1 Test Split Performance, 4.2 H1 Tabular vs Graph Analysis, 4.3 Message-Passing Harm Ablation, 4.4 Multi-Seed Temporal Ablation, 4.5 Temporal Concept Drift"),
        ("Chapter 5: Explainability Results", "5.1 Subgraph Sparsity & Budget Discreteness, 5.2 Contrast 1 Results, 5.3 Contrast 2 Bootstrap Results, 5.4 Visualizations, 5.5 Interactive Demo, 5.6 Epistemic Boundaries"),
        ("Chapter 6: Conclusion & Future Work", "6.1 Research Question Answers, 6.2 Verified Contributions, 6.3 Limitations, 6.4 Future Work, 6.5 Concluding Remarks"),
        ("References", "Primary Academic Literature and Verified Official Records"),
        ("Appendix", "A: Mathematical Formulations, B: Reproducibility Commands, C: Checkpoint & Code Artifact Hashes")
    ]
    for ch, subs in toc_items:
        p_toc = doc.add_paragraph()
        p_toc.paragraph_format.space_after = Pt(2)
        r_ch = p_toc.add_run(ch)
        r_ch.bold = True
        r_ch.font.name = 'Times New Roman'
        r_ch.font.size = Pt(11)
        p_sub_toc = doc.add_paragraph(f"    {subs}")
        p_sub_toc.paragraph_format.space_after = Pt(4)
        for r in p_sub_toc.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            
    doc.add_page_break()
    
    # Chapters 1 to 6
    build_chapter_1(doc, is_standalone=False)
    doc.add_page_break()
    build_chapter_2(doc, is_standalone=False)
    doc.add_page_break()
    build_chapter_3(doc, is_standalone=False)
    doc.add_page_break()
    build_chapter_4(doc, is_standalone=False)
    doc.add_page_break()
    build_chapter_5(doc, is_standalone=False)
    doc.add_page_break()
    build_chapter_6(doc, is_standalone=False)
    doc.add_page_break()
    
    # References
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
    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.5)
        p_ref.paragraph_format.first_line_indent = Inches(-0.5)
        p_ref.paragraph_format.space_after = Pt(6)
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        
    doc.add_page_break()
    
    # Appendix
    add_h1(doc, "Appendix: Mathematical Proofs & Artifact Verification")
    add_h2(doc, "Appendix A: Mathematical Formulations of C1 and C2")
    add_p(doc, 
        "1. Contrast 1 Advantage Difference:\n"
        "   Adv_M(v) = D_(GE, M)(v) - (1/30) * sum_(d=1)^30 D_(Rnd, M)^(d)(v)\n"
        "   Delta_adj(v) = Adv_ETGNN(v) - Adv_Matched_GAT(v)\n"
        "   Observed mean: Delta_adj = +0.002903, transaction sample standard deviation s = +/-0.096180.\n\n"
        "2. Contrast 2 Paired Differences:\n"
        "   Delta_pair = Adv(incorrect) - Adv(correct)\n"
        "   FP–TP: Delta_pair = Adv(FP) - Adv(TP)\n"
        "   FN–TN: Delta_pair = Adv(FN) - Adv(TN)\n"
        "   Resampled via matched-pair cluster bootstrap (B = 2,000 iterations, seed 20261020).")
    
    add_h2(doc, "Appendix B: Cryptographic Checksums of Research Artifacts")
    add_p(doc, 
        "• ETGNN_Main_C1_C2_Results.zip: SHA256 = 550092c5f77bcb51702986006c48c8fec29ee98ff564f833ad17f780a43872fc\n"
        "• ETGNN_Main_C1_C2_Execution_Bundle.zip: SHA256 = 1a48608440285a21092e0df406e257eb897818be32a76f2f9f1b409ea6402e86\n"
        "• Frozen Main Benchmark Runner Snapshot: SHA256 = ef67f742fba5053b211a76f7596ff172551a87b1c3c990b7933994bb0a174c8c\n"
        "• Full Integration Verification Status: 100% Passed (14/14 automated assertions passed with 1e-10 precision).")

    master_path = os.path.join(OUTPUT_DIR, "ETGNN_Complete_Thesis_Draft.docx")
    doc.save(master_path)
    print(f"[SUCCESS] Complete Thesis Draft: {master_path} generated successfully.")

if __name__ == "__main__":
    print("=== Generating ETGNN Thesis Word Documents ===")
    build_chapter_1()
    build_chapter_2()
    build_chapter_3()
    build_chapter_4()
    build_chapter_5()
    build_chapter_6()
    build_complete_thesis()
    print("=== All Thesis Word Documents Generated Successfully ===")
