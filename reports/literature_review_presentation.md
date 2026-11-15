# Explainable Temporal Graph Neural Networks for Financial Fraud Detection (ETGNN)
## Undergraduate Research Literature Review & Experimental Methodology Presentation

**Researcher:** P.V.H.S. Vitharama ([STUDENT-ID-REDACTED])  
**Supervisor:** Prof. Y.P.R.D. Yapa  
**Institution:** Kandy Innovation Centre, National Institute of Business Management (NIBM)  

---

### Slide 1: Research Context & Graph Formulation
* **Domain:** Illicit transaction detection in cryptocurrency financial networks (ACFE estimates ~5% global revenue loss to occupational and transactional financial fraud).
* **Graph Structure in Elliptic:**
  * **Nodes:** Represent individual Bitcoin transactions ($N = 203,769$).
  * **Edges:** Represent directed payment flows between transactions ($E = 234,355$).
  * *Important Distinction:* Accounts/wallets are not explicit nodes in this benchmark; nodes represent single transactional events. All edges connect transactions within the same time step (no cross-time-step edges exist in the raw dataset).
* **Feature Representation:**
  * **166 features total**, including the discrete time step.
  * **165 predictor features** when time step is separated (comprising 93 local transaction attributes and 72 one-hop neighborhood aggregations).

---

### Slide 2: Traditional Machine Learning Baselines & Relational Boundaries
* **Models Evaluated:** Random Forest (RF) and XGBoost.
* **Mechanism & Relational Information:**
  * Tabular classifiers treat rows independently and **do not perform explicit graph message passing**.
  * However, they **can utilize relational information through engineered features**, notably Elliptic's 72 precomputed neighborhood summary statistics (e.g., minimum, maximum, standard deviation of neighbor transaction amounts).
* **Class Imbalance & Thresholding:**
  * Minority illicit class (~9.8% in labelled data; ~2.2% in total data).
  * Imbalance addressed via class-weighted loss and threshold tuning strictly on the validation period.

---

### Slide 3: Static Graph Neural Networks (GCN, GraphSAGE, GAT)
* **Message Passing Paradigms:**
  * **GCN (Kipf & Welling, 2017):** Full neighborhood isotropic spectral-style aggregation.
  * **GraphSAGE (Hamilton et al., 2017):** Inductive neighborhood sampling and feature aggregation.
  * **GAT (Veličković et al., 2018):** Parametric self-attention assigning dynamic edge coefficients across transaction neighbors.
* **Role in Research Progression:**
  * Serves as the static relational baseline directly testing whether dynamic message passing on raw topology provides predictive gain over tabular classifiers.
  * GAT provides the foundational architecture for our temporal model extension.

---

### Slide 4: Temporal Modeling in Transaction Networks
* **Dynamic Behavior:** Fraud schemes evolve across time periods; static snapshots cannot parameterize temporal sequence.
* **Temporal Approaches in Literature:**
  * **STAN (Cheng et al., 2020):** Spatio-temporal dual-attention framework for transaction sequences.
  * **EvolveGCN (Pareja et al., 2020):** Recurrently adapts GCN weight matrices over snapshot sequences (also evaluated by Weber et al., 2019).
  * **TGN (Rossi et al., 2020):** Continuous-time event-driven memory network (stretch objective).
* **Experimental Split Protocol (Defined for this Study):**
  * **Train Split:** Time steps 1 to 32 (supervised training).
  * **Validation Split:** Time steps 33 to 40 (checkpoint selection and threshold tuning).
  * **Held-Out Test Split:** Time steps 41 to 49 (single final evaluation).
* **Leakage Mitigation:**
  * Chronological splitting prevents future-to-past temporal leakage.
  * Preprocessing audit: Feature scaler is fitted exclusively on $t = 1..32$ data before transforming validation and test partitions.

---

### Slide 5: Recent Temporal and Explainable Fraud Frameworks (Recent Work)
* **XBlock-ExplainNet:**
  * Heterogeneous blockchain graph network with multi-relational attention and rule-extraction modules. Evaluated primarily on smart-contract and phishing address datasets.
* **XG-ScamNet:**
  * Temporal transaction flow network targeting token scam detection with hierarchical graph attention.
* **ML-GNN & MM-TFDX:**
  * Multi-layer and multi-modal temporal fraud detection frameworks combining transaction metadata with dynamic graph embeddings.
* **Methodological Distinctions:**
  * Differ widely in graph formulation (account-level vs transaction-level), evaluation protocols (random k-fold vs strict chronological splits), and whether neighborhood summaries are pre-engineered or dynamically learned.

---

### Slide 6: Feature Preservation & Temporal Explanations (Recent Work)
* **FG-EGCN (Feature-Preserving Evolving GCN):**
  * Identifies the degradation of initial node attributes during multi-hop graph convolutions in temporal networks and introduces residual feature-preservation gates.
* **GRExplainer (Graph Recurrent Explainer — Preprint):**
  * Investigates perturbation-based explanations tailored to temporal message passing and recurrent hidden states.
* **Critical Methodological Perspective:**
  * While these preprints and recent studies propose combined architectures, empirical evaluations frequently omit ablation controls (e.g., zero-time controls, empty-edge controls, or random-deletion baselines).
  * Our work isolates each component under a controlled benchmark.

---

### Slide 7: Critical Literature Comparison Matrix

| Study | Graph Type | Temporal Modeling | Explainability Module | Primary Contribution | Limitation for this Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Weber et al. (2019)** | Directed Transaction Flow | Evaluates GCN & EvolveGCN | ✗ None | Benchmark Elliptic dataset & ML vs GNN comparison | No instance-level explanation mechanism |
| **Dou et al. (2020) CARE-GNN** | Multi-relation Account Graph | ✗ Static | ✗ None | RL-guided neighbor filtering against camouflage | Focuses on static multi-relation graphs; no temporal encoding |
| **Liu et al. (2021) PC-GNN** | Heterogeneous Graph | ✗ Static | ✗ None | Label-balanced sampling under extreme class imbalance | Static graph focus; no post-hoc attribution module |
| **Cheng et al. (2020) STAN** | Spatio-temporal Grid/Sequence | Sequence Attention | ✗ None | Joint spatio-temporal modeling for card fraud | Non-GNN architecture; black-box predictions |
| **Rao et al. (2022) xFraud** | Heterogeneous Transaction Graph | ✗ Snapshot-based | Attention & Path Extraction | End-to-end explainable fraud detection framework | Focuses on heterogeneous static relations rather than discrete temporal encoding |
| **Proposed ETGNN** | Directed Transaction Flow | Fixed Sinusoidal Time Encoding | GNNExplainer with Deletion Controls | Matched ablation of time encoding, message passing & explanation controls | Operates within discrete snapshots without cross-time-step edges |

---

### Slide 8: Proposed ETGNN Architecture
* **Backbone:** 2-layer Graph Attention Network (GAT).
* **Temporal Encoding:**
  * Discrete time steps $t \in [1, 49]$ mapped via fixed sinusoidal positional encoding ($\text{dim} = 16$).
  * Formulated as:
    $$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$
  * Concatenated with node features before graph attention layers ($165 + 16 = 181$ input dimensions).
* **Architectural Boundary:**
  * Operates without cross-snapshot recurrent memory units; messages propagate strictly along intra-time-step directed payment flows.

---

### Slide 9: Post-Hoc Explainability & Evaluation Methodology (RQ3)
* **Module:** GNNExplainer generating instance-level edge masks and node feature masks via mutual information maximization.
* **Evaluation Dimensions:**
  * **Faithfulness (Fidelity):** Measuring signed change in predicted class probability under top-ranked component deletion.
  * **Sparsity:** Quantifying the proportion of edges/features pruned under threshold filtering ($0.3, 0.5, 0.7$).
  * **Stability & Controls:** Benchmarking top-ranked edge removal against **random edge removal** and **bottom-ranked edge removal** across TP, FP, FN, and TN cases.
* **Theoretical Guardrail:**
  * Post-hoc XAI does not alter the trained model parameters.
  * Masks identify influential inputs utilized by the model; they do not establish external causality of fraud.

---

### Slide 10: Research Gap Statement (Refined & Bounded)
> *"Existing studies investigate temporal graph learning, feature preservation, and explainable fraud detection, including frameworks that combine these capabilities. This study examines the contribution of sinusoidal time encoding and graph message passing to illicit-transaction detection on Elliptic through matched baselines and chronological evaluation. It also assesses post-hoc explanation faithfulness using controlled perturbations. The literature comparison will establish which aspects of this evaluation remain insufficiently addressed."*

---

### Slide 11: Controlled Experimental Methodology & Ablation Suite
To isolate individual modeling mechanisms without confounding:
1. **Time-Encoding Ablation (RQ2):**
   * **GAT (Static)** vs. **Matched GAT (Zero-Time, identical dims)** vs. **ETGNN (Sinusoidal Time)**.
2. **Neighborhood Summary Ablation:**
   * **Full Features (165)** vs. **Local-Only Features (93)** on Random Forest and ETGNN.
3. **Message-Passing Ablation (Topological Contribution):**
   * **ETGNN with graph edges** vs. **ETGNN with empty edge tensor (isolated nodes)** on identical local features and splits.
4. **Reliability & Uncertainty Protocol:**
   * Multi-seed repeated training (Seeds 7, 42, 123) with mean ± standard deviation reporting on validation AP.
   * Single final evaluation on held-out test split ($t = 41..49$) using frozen validation-tuned thresholds.

---

### Slide 12: Summary of Empirical Findings & Defense Discussion
* **RQ1 (GNN vs. Tabular ML):**
  * Random Forest achieved the highest test Average Precision (AP: **0.6323**; local-only AP: **0.6301**).
  * In the tested configurations, message passing reduced performance compared to tabular baselines.
* **RQ2 (Temporal Contribution):**
  * On the single test split, ETGNN achieved **0.2250 AP** vs. Matched GAT's **0.2099 AP** (an observed difference of **+0.0151 AP**) and static GAT's **0.1155 AP**.
  * Multi-seed validation AP averaged $0.5292 \pm 0.0160$ (ETGNN) vs. $0.4781 \pm 0.0670$ (Matched GAT), providing indicative validation evidence.
* **Message-Passing Ablation Finding:**
  * Retraining ETGNN without graph edges on local features increased test AP from **0.2344** to **0.4812**, demonstrating that message passing on this intra-snapshot graph reduced classification quality.
* **RQ3 (Explainability Faithfulness):**
  * In the four examined cases, top-ranked edge deletion produced a substantial prediction change for the false-positive case ($-0.1217$, flipping predicted class), while effects for TP, FN, and TN cases were small and comparable to random controls. These results provide limited evidence of edge-explanation faithfulness and require broader evaluation.
