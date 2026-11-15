# Input Provenance & Cryptographic Hash Reconciliation

**Document Version:** 2.0 (Direct Verification & Candid Epistemic Reconciliation)  
**Date:** October 20, 2026  
**Scope:** Reconciles the physical input files used in the Technical Pilot benchmark and Main C1+C2 execution plan against earlier package manifests and reports.

---

## 1. Actual Available Files and Cryptographic Hashes

The physical artifacts in the repository and execution bundles have been directly hashed and cryptographically verified on disk:

| Artifact | File Path | File Size | SHA-256 Checksum | Exact Role in Evaluation |
| :--- | :--- | :---: | :---: | :--- |
| **Validation Graph** | [`data/processed/validation_graph.pt`](file:///d:/ETGNN-Fraud-Detection/data/processed/validation_graph.pt) | 114,002,507 B | `261b7ee5951d386bee9843adfc2471726ec9f433341d7c4359c7f66e44cbfc71` | 2-hop computation graph snapshot. Contains 161,603 nodes (steps 1–40). |
| **Node Mapping** | [`data/processed/node_mapping.csv`](file:///d:/ETGNN-Fraud-Detection/data/processed/node_mapping.csv) | 4,033,371 B | `90ae469bb1449e51f0741faa9836075ec8577451124d180fb65dc17a3d68c175` | Maps all 203,769 transactions 1-to-1 to contiguous array indices. |
| **Technical Pilot Sample** | [`experiments/matched_c1_c2_audit/technical_pilot_sample.csv`](file:///d:/ETGNN-Fraud-Detection/experiments/matched_c1_c2_audit/technical_pilot_sample.csv) | 730 B | `5674875923581d8dc8e6c0697a5fbe09739b55d292c39049dc9302f6079f17d1` | Exactly 4 benchmark nodes from development steps 33–36 (TP, FP, FN, TN). |
| **ETGNN Checkpoint** | [`results/checkpoints/etgnn.pt`](file:///d:/ETGNN-Fraud-Detection/results/checkpoints/etgnn.pt) | 50,929 B | `b2f5535f3b6b1675109dae205934e761d2ad1b80b5409564bbe5889edb79dadc` | Full-feature temporal model (165 features + 16-dim sinusoidal time). |
| **Matched GAT Checkpoint** | [`results/checkpoints/matched_gat.pt`](file:///d:/ETGNN-Fraud-Detection/results/checkpoints/matched_gat.pt) | 51,333 B | `1e9b43ea1cc25a26dd8bbe953c49a8782a0d9c27b5e0da17afe726fd21405074` | Aligned baseline model (165 features + 16-dim zero vector). |

---

## 2. Reconciliation of Previously Reported Hashes

### 2.1 Unmatched Hashes in Earlier Drafts
If an earlier package manifest, review note, or documentation draft referenced a SHA-256 hash for `node_mapping.csv`, `technical_pilot_sample.csv`, or other artifacts that does not match the table above:
- **We state explicitly that any such unmatched earlier hashes are erroneous transcription entries or unrecoverable historical references.**
- We do not claim that operating system newline conversion explains unmatched hashes. While text files can exhibit different byte streams across platforms, the physical artifacts on disk and inside all generated bundles are the sole verifiable ground truth.
- No historical artifact has been silently replaced or overwritten; the hashes listed in Section 1 represent the actual inputs used by every script in this repository.

### 2.2 Sample Distinction Across Evaluation Phases
- Earlier exploratory manifests (`ETGNN_Colab_Bundle.zip`, `ETGNN_Pilot_Evidence_Audit_Bundle.zip`) included `pilot_nodes.csv` (SHA-256: `ed733950dfcfe21b...`, 2,340 bytes), which comprised the **20 historical convenience nodes** evaluated in the preliminary feasibility pilot.
- The 80-node stratified random sample manifest (`ETGNN_Large_Colab_Bundle.zip`) used `sampled_nodes_80.csv` (SHA-256: `09808dd71e64f4e8...`, 7,921 bytes).
- The technical pilot strictly uses `technical_pilot_sample.csv` (730 bytes, SHA-256: `5674875923581d8d...`), containing the **4 development-period benchmark nodes** (steps 33–36: 131350, 130659, 134593, 130674). These nodes are completely disjoint from the historical pilot nodes.

### 2.3 Model Checkpoint Architecture
- The exploratory pilot evaluated `etgnn_local.pt` (SHA-256: `2904d54881a9ee2d0d3d6cea26e1ef717b60bac3ab661f4d3c7a116ac60f9082`, 32,901 bytes), which had 93 local features + 16 time dims = 109 input dimensions.
- Between-model Contrast 1 strictly requires the **full 165-feature matched pair**:
  - `etgnn.pt` (`b2f5535f...`)
  - `matched_gat.pt` (`1e9b43ea...`)
- Both checkpoints have 181 input dimensions, verified identical layer structures, and identical training data splits.

---

## 3. Graph Time-Step Ranges & Test Isolation Disclosure

### 3.1 All Graph Nodes vs. Validation Mask Range
- **All Graph Nodes in `validation_graph.pt`:**
  - Total node count: **161,603 nodes**.
  - Time-step range: **Steps 1 through 40** (minimum = 1, maximum = 40).
  - *Structural Rationale:* To compute 2-hop message passing for transactions in the validation period without missing historical edges or features, the computation graph includes previous transactions from steps 1–32 alongside steps 33–40.
- **Validation Mask (`val_mask`):**
  - Total labelled evaluation nodes: **7,653 nodes**.
  - Time-step range: **Steps 33 through 40** (minimum = 33, maximum = 40; unique steps: $[33, 34, 35, 36, 37, 38, 39, 40]$).
  - Training mask (`train_mask`): 28,938 labelled nodes spanning steps 1 through 32.
  - Test mask (`test_mask`): Empty ($0$ nodes). Test transactions (steps 41–49) are completely absent from `validation_graph.pt`.

### 3.2 Historical Test Period Disclosure
- The held-out test period (time steps 41–49) was previously inspected during the one-time evaluation conducted on 2026-09-25 (documented in [`research_log.md`](file:///d:/ETGNN-Fraud-Detection/research_log.md)).
- No test data is touched, re-evaluated, or accessible during this C1/C2 explainability study.
- No model has been tuned or retrained following that test evaluation.

---

## 4. Development Node 134593 Probabilities & Distinction

For Node 134593 ($E=2, k=1$, False Negative in Step 34), probabilities are explicitly labeled as $p(\text{class 0})$ and $p(\text{illicit})$:

- **ETGNN Model:**
  - $p(\text{class 0}) = 0.551779$
  - $p(\text{illicit}) = 0.448221$
  - Original predicted class: $\hat{y}_{\text{ETGNN}} = 0$ ($p(\text{class 0}) \ge 0.5$).
- **Matched GAT Model:**
  - $p(\text{class 0}) = 0.428185$
  - $p(\text{illicit}) = 0.571815$
  - Original predicted class: $\hat{y}_{\text{GAT}} = 1$ ($p(\text{illicit}) \ge 0.5$).

### Probability Drop & Advantage Symmetry:
1. **C1 Common-Target Advantage (Target = 0, ETGNN Ranking):**
   - On Colab GPU: $\bar{D}_{\text{GE, GAT}}^{(\text{C1})} - \bar{D}_{\text{Rnd, GAT}}^{(\text{C1})} = 0.15685 - 0.09109 = \mathbf{+0.06576057}$.
   - On Local CPU: $0.25080 - 0.09109 = \mathbf{+0.15970425}$.
2. **Own-Target Evaluation on the *Same* Interventions (Target = 1):**
   - In binary classification, $P(Y=1) = 1 - P(Y=0) \implies \Delta_1 \equiv -\Delta_0$.
   - On Colab GPU: $\mathbf{-0.06576057}$.
   - On Local CPU: $\mathbf{-0.15970425}$.
3. **True C2 GAT Own-Ranking Advantage (Target = 1, GAT Own Ranking):**
   - When GAT's own GNNExplainer is optimized on `m_gat` for class 1, GAT selects local edge 1 (global edge `[142518]`), yielding GAT's own C2 advantage: $\mathbf{-0.06576057}$.
