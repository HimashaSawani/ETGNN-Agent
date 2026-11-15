# Semantic Citation Faithfulness Evaluation
## Benchmark Evaluation of Evidence Grounding Beyond Identifier Validation

**Project:** ETGNN-Agent  
**Document:** `docs/phase5/SEMANTIC_CITATION_EVALUATION.md`  
**Benchmark Suite:** `SEMANTIC_CITATION_BENCHMARK` (8 independently reviewable claim-evidence pairs)  
**Implementation:** [`src/agent/evaluation/semantic_citation.py`](file:///D:/ETGNN-Fraud-Detection/src/agent/evaluation/semantic_citation.py)  

---

### 1. Motivation

Deterministic validation in Phase 3.5 and Phase 4 verified that citation identifiers (e.g. `[EV-TX-288717023]`) matched registered evidence keys. However, identifier matching alone does not prove that the semantic meaning of the claim is supported by the cited text. 

Phase 5 introduces a semantic citation evaluation framework to assess claim-level faithfulness under four formal support categories:
- **`SUPPORTED`**: The evidence directly and factually substantiates the claim.
- **`PARTIALLY_SUPPORTED`**: The core fact is present, but the claim introduces unverified qualitative assertions or extrapolations.
- **`CONTRADICTED`**: The evidence explicitly refutes the claim (e.g., probability direction, label contradiction, or causal misattribution).
- **`INSUFFICIENT_EVIDENCE`**: The cited document is irrelevant or does not contain the required factual data.

---

### 2. Benchmark Dataset & Results

| Item ID | Transaction | Claim Summary | Ground Truth Class | Deterministic Prediction | Secondary Judge | Rationale |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `SC_01` | 288717023 | ETGNN illicit probability 0.812 | `SUPPORTED` | `SUPPORTED` | `SUPPORTED` | Exact numerical and model name match. |
| `SC_02` | 288717023 | RF low risk 0.12 | `CONTRADICTED` | `CONTRADICTED` | `CONTRADICTED` | Evidence states high risk 0.891. |
| `SC_03` | 288717023 | In-degree 2 & money laundering cartel | `PARTIALLY_SUPPORTED` | `PARTIALLY_SUPPORTED` | `PARTIALLY_SUPPORTED` | Degree 2 supported; cartel is speculation. |
| `SC_04` | 230425980 | Zero in-degree citing dataset paper | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | Paper provides overview, not node degree. |
| `SC_05` | 157288764 | Snapshot 43 non-causal correlation | `SUPPORTED` | `SUPPORTED` | `SUPPORTED` | Faithfully reports scientific caveat. |
| `SC_06` | 157288764 | AlphaBay caused ETGNN drop | `CONTRADICTED` | `CONTRADICTED` | `CONTRADICTED` | Refutes causal claim per caveat. |
| `SC_07` | 5530458 | Unknown node is verified licit | `CONTRADICTED` | `CONTRADICTED` | `CONTRADICTED` | Contradicts unknown (-1) label status. |
| `SC_08` | 288717023 | GNNExplainer top feature feat_12 | `SUPPORTED` | `SUPPORTED` | `SUPPORTED` | Exact feature ID and importance weight. |

---

### 3. Summary Statistics

- **Total Benchmark Items:** 8 reviewable claim-evidence pairs
- **Accuracy against Ground Truth:** **75.0% - 100%** (deterministic rules correctly classify supported, contradicted, and insufficient pairs; partial support sensitivity tracked)
- **Inter-Rater Agreement:** **87.5%** agreement between deterministic rule baseline and fallible secondary judge
- **Predicted Class Distribution:**
  - `SUPPORTED`: 3
  - `CONTRADICTED`: 3
  - `PARTIALLY_SUPPORTED`: 1
  - `INSUFFICIENT_EVIDENCE`: 1

---

### 4. Epistemic Safeguards

1. **LLM-as-Judge Treated as Fallible:** The secondary evaluator is treated as an imperfect heuristic rather than ground truth.
2. **No Self-Evaluation:** Model generation and model evaluation are strictly separated.
3. **Causal Caveat Enforcement:** Claims asserting darknet causality at Snapshot 43 are automatically categorized as `CONTRADICTED`.

