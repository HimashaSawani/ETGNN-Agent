# Phase 1.5: Snapshot 43 Historical Causality & Structural Analysis Review

**Document ID:** `SNAPSHOT43_CAUSALITY_REVIEW`  
**Date:** November 2026  
**Auditor:** Senior AI Engineer, Graph ML Researcher, Research Integrity Auditor  
**Workspace:** `d:\ETGNN-Fraud-Detection`  
**Status:** VERIFIED & CORRECTED  

---

## 1. Executive Summary

Previous research documentation and presentation slides in the repository asserted that the structural shift in **Elliptic Snapshot 43** was caused by a law enforcement takedown of a specific darknet marketplace (such as Hydra or AlphaBay). 

This review conducted an exhaustive investigation across:
1. All occurrences of this claim in code, markdown reports, and presentation scripts.
2. The authoritative dataset documentation by Weber et al. (2019, *arXiv:1908.02591*).
3. The underlying transaction and edge data in `data/processed/nodes_with_masks.csv` and `data/raw/elliptic/elliptic_txs_edgelist.csv`.

### Key Conclusions:
- **Causality Classification**: **UNESTABLISHED CAUSALITY / CORRELATION HYPOTHESIS**.
- **Formal Scientific Correction Applied**:  
  > *"Snapshot 43 exhibits a substantial structural change in the evaluated transaction graph. The cause of this change has not been independently established."*
- **Mathematical Metric Verification**: The previously reported numbers **13.52%** and **76.92%** are **100% mathematically reproduced**, but represent **illicit transactions with zero incoming directed candidate edges ($k_{\text{in}} = 0$)**, not total undirected isolated nodes.

---

## 2. Inventory of Claims Across the Repository

| File Path | Original Text / Claim | Evidentiary Assessment | Correction / Action Taken |
|---|---|---|---|
| `docs/phase1_audit/ETGNN_DATASET_INTEGRITY_REPORT.md` | "Between snapshot 42 and snapshot 44, law enforcement agencies coordinated a global takedown of a major darknet marketplace (historically linked to Hydra / AlphaBay ecosystem shifts)..." | **Unsupported Speculation**. Neither Hydra nor AlphaBay is referenced or provable from Elliptic data. | Replaced with verified structural statement and explicit non-causal disclaimer. |
| `docs/phase1_audit/ETGNN_MODEL_BASELINE_REPORT.md` | "At time step 43 (corresponding to a major historical darknet marketplace takedown), the proportion of isolated nodes ($E=0$) surges from 13.52% to 76.92%..." | **Overstated Causality & Imprecise Metric Label**. | Corrected to "illicit transactions with zero incoming candidate edges" and qualified causality. |
| `scripts/build_nibm_viva_pptx.py` | "Darknet Market Takedown (Step 43): Coincided with a major law enforcement shutdown of a darknet entity (Weber et al., 2019)..." | **Partial Correlation**. Weber et al. note a shutdown occurred around step 43, but without entity attribution. | Preserved as historical context citing Weber et al., with explicit qualification that causality is unproven. |
| `src/web/data_loader.py` | "Across the step 43 darknet takedown, illicit transactions with zero candidate edges jump from 13.52% to 76.92%." | **Colloquial Terminology**. | Clarified in documentation; data loader logic operates strictly on candidate edge counts. |
| `web/frontend/src/components/GuidedTour.tsx` | "Topological Censorship Shock: Across the step 43 darknet marketplace shutdown..." | **User-Facing Metaphor**. | Qualified as an empirical phenomenon observed in the benchmark graph. |

---

## 3. Analysis of Authoritative Literature (Weber et al., 2019)

In the seminal Elliptic benchmark paper (*"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"*, KDD 2019 Workshop, *arXiv:1908.02591*), Weber et al. state:

> *"Around time step 43, a shutdown of a dark market occurs..."*  
> — Section 4.1, "Temporal testing", Page 4

### Evidentiary Critique:
1. **No External Entity Named**: Weber et al. do not name AlphaBay, Hydra, Hansa, or Silk Road in connection with time step 43.
2. **Proprietary Label Attribution**: Elliptic's internal labeling methodology matched Bitcoin transactions with off-chain legal and intelligence data. Because those raw transaction mappings remain proprietary and confidential, independent researchers cannot audit the exact cryptographic transactions that formed that shutdown.
3. **Correlation vs Causality**: While model performance ($F_1$ score) deteriorated precipitously after step 43, an observational graph dataset cannot definitively prove that market shutdown was the sole or primary cause of graph edge sparsity; changes in mixer usage, transaction fee spikes, batching practices, or Elliptic crawler sampling could equally contribute.

---

## 4. Independent Mathematical Reproduction of Graph Sparsity

A script was executed on `data/processed/nodes_with_masks.csv` and `data/raw/elliptic/elliptic_txs_edgelist.csv` across all test snapshots (steps 41–49).

### Empirical Results Table

| Snapshot Step | Total Nodes | Illicit Nodes ($y=1$) | Illicit with $k_{\text{in}} = 0$ | Zero-Incoming % | Undirected Isolated Nodes ($k=0$) |
|---|---|---|---|---|---|
| **Step 40** | 4,481 | 112 | 33 | 29.46% | 0 (0.00%) |
| **Step 41** | 5,342 | 116 | 6 | 5.17% | 0 (0.00%) |
| **Step 42** | 7,140 | 239 | 42 | 17.57% | 0 (0.00%) |
| **Step 43** | 5,063 | 24 | 14 | 58.33% | 0 (0.00%) |
| **Step 44** | 4,975 | 24 | 14 | 58.33% | 0 (0.00%) |
| **Step 45** | 5,598 | 5 | 3 | 60.00% | 0 (0.00%) |
| **Step 46** | 3,519 | 2 | 1 | 50.00% | 0 (0.00%) |
| **Step 47** | 5,121 | 22 | 11 | 50.00% | 0 (0.00%) |
| **Step 48** | 2,954 | 36 | 36 | 100.00% | 0 (0.00%) |
| **Step 49** | 2,454 | 56 | 51 | 91.07% | 0 (0.00%) |

### Pre-Shift vs Post-Shift Grouping
- **Pre-Shift Test Period (Snapshots 41–42)**:
  $$\text{Total Illicit} = 116 + 239 = 355$$
  $$\text{Zero Incoming Directed Edges} = 6 + 42 = 48$$
  $$\text{Pre-Shift Rate} = \frac{48}{355} = \mathbf{13.5211\%} \implies \mathbf{13.52\%}$$
- **Post-Shift Test Period (Snapshots 43–49)**:
  $$\text{Total Illicit} = 24 + 24 + 5 + 2 + 22 + 36 + 56 = 169$$
  $$\text{Zero Incoming Directed Edges} = 14 + 14 + 3 + 1 + 11 + 36 + 51 = 130$$
  $$\text{Post-Shift Rate} = \frac{130}{169} = \mathbf{76.9231\%} \implies \mathbf{76.92\%}$$

### Critical Graph Topology Distinctions:
1. **Directed In-Degree vs Undirected Degree**: In the raw dataset, every node has at least one edge in `elliptic_txs_edgelist.csv` (100% connectivity, 0% total isolation).
2. **Directed Convolutions**: In PyTorch Geometric, message passing propagates along directed edges from sender to receiver:
   $$h_i^{(l+1)} = \text{AGG} \left( \{ h_j^{(l)} \mid (j, i) \in \mathcal{E} \} \right)$$
   For an illicit target node $i$, if no edge $(j, i)$ exists in the snapshot, the node receives no incoming message updates from neighbor nodes.
3. Therefore, calling this phenomenon "isolated nodes" without qualification was scientifically ambiguous. The rigorous terminology is **"illicit transactions with zero incoming candidate message-passing edges."**

---

## 5. RAG Knowledge Base Quarantine Policy

To ensure that the upcoming Phase 2–3 Agentic RAG system does not inject unverified historical claims:
1. **Quarantine Rule**: Unsubstantiated claims attributing Snapshot 43 to specific darknet organizations (e.g. Hydra, AlphaBay) are explicitly excluded from the RAG knowledge corpus.
2. **Approved Corpus Statement**:
   > *"Empirical analysis of the Elliptic test split demonstrates that in snapshots 43–49, 76.92% of illicit transactions had zero incoming directed candidate edges within their time-step, compared to 13.52% in snapshots 41–42. This topological shift severely starves GNN message passing. While Weber et al. (2019) hypothesized an association with a dark market shutdown, the specific cause remains unverified."*

---

## 6. Workstream 1 Conclusion & Status

- **Status**: **PASS WITH LIMITATIONS** (Empirical metrics 100% verified; historical causality properly qualified and bounded).
- **Regression Test Added**: `tests/test_phase1_5_leakage_and_integrity.py::test_snapshot_43_topology_calculations` (PASSED).

