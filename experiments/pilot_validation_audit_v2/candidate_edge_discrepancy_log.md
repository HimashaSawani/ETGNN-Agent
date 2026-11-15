# Candidate Edge Count Reconciliation Log: 2,712 vs. 2,544

## 1. Discrepancy Statement
In draft document `pilot_report.md` (Section 3.2), the candidate edge count was reported as:
> *"Candidate edge count ranged from 6 (node 157389) to 367 (node 148292). Total candidate edges across 20 nodes = 2,544."*

However, the empirical evidence files show:
- `pilot_nodes.csv` sum of `candidate_edges_count`: **2,712**
- `candidate_edges.csv` row count: **2,712**
- `raw_interventions.csv` candidate edge references: **2,712**
- Graph extraction code `k_hop_subgraph(..., flow="source_to_target")` sum: **2,712**

Numerical discrepancy: $2,712 - 2,544 = +168$ edges (+6.6%).

---

## 2. Root Cause Analysis
Re-running the extraction code `k_hop_subgraph(pos, 2, val_graph["edge_index"], flow="source_to_target")` against `data/processed/validation_graph.pt` on all 20 nodes proves that `2,712` is the true, ground-truth candidate edge count.

The number `2,544` was a manual drafting addition error in the preliminary narrative report text. Every raw trial record, candidate edge record, and model intervention computed between 2026-10-19 and the Colab audit strictly used the correct candidate counts summing to **2,712**.

---

## 3. Per-Node Ground-Truth Reconciliation Table

| Node Index | Category | Time Step | In-Degree | Subgraph Nodes | `pilot_nodes.csv` Count | `candidate_edges.csv` Rows | Extraction Code Count | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 148166 | TP | 37 | 177 | 179 | 233 | 233 | 233 | MATCH |
| 130726 | TP | 33 | 166 | 172 | 212 | 212 | 212 | MATCH |
| 133786 | TP | 34 | 58 | 78 | 90 | 90 | 90 | MATCH |
| 133832 | TP | 34 | 53 | 72 | 72 | 72 | 72 | MATCH |
| 133884 | TP | 34 | 53 | 67 | 73 | 73 | 73 | MATCH |
| **TP Subtotal** | | | | | **680** | **680** | **680** | **MATCH** |
| 130690 | FP | 33 | 65 | 141 | 147 | 147 | 147 | MATCH |
| 130935 | FP | 33 | 55 | 112 | 111 | 111 | 111 | MATCH |
| 136379 | FP | 35 | 54 | 134 | 153 | 153 | 153 | MATCH |
| 154421 | FP | 39 | 47 | 56 | 61 | 61 | 61 | MATCH |
| 133801 | FP | 34 | 40 | 72 | 72 | 72 | 72 | MATCH |
| **FP Subtotal** | | | | | **544** | **544** | **544** | **MATCH** |
| 142234 | FN | 36 | 43 | 81 | 81 | 81 | 81 | MATCH |
| 142515 | FN | 36 | 38 | 46 | 45 | 45 | 45 | MATCH |
| 148265 | FN | 37 | 11 | 12 | 11 | 11 | 11 | MATCH |
| 157211 | FN | 40 | 5 | 17 | 16 | 16 | 16 | MATCH |
| 157389 | FN | 40 | 5 | 7 | 6 | 6 | 6 | MATCH |
| **FN Subtotal** | | | | | **159** | **159** | **159** | **MATCH** |
| 148292 | TN | 37 | 241 | 315 | 367 | 367 | 367 | MATCH |
| 154471 | TN | 39 | 148 | 271 | 286 | 286 | 286 | MATCH |
| 148195 | TN | 37 | 129 | 320 | 363 | 363 | 363 | MATCH |
| 154373 | TN | 39 | 128 | 153 | 154 | 154 | 154 | MATCH |
| 151484 | TN | 38 | 124 | 154 | 159 | 159 | 159 | MATCH |
| **TN Subtotal** | | | | | **1,329** | **1,329** | **1,329** | **MATCH** |
| **GRAND TOTAL** | | | | | **2,712** | **2,712** | **2,712** | **VERIFIED** |

---

## 4. Conclusion and Corrective Action
The true, verified candidate edge count is **2,712**.
All versioned reports, summary tables, and documentation have been updated to cite **2,712**.
