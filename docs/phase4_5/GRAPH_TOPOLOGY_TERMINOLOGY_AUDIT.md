# Directed Graph Topology Terminology & Semantics Audit (Phase 4.5)

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 4.5 Stabilization & Scientific Integrity Audit  
**Status:** COMPLETE & VERIFIED  

---

## 1. Scientific & Topological Problem Statement

The Elliptic dataset represents Bitcoin payment flows as a directed graph $G_t = (V_t, E_t)$ partitioned into 49 discrete snapshots. Each node $v \in V_t$ represents a transaction, and each directed edge $(u, v) \in E_t$ represents a payment flow from input transaction $u$ to output transaction $v$ within snapshot $t$.

During Phase 4, certain UI badges and textual descriptions risked over-interpreting directed degree statistics:
1. **Conflating `in_degree = 0` with Complete Graph Isolation or Fund Origination:** A transaction with `in_degree = 0` and `out_degree > 0` has outgoing edges within the snapshot; it is **not** a fully isolated node, nor does `in_degree = 0` prove that the transaction originated funds (e.g., coinbase generation), since upstream parent transactions from earlier snapshots or outside the sampled subgraph are simply absent from the snapshot's observed edge set.
2. **Conflating `out_degree = 0` with Unspent Funds or Final Accumulation:** A transaction with `out_degree = 0` and `in_degree > 0` indicates an absence of observed downstream spending edges within the snapshot sample, not proof of terminal wallet hoarding or unspent outputs on the full Bitcoin ledger.
3. **Counterparty Labeling Semantics:** Referring to connected nodes with ground-truth label `1` as *"verified illicit counterparties"* risked implying confirmed real-world identity or direct criminal counterparty verification beyond the dataset's anonymized node labels.

---

## 2. Enforced Topological Distinctions

### 2.1 Zero Incoming Observed Edges (`in_degree = 0, out_degree > 0`)
- **UI Badge:** `0 Observed In-Edges ({outDegree} Out)`
- **Report & Prompt Qualification:**
  > *"Topological Qualification: Node has zero incoming observed edges represented in the directed graph; this indicates an absence of observed upstream edges in the dataset sample, not proof that the transaction originated funds."*

### 2.2 Zero Outgoing Observed Edges (`in_degree > 0, out_degree = 0`)
- **UI Badge:** `{inDegree} In-Edges (0 Observed Out)`
- **Report & Prompt Qualification:**
  > *"Topological Qualification: Node has zero outgoing observed edges represented in the directed graph; this indicates an absence of observed downstream spending edges, not proof of unspent funds or final accumulation."*

### 2.3 Fully Isolated Node (`in_degree = 0 AND out_degree = 0`)
- **UI Badge:** `Fully Isolated Node (0 In, 0 Out)`
- **Report & Prompt Qualification:**
  > *"Topological Qualification: Node is fully isolated in snapshot graph (both in-degree and out-degree are 0); structural message passing is inactive and prediction relies strictly on local tabular attributes."*

### 2.4 Neighbor Label Provenance Terminology
- Replaced *"Illicit Counterparties"* / *"Verified Illicit Counterparties"* with **`Illicit-Label Neighbors`**.
- Added explicit topological disclaimer in `InteractiveGraphEvidenceViewer.tsx`:
  > *"Elliptic edges represent directed transaction payment flows within a discrete temporal snapshot, not confirmed wallet ownership, transaction amounts, or direct real-world counterparty relationships."*

---

## 3. Files Audited & Updated

| File | Update Summary |
| :--- | :--- |
| `web/frontend/src/components/agent/InteractiveGraphEvidenceViewer.tsx` | Differentiated `isFullyIsolated` (`in=0, out=0`), `isZeroIncoming` (`in=0, out>0`), and `isZeroOutgoing` (`in>0, out=0`) badges; renamed metric card to `Illicit-Label Neighbors`; added topological disclaimer. |
| `src/agent/llm/provider.py` | Updated `MockLLMProvider` and `EvidenceOnlyProvider` to emit distinct topological qualifications for `is_fully_isolated`, `is_isolated_in`, and `out_deg == 0`. |
| `src/agent/llm/prompts.py` | Added Directive 7 enforcing strict distinctions between zero in-degree, zero out-degree, full isolation, and `illicit-label neighbors`. |
| `src/agent/validation/report_validator.py` | Enforced automatic injection of the exact degree-specific qualification when `in_deg == 0` or `out_deg == 0`. |
| `src/agent/reports/pdf_exporter.py` | Updated neighborhood summary to report `illicit-label neighbors` and accurately distinguish `is_fully_isolated`. |

---

## 4. Verification Evidence

1. **Automated Python Unit Tests (`tests/test_phase4_5_reliability_and_integrity.py`):**
   - `test_graph_topology_degree_distinctions`: Verifies `ReportValidator` produces distinct, accurate qualifications across Case 1 (`in=0, out=2`), Case 2 (`in=2, out=0`), and Case 3 (`in=0, out=0`). — **PASSED**
2. **Real-Browser Playwright E2E Verification (`web/frontend/e2e/investigation_copilot.spec.ts`):**
   - `Scenario H`: Verifies `0 Observed In-Edges` and `Illicit-Label Neighbors` are rendered in Chromium for a node with `in_degree=0, out_degree=2`. — **PASSED (2.2s)**

