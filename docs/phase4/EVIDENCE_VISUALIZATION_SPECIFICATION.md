# ETGNN-Agent — Phase 4: Evidence Visualization Specification
## Multi-Hop Graph Rendering, Snapshot 43 Topology Qualifications & GNNExplainer Attribution

**Components:**
- `InteractiveGraphEvidenceViewer.tsx`
- `ModelComparisonPanel.tsx`
- `EvidenceExplorer.tsx`

---

## 1. Visualizing Multi-Hop Topological Evidence

Bitcoin transactions operate as directed acyclic graphs (DAGs) representing currency flow between UTXOs and wallets. In the Elliptic dataset:
- Nodes represent transactions.
- Directed edges represent payment flows from input transactions to output transactions.

The `InteractiveGraphEvidenceViewer` parses and displays:
1. **In-Degree ($\text{deg}^-$)**: Number of parent transaction inputs directly funding the transaction.
2. **Out-Degree ($\text{deg}^+$)**: Number of spending outputs forwarding funds to recipient transactions.
3. **2-Hop Neighborhood Size**: Total transaction count within a 2-hop radius, reflecting local clustering density.
4. **Illicit Counterparty Taint**: Count of verified illicit neighbors within the 2-hop envelope.

---

## 2. Snapshot 43 Boundary Conditions & Qualification Taxonomy

In the Elliptic dataset, Snapshot 43 corresponds to the historical shutdown of the Silk Road / Hydra marketplace darknet services:
- **Structural Anomaly**: In Snapshot 43, graph density drops significantly, and illicit transaction clustering exhibits non-stationary distribution shifts.
- **Topological Edge Cases**:
  - `ISOLATED_NODE` ($\text{deg}^- = 0 \land \text{deg}^+ = 0$): Transaction has no observed inputs or outputs in the temporal snapshot. The copilot displays:
    > *Isolated Transaction: Node has 0 incoming and 0 outgoing edges in this temporal window. GNN message passing relies strictly on local features.*
  - `SOURCE_TRANSACTION` ($\text{deg}^- = 0 \land \text{deg}^+ > 0$): Funding origin or unlinked input.
  - `SINK_TRANSACTION` ($\text{deg}^- > 0 \land \text{deg}^+ = 0$): Final destination or unspent output.

---

## 3. GNNExplainer Visual Integration

The drawer visualizes GNNExplainer subgraphs and importance weights:
1. **Top Influential Neighbors**: Ranked list of neighbor node IDs with their edge attribution weights ($w_{ij} \in [0, 1]$).
2. **Top Influential Features**: High-salience local transaction attributes identified by the explainer's feature mask optimization.
3. **Cross-Tab Deep Linking**: Clicking **"Open in Full Visualizer"** navigates directly to the Cytoscape/ForceGraph visualizer in the main platform view with the target node selected.

