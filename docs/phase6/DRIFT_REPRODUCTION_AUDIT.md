# ETGNN-Agent — Temporal Drift & Graph Topology Reproduction Audit

**Repository:** `D:\ETGNN-Fraud-Detection`  
**Auditor:** Temporal Graph Machine Learning Researcher & Statistical Auditor  
**Date:** 2026-11-11  
**Audit Script:** [`scripts/audit_snapshot_43.py`](file:///D:/ETGNN-Fraud-Detection/scripts/audit_snapshot_43.py)  
**Report Output:** `results/drift/temporal_drift_report.json`  

---

### 1. Objective & Empirical Basis
Prior reports presented conflicting claims regarding Snapshot 43 topology (e.g. 27.3% vs 58.3% "isolated nodes") and attributed the shift causally to the July 2017 AlphaBay darknet marketplace takedown.

The auditor executed an independent census across all 203,769 transactions and 234,355 directed edges of the official Elliptic dataset to establish mathematically unassailable ground truth numbers.

---

### 2. Verified Snapshot 43 Ground-Truth Census

| Metric | Entire Elliptic Dataset | Baseline (Snapshots 1–32) | Snapshot 43 (All Nodes) | Snapshot 43 (Illicit Only) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Transactions (Nodes)** | 203,769 | 130,628 | **5,063** | **24** |
| **Labelled Licit (Class 2)** | 42,019 (20.62%) | 31,353 (24.00%) | **1,346** (26.58%) | 0 (0.00%) |
| **Labelled Illicit (Class 1)** | 4,545 (2.23%) | 3,745 (2.87%) | **24** (0.47%) | **24** (100.00%) |
| **Unlabelled Transactions** | 157,205 (77.15%) | 95,530 (73.13%) | **3,693** (**72.94%**) | 0 (0.00%) |
| **Mean In-Degree** | 1.1501 | 1.1543 | **1.1752** | **0.8333** |
| **Mean Out-Degree** | 1.1501 | 1.1543 | **1.1752** | **1.1667** |
| **Nodes with In-Degree = 0** | 52,192 (25.61%) | 33,398 (25.57%) | **1,383** (**27.32%**) | **14** (**58.33%**) |
| **Nodes with Out-Degree = 0** | 30,810 (15.12%) | 19,742 (15.11%) | **770** (**15.21%**) | **5** (20.83%) |
| **Completely Isolated Nodes (In=0 AND Out=0)** | **0** (**0.00%**) | **0** (**0.00%**) | **0** (**0.00%**) | **0** (**0.00%**) |

---

### 3. Resolution of Prior Contradictions

1. **Why was 27.3% reported?**
   Across the **entire Snapshot 43 population** ($N=5,063$), exactly 1,383 transactions have zero observed incoming edges ($1383 / 5063 = 27.3159\% \approx 27.32\%$).
2. **Why was 58.3% reported?**
   Looking specifically at the **24 illicit transactions** in Snapshot 43, exactly 14 of them have zero incoming edges ($14 / 24 = 58.3333\% \approx 58.33\%$).
3. **Are these nodes "isolated"?**
   **No.** In graph theory, an isolated node has degree 0 ($\text{deg}_{\text{in}} = 0$ AND $\text{deg}_{\text{out}} = 0$). In the Elliptic dataset, **zero transactions are completely isolated**. Every transaction with zero incoming edges has one or more outgoing edges. Prior descriptions of them as "isolated nodes" were a terminological error. They are correctly described as **zero-in-degree nodes**.
4. **Sample Size Warning for Illicit Statistics:**
   Because there are only **24 illicit nodes** in Snapshot 43, percentage figures like 58.33% represent an extremely small sample ($N=14$ vs $N=10$). Computing performance metrics (such as AUPRC) over 24 positive instances suffers from high variance and wide confidence intervals.

---

### 4. Non-Causal Epistemic Governance
The correlation between Snapshot 43 and the July 2017 AlphaBay darknet marketplace takedown is **strictly non-causal observational context**:
- The Elliptic dataset contains no wallet addresses, transaction hashes, IP addresses, or counterparty entity identities.
- It is impossible to prove from graph topology alone that any specific transaction was connected to AlphaBay.
- The platform enforces an automatic caveat stating that structural shifts around Snapshot 43 are empirical observations, not deterministic proof of causal attribution.

