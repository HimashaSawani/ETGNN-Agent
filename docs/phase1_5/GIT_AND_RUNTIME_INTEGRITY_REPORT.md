# Phase 1.5: Git & Runtime Integrity Report

**Document ID:** `GIT_AND_RUNTIME_INTEGRITY_REPORT`  
**Date:** November 2026  
**Auditor:** Senior AI Engineer, MLOps Engineer, Software Architect  
**Workspace:** `d:\ETGNN-Fraud-Detection`  
**Status:** COMPLETE & PRESERVED  

---

## 1. Executive Summary

This report documents the version control and runtime integrity audit conducted across the ETGNN repository. It fulfills the strict Phase 1.5 change-control requirement: **preserve all existing user modifications without auto-committing, pushing, resetting, or altering git history.**

---

## 2. Git Status & Branch Tracking Analysis

### Initial Git State (Audit Baseline)
- **Branch**: `main`
- **Tracking Status**: `ahead of 'origin/main' by 13 commits`
- **Unstaged Changes**:
  - `modified: src/web/data_loader.py`
- **Untracked Artifacts**: Non-breaking thesis build scripts (`build_final_v5_report_docx.py`, `build_nibm_viva_pptx.py`), compiled figures (`figure_3_5_demo_interface.png`, `figure_4_1_temporal_eda.png`), and Phase 1 documentation directories.

### Final Git State (Post Phase 1.5)
- **Branch**: `main` (unchanged)
- **Tracking Status**: `ahead of 'origin/main' by 13 commits` (unchanged)
- **Unstaged Changes**:
  - `modified: src/web/data_loader.py` (strictly preserved)
  - `modified: docs/phase1_audit/ETGNN_DATASET_INTEGRITY_REPORT.md` (Snapshot 43 causality correction)
  - `modified: docs/phase1_audit/ETGNN_MODEL_BASELINE_REPORT.md` (Snapshot 43 causality correction)
- **Untracked Additions**:
  - `tests/test_phase1_5_leakage_and_integrity.py` (new regression test suite)
  - `docs/phase1_5/` (Phase 1.5 reports)
  - `scratch/reproduce_metrics.py` (deterministic reproduction script)

---

## 3. Deep Audit of Existing Modification in `src/web/data_loader.py`

### 3.1 The Modification Diff
```diff
--- a/src/web/data_loader.py
+++ b/src/web/data_loader.py
@@ -755,54 +755,77 @@ class ResearchDataLoader:
         sub_x = self.graph["x"][subset]
         sub_time = self.graph["time_step"][subset]
 
-        if sub_edge_index.numel() == 0:
+        cand_indices = (sub_edge_index[1] == center_sub_idx).nonzero().flatten()
+        E_cand = len(cand_indices)
+        orig_prob = float(self.etgnn_probs[node_idx])
+
+        if sub_edge_index.numel() == 0 or E_cand == 0:
             return {
                 "transaction_id": tx_id,
                 "node_index": node_idx,
                 "is_frozen_research_artifact": False,
                 "is_live_demonstration": True,
-                "message": "Node is an isolated transaction with degree 0; no edges to explain.",
-                "subgraph_nodes": len(subset),
-                "subgraph_edges": 0,
+                "provenance": "Isolated / Zero-incoming degree node in local time-step",
+                "message": "Node has 0 incoming candidate edges in this time-step; no message-passing edges contribute to this node's representation.",
+                "disclaimer": "This node has 0 incoming candidate edges; edge attribution is not applicable.",
+                "subgraph_nodes": int(len(subset)),
+                "subgraph_edges": int(sub_edge_index.shape[1]),
+                "original_predicted_class_probability": orig_prob,
+                "probability_after_intervention": orig_prob,
+                "probability_after_removing_top_edges": orig_prob,
+                "signed_drop": 0.0,
+                "predicted_class_before": int(orig_prob >= 0.5),
+                "predicted_class_after": int(orig_prob >= 0.5),
+                "class_changed": False,
+                "candidate_edges_count": 0,
+                "budget_k": 0,
+                "discrete_deletion_pct": 0.0,
+                "top_edges": [],
             }
```

### 3.2 Correctness & Safety Verification
1. **The Root Cause**: PyTorch Geometric's `GNNExplainer` computes edge-level gradients by optimizing a continuous edge mask over candidate edges pointing to the target node. When a transaction has no incoming directed edges ($E_{\text{cand}} = 0$, as occurs for 76.92% of illicit transactions in test steps 43–49), PyG raises an empty-tensor exception or produces NaN gradients, causing a 500 server crash in FastAPI.
2. **Behavior Under Fallback**:
   - Explicitly populates `candidate_edges_count: 0`.
   - Populates `signed_drop: 0.0` (accurately reflecting that removing zero edges causes zero change in probability).
   - Injects a clear, transparent disclaimer: `"This node has 0 incoming candidate edges; edge attribution is not applicable."`
   - Sets `top_edges: []`.
3. **Misleading Results Check**: The fallback does **not** manufacture synthetic edge attributions or misattribute risk. It transparently discloses the topological reality to the user and API consumer.
4. **Automated Test Coverage**: Verified by regression test `tests/test_phase1_5_leakage_and_integrity.py::test_isolated_node_explanation_handling` (PASSED in 34.97s).

---

## 4. Preservation Declaration

No commits, pushes, rebases, or branch resets were initiated. All existing code changes remain intact in the local working tree.

