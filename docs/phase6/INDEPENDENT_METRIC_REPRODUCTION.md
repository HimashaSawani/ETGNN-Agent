# ETGNN-Agent — Independent Metric Reproduction & Forensic Audit

**Repository:** `D:\ETGNN-Fraud-Detection`  
**Auditor:** Independent ML Evaluation Auditor  
**Date:** 2026-11-11  

---

### 1. Audit Scope & Methodology
The auditor inspected `results/benchmarks/phase5_evaluation_results.json` and `results/benchmarks/phase5_agent_ablation_results.json` alongside the test runner `src/agent/evaluation/runner.py` and evaluation dataset `src/agent/evaluation/datasets/phase5_eval_dataset.py`.

The goal was to verify whether the reported benchmark percentages were mathematically derivable from recorded integer counts over evaluation fixtures, or whether they represented synthetic/heuristic estimates.

---

### 2. Discrepancy Analysis: The Non-Reproducibility Finding

#### Primary Finding: Macro Metrics vs. Integer Arithmetic
In Phase 5 narrative reports, the following macro-level performance metrics were asserted:
- **Citation Precision:** 98.5%
- **Citation Coverage:** 96.5%
- **Numerical Correctness:** 99.1%
- **Tool Selection Accuracy:** 94.2%

When inspecting the underlying benchmark run:
- In `results/benchmarks/phase5_evaluation_results.json`, evaluation is run over $N=6$ core task fixtures (`T1` through `T6`).
- In `results/benchmarks/phase5_agent_ablation_results.json`, evaluation is run over $N=11$ task fixtures (`T1` through `T11`).

**Mathematical Impossibility:**
For $N=6$:
- $5/6 \approx 83.33\%$
- $6/6 = 100.00\%$

For $N=11$:
- $10/11 \approx 90.91\%$
- $11/11 = 100.00\%$

Even considering individual citation counts (e.g. 23 citations across 6 tasks):
- $22/23 \approx 95.65\%$
- $23/23 = 100.00\%$

Numbers such as `98.5%`, `96.5%`, `99.1%`, and `94.2%` **cannot be produced by any integer ratio** of successful items to total items over the evaluated tasks.

**Conclusion:** The reported macro figures in earlier documentation were **heuristic narrative estimates** rather than strict empirical reductions from recorded test runs.

---

### 3. Exact Empirical Ground-Truth Numbers (Phase 6 Audit)

The auditor re-ran `run_phase5_ablation_benchmark()` directly on the repository code. Below are the verified integer ratios and mathematically exact percentages across all 6 evaluated systems on $N=11$ fixtures:

| Evaluated System | Tasks Completed | Total Claims Checked | Supported Claims | Exact Claim Support Rate | Valid Citations / Total Citations | Exact Citation Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full LangGraph + Graph-RAG (Production)** | 11 / 11 | 38 | 38 | **100.00%** (38/38) | 33 / 33 | **100.00%** |
| **Evidence-Only Deterministic Baseline** | 11 / 11 | 22 | 22 | **100.00%** (22/22) | 22 / 22 | **100.00%** |
| **No-Validation Workflow (Ablation)** | 11 / 11 | 38 | 31 | **81.58%** (31/38) | 28 / 33 | **84.85%** |
| **Simulated Single-Pass Heuristic** | 11 / 11 | 18 | 11 | **61.11%** (11/18) | 7 / 18 | **38.89%** |
| **Simulated Tabular-Only Heuristic** | 11 / 11 | 24 | 19 | **79.17%** (19/24) | 18 / 24 | **75.00%** |
| **Isolated Topology-Only Baseline** | 11 / 11 | 20 | 18 | **90.00%** (18/20) | 16 / 20 | **80.00%** |

---

### 4. Integrity Rectification
1. Synthetic macro figures have been deprecated and removed from the active monitoring dashboard.
2. The monitoring dashboard and API now expose verified integer counts, exact fractions, and system-level breakdowns.
3. Every claim verification outcome is linked directly to its `EvidenceRecord` ID in `results/benchmarks/phase5_agent_ablation_results.json`.

