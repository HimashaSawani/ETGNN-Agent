# ETGNN-Agent — Ablation Validity Audit

**Repository:** `D:\ETGNN-Fraud-Detection`  
**Auditor:** Independent ML Evaluation Auditor  
**Date:** 2026-11-11  

---

### 1. Executive Summary
A critical audit requirement for Phase 6 was to inspect how ablation baselines were executed in `src/agent/evaluation/runner.py`. Specifically, did the "Single-Pass LLM without RAG" and "LLM + RAG without Graph Tools" ablations make actual calls to an LLM provider, or were they simulated heuristics?

### 2. Code Inspection & Verification
Inspection of `src/agent/evaluation/runner.py` revealed:
```python
def _run_single_pass_llm_ablation(task: EvaluationTask) -> Dict[str, Any]:
    # Formulates prompt without retrieved RAG documents
    # Executes local deterministic regex and feature extractors
    ...
```
Neither of these ablation functions dispatched requests to `gemini-1.5-pro`, `claude-3-5-sonnet`, or `ollama`. Instead, they executed Python deterministic mock generators that omitted RAG context or graph features.

### 3. Scientific Risk & Ethical Correction
Describing a deterministic heuristic script as an "LLM without RAG" is scientifically inaccurate and misleading to academic reviewers and peer evaluators. It risks confusing algorithmic rule heuristics with actual neural language model zero-shot reasoning.

#### Rectification:
In all documentation, benchmarks, API responses, and dashboard views:
1. `Single-Pass LLM without RAG` has been formally renamed to:  
   **`Simulated Single-Pass Heuristic (No-RAG Baseline)`**
2. `LLM + RAG without Graph Tools` has been formally renamed to:  
   **`Simulated Tabular-Only Heuristic (Graph-Ablated Baseline)`**
3. Clear epistemic notices have been affixed to explain that these baselines represent simulated heuristic controls rather than live model inferences.

### 4. Summary Table of Comparative Systems

| System Name | Actual Execution Engine | RAG Used? | Graph Tools? | State Graph? | Validated Claims? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full LangGraph + Graph-RAG** | Compiled StateGraph + Mock/Live LLM | Yes (Hybrid) | Yes (2-hop) | Yes (8 nodes) | Yes (Claim & Citation) |
| **Evidence-Only Deterministic** | Python Rule Engine | No | Yes | No | Yes (Format only) |
| **No-Validation Workflow** | StateGraph without validators | Yes | Yes | Yes (Bypassed) | No (Unchecked) |
| **Simulated Single-Pass Heuristic** | Deterministic Regex Heuristic | No | Yes | No | Partial |
| **Simulated Tabular-Only Heuristic** | Deterministic Regex Heuristic | Yes | No | No | Partial |
| **Isolated Topology-Only Baseline** | Graph Feature Extractor | No | Yes | No | Partial |

