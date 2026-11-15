# Agent Evaluation Methodology
## Modular Evaluation Framework for LangGraph Bitcoin Fraud Investigation

**Project:** ETGNN-Agent  
**Document:** `docs/phase5/AGENT_EVALUATION_METHODOLOGY.md`  
**Evaluation Scope:** Factual Grounding, Tool Execution, Retrieval Performance, Temporal Leakage Prevention, and System Latency

---

### 1. Methodology Overview

The Phase 5 agent evaluation framework provides reproducible benchmarking of the 11-stage LangGraph forensic investigation workflow. Rather than relying on qualitative impressions or single-pass LLM self-evaluations, the framework establishes explicit mathematical metrics, predefined denominators, and versioned transaction fixtures.

```
+-----------------------------------------------------------------------------+
|                      PHASE 5 AGENT EVALUATION SUITE                         |
+-----------------------------------------------------------------------------+
|  [Datasets]                                                                 |
|   - Synthetic Fixtures (Known degrees, controlled probabilities)            |
|   - Real Development Cohorts (Connected Illicit, High-Risk, Licit Baseline) |
|   - Temporal Leakage Probes (Future queries, Boundary attacks)              |
|   - Adversarial Traps (Prompt injections, Unknown-label assumptions)        |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|  [Metric Evaluators]                                                        |
|   - GroundingMetricsEvaluator (Citation precision, Numerical accuracy)      |
|   - ToolMetricsEvaluator (Selection accuracy, Budget compliance, Retries)   |
|   - RetrievalMetricsEvaluator (Recall@K, Precision@K, MRR, Coverage)        |
|   - TemporalMetricsEvaluator (Boundary enforcement, Label masking)          |
|   - LatencyMetricsEvaluator (Mean, Median, P95, Failure rates)              |
|   - SemanticCitationEvaluator (4-class support classification)              |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|  [Output Artifacts]                                                         |
|   - results/benchmarks/phase5_agent_ablation_results.json                   |
|   - Telemetry Event Logs via event_sink                                     |
+-----------------------------------------------------------------------------+
```

---

### 2. Metric Formulations and Denominators

Every metric is defined with an explicit denominator to prevent distorted percentages:

#### A. Grounding Metrics (`src/agent/evaluation/grounding_metrics.py`)
- **Citation Existence Rate:**  
  $$\text{Existence Rate} = \frac{N_{\text{reports with } \ge 1 \text{ citation}}}{N_{\text{total evaluated reports}}}$$
- **Citation Precision:**  
  $$\text{Citation Precision} = \frac{\sum \text{Valid Citation Identifiers Resolved}}{\sum \text{Total Citation Identifiers Cited}}$$
- **Citation Coverage:**  
  $$\text{Citation Coverage} = \frac{N_{\text{grounded numerical claims}} + \mathbb{I}(\text{has citations})}{N_{\text{expected numerical facts}} + 1}$$
- **Numerical Correctness Rate:**  
  $$\text{Numerical Correctness} = \frac{\sum \text{Matched Numbers within Tolerance}}{\sum \text{Expected Numerical Values}}$$
  *Tolerance:* $\pm 0.015$ for probabilities; exact integers for graph degrees.
- **Unsupported Claim Rate:**  
  $$\text{Unsupported Rate} = \frac{N_{\text{invalid citations}} + N_{\text{hallucinated models}}}{N_{\text{total claims asserted}}}$$

#### B. Tool and Workflow Metrics (`src/agent/evaluation/tool_metrics.py`)
- **Tool Selection Accuracy:** Jaccard overlap between executed tools and expected tools.
- **Tool Budget Compliance:** Fraction of investigations finishing within $\le 8$ tool invocations.
- **Correct Abstention Rate:**  
  $$\text{Abstention Rate} = \frac{N_{\text{correctly abstained}}}{N_{\text{tasks requiring intentional abstention}}}$$

#### C. Retrieval Metrics (`src/agent/evaluation/retrieval_metrics.py`)
- **Mean Reciprocal Rank (MRR):**  
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **Recall@K (K=5):** Fraction of relevant evidence documents present in the top 5 retrieved items.

#### D. Latency and Reliability Metrics (`src/agent/evaluation/latency_metrics.py`)
- **Mean, Median, and 95th Percentile Latency:** Measured in milliseconds across all workflow invocations.
- **Timeout Rate:** Invocations exceeding 30,000 ms.
- **Provider Error Rate:** LLM API network errors or HTTP 5xx responses.

---

### 3. Separation of Synthetic and Real Evaluation Data

To maintain scientific integrity:
1. **Synthetic fixtures** (e.g. `999000001`, `999000002`) verify tool limits and graph traversal logic without risking data leakage.
2. **Real transactions** (e.g. `288717023`, `232629023`, `230425980`) evaluate multi-model consensus and GNNExplainer interpretation.
3. **Adversarial fixtures** probe prompt injections and test label preservation.

