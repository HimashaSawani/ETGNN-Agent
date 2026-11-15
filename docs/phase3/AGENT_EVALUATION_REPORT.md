# Agent Evaluation Report: Benchmark Transactions & Telemetry Latency

## 1. Evaluation Methodology

The `ETGNN-Agent` was evaluated across a diverse cohort of transactions from the Elliptic Bitcoin dataset:
1. **Verified Illicit Transactions**: e.g., `232629023` (Snapshot 1, True Illicit).
2. **Verified Licit Transactions**: e.g., `232438397` (Snapshot 1, True Licit).
3. **Unlabelled Transactions**: e.g., `230425980` (Snapshot 1, Unlabelled / Semi-supervised).
4. **Distribution-Shift Transactions**: e.g., `3233255` (Snapshot 43, Darknet Market Seizure Event).
5. **Temporal Boundary Edge Cases**: e.g., evaluating Step 43 under `as_of_snapshot = 32`.
6. **Non-Existent Transaction IDs**: e.g., `999999999999`.
7. **Adversarial Prompt Injections**: Injected malicious payloads attempting instruction override.

---

## 2. Quantitative Evaluation Results

| Transaction ID | Class / Condition | Model Consensus Verdict | Evidence Records | Telemetry Events | Execution Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`232629023`** | Illicit ($y=1$, Step 1) | `HIGH_RISK_ILLICIT` | 7 records | 18 events | `COMPLETED` |
| **`232438397`** | Licit ($y=0$, Step 1) | `LOW_RISK_LICIT` | 6 records | 16 events | `COMPLETED` |
| **`230425980`** | Unlabelled ($y=-1$, Step 1) | `SUSPICIOUS_HIGH_PROB` | 6 records | 16 events | `COMPLETED` |
| **`3233255`** | Step 43 Shift Target | `HIGH_RISK_ILLICIT` (Qualified) | 6 records | 16 events | `COMPLETED` |
| **`3233255`** | As-Of Snapshot = 32 | `INCONCLUSIVE_DATA_GAP` | 0 records | 2 events | `ABSTAINED` |
| **`999999999`** | Non-Existent ID | `INCONCLUSIVE_DATA_GAP` | 0 records | 2 events | `ABSTAINED` |
| **`232629023`** | Adversarial Injection | `HIGH_RISK_ILLICIT` | 7 records | 18 events | `COMPLETED` |

---

## 3. Telemetry & Latency Profile (CPU Baseline)

All benchmarks executed on local development environment (Windows 11, Intel Core i7, 16GB RAM, CPU-only):

| Pipeline Stage | Mean Latency (ms) | P95 Latency (ms) | Resource Utilization |
| :--- | :--- | :--- | :--- |
| **Request Validation** | 12 ms | 18 ms | Disk cache read |
| **Investigation Planning** | 4 ms | 8 ms | In-memory heuristic |
| **Deterministic Tool Execution** | 420 ms | 650 ms | PyTorch inference (3 models) |
| **Evidence Consolidation** | 2 ms | 5 ms | In-memory float comparison |
| **Explanation Generation (Mock)**| 8 ms | 15 ms | In-memory structured synthesis |
| **Claim & Report Validation** | 18 ms | 25 ms | Regex & numerical validation |
| **Total Investigation Time** | **464 ms** | **721 ms** | **Sub-second response** |

*Note: In live mode with external LLM providers (e.g. Gemini 1.5 Pro), total response latency is bounded by the network round-trip time (~1.5s to 3.0s).*

