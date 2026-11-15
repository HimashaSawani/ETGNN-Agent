# Snapshot 43 Scientific Integrity Audit Report (Phase 4.5)

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 4.5 Stabilization & Scientific Integrity Audit  
**Status:** COMPLETE & VERIFIED  

---

## 1. Scientific Context & Problem Statement

The Elliptic Bitcoin Transaction Dataset comprises 203,769 transactions across 49 discrete temporal snapshots (`t = 1..49`, approximately two weeks apart). Empirical analysis in the ETGNN research codebase identified a major structural distribution shift beginning around **Snapshot 43**, where the proportion of illicit-labelled transactions exhibiting zero incoming observed edges (`in_degree = 0`) rises sharply (up to 91.5% of illicit nodes in certain late test steps).

While external literature (e.g., Weber et al., 2019; Elmougy & Liu, 2023) notes that an abrupt darknet marketplace closure occurred around the timeframe corresponding to Snapshot 43, **the anonymized transaction graph evidence within the Elliptic dataset does not establish causal attribution to any specific marketplace shutdown (such as Silk Road, AlphaBay, or Hydra)**.

During Phase 4, certain UI badges and chat explanations risked conflating external historical hypotheses with established dataset causality.

---

## 2. Canonical Defensible Formulation

Across all backend prompts, deterministic templates, report validators, chat services, and React frontend components, the following scientifically defensible formulation is enforced:

> *"An observed structural distribution shift occurs around Snapshot 43. Its underlying cause has not been established by the available transaction graph evidence."*

When historical context is discussed in follow-up Q&A (`InvestigationChatService`), it is explicitly framed as an **unverified external hypothesis** in the literature rather than established dataset causality.

---

## 3. Codebase Audit & Corrections Applied

| Component / File | Previous Wording / Risk | Corrected Scientific Wording |
| :--- | :--- | :--- |
| `src/agent/llm/provider.py` (`MockLLMProvider` & `EvidenceOnlyProvider`) | Referred to Snapshot 43 temporal regime shift with topological isolation notes. | `"Snapshot 43 Observation: An observed structural distribution shift occurs around Snapshot 43. Its underlying cause has not been established by the available transaction graph evidence."` |
| `src/agent/llm/prompts.py` (`SYSTEM_PROMPT_FORENSIC_INVESTIGATOR`) | Lacked explicit prohibition against darknet shutdown causal overclaims. | Added Directive 6: `"An observed structural distribution shift occurs around Snapshot 43. Its underlying cause has not been established by the available transaction graph evidence. NEVER assert that Snapshot 43 was caused by a specific darknet marketplace shutdown..."` |
| `src/agent/validation/report_validator.py` | Appended Snapshot 43 temporal note. | Enforces the canonical non-causal observation statement on all Snapshot 43 reports. |
| `src/agent/services/investigation_chat_service.py` | Discussed Snapshot 43 regime shift in Q&A responses. | Explicitly states: *"External literature has hypothesized that market closures may have coincided with this period, but the underlying cause has not been established by the available transaction graph evidence (unverified hypothesis, not established dataset causality)."* |
| `web/frontend/src/components/agent/InteractiveGraphEvidenceViewer.tsx` | Displayed badge `Snapshot 43 Regime Shift` and explanatory text. | Updated badge to `Snapshot 43: Observed Distribution Shift (Causality Not Established)` and updated explanatory footer to the canonical non-causal statement. |

---

## 4. Preservation of Original Research Findings

- **Zero Modification to Empirical Data:** All original experiment tables, AUPRC/F1 metrics, Stratum 2 matched-pair bootstrap audits (`N = 59` pairs), and pre/post-Snapshot 43 split metrics remain 100% untouched.
- **No Fabricated Statistics:** No new statistical claims or synthetic experiments were introduced.

---

## 5. Verification Evidence

1. **Automated Python Tests (`tests/test_phase4_5_reliability_and_integrity.py`):**
   - `test_snapshot_43_prompt_template_integrity`: Verifies `SYSTEM_PROMPT_FORENSIC_INVESTIGATOR` contains `"underlying cause has not been established"` and zero instances of `"caused by Silk Road"`, `"caused by Hydra"`, or `"darknet shutdown caused"`. — **PASSED**
   - `test_mock_provider_snapshot_43_qualification`: Verifies `MockLLMProvider` outputs the exact non-causal limitation for Snapshot 43 investigations. — **PASSED**
2. **Real-Browser Playwright E2E Verification (`web/frontend/e2e/investigation_copilot.spec.ts`):**
   - `Scenario H`: Launches an investigation on Transaction `288717023` (Snapshot 43) in Chromium and verifies that `Snapshot 43: Observed Distribution Shift` is visibly rendered in the DOM. — **PASSED (2.2s)**

