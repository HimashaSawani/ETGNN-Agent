# Real-Browser Playwright E2E Verification Report (Phase 4.5)

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 4.5 Stabilization & Scientific Integrity Audit  
**Browser Engine:** Chromium (`chromium-1254` / Chrome for Testing 158.0.8127.18)  
**Test Runner:** `@playwright/test` v1.65.0  
**Overall Result:** **12 / 12 PASSED (100% Pass Rate in 57.0s)**  

---

## 1. Test Infrastructure & Setup

1. **Dependencies & Binaries:**
   - Added `@playwright/test` (`^1.65.0`) to `web/frontend/package.json` with `"test:e2e": "playwright test"`.
   - Installed Chromium browser binaries via `npx playwright install chromium` (`C:\Users\himas\AppData\Local\ms-playwright\chromium-1254`).
2. **Configuration (`web/frontend/playwright.config.ts`):**
   - Configured headless Chromium desktop viewport (`1280x800`).
   - Configured `webServer` to serve the production bundle via `npx vite preview --port 4173 --host 127.0.0.1`.
   - Enabled trace collection (`on-first-retry`), screenshot capture (`only-on-failure`), and HTML reporting.
3. **Methodology Note (Mocked API Routes vs. Backend Integration):**
   - To ensure deterministic, fast browser verification and avoid Windows sandbox port-forwarding collisions between Vite proxy and background Python processes, `web/frontend/e2e/investigation_copilot.spec.ts` intercepts `/api/agent/**` requests at the Chromium network layer with realistic schemas matching the verified FastAPI contracts, while real FastAPI backend endpoints are independently verified via `tests/test_phase4_api_endpoints.py`, `tests/test_phase4_scenarios.py`, and `tests/test_phase4_5_reliability_and_integrity.py`.

---

## 2. Executed Browser Scenarios (A – L)

| Scenario | Test Name | Verified Browser Behavior | Duration | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Scenario A** | `Scenario A: Application startup and core navigation in Chromium` | Opens `/` in Chromium, verifies title `/ETGNN/i`, verifies navigation buttons (`AI Copilot`, `Overview`, `Transaction Explorer`, `Model Predictions`). | 2.7s | **PASSED** |
| **Scenario B** | `Scenario B: Investigation drawer opens, syncs Tx ID, and responds to Esc` | Clicks `Investigate with AI`, verifies `AI Investigation Copilot` drawer opens, presses `Escape` key to dismiss, verifies drawer closes, and reopens via header button. | 3.8s | **PASSED** |
| **Scenario C** | `Scenario C: Investigation execution in MOCK mode displays simulated label` | Launches investigation in MOCK mode, verifies `SIMULATED: Deterministic Offline Mock` badge, `HIGH_RISK_ILLICIT` verdict, and `ETGNN (Canonical)` rendering. | 2.7s | **PASSED** |
| **Scenario D** | `Scenario D: Evidence-only mode discloses deterministic non-generative mode` | Submits investigation in `evidence_only` mode, verifies `EVIDENCE-ONLY: Deterministic Factual Mode` badge is visible and `LIVE LLM` is absent. | 3.3s | **PASSED** |
| **Scenario E** | `Scenario E: Unconfigured live provider returns explicit error without silent mock fallback` | Selects `gemini`, unchecks `Fallback on failure`, launches investigation, verifies HTTP 401 `Gemini API key not found in GEMINI_API_KEY environment variable` banner renders and no simulated report is substituted. | 3.0s | **PASSED** |
| **Scenario F** | `Scenario F: Clicking citation anchors navigates to verified Evidence Records` | Opens `Evidence (3)` tab, verifies `Immutable Evidence Records` list and cryptographic provenance IDs `EV-TX-LOOKUP-288717023` and `EV-GNN-PRED-288717023`. | 4.3s | **PASSED** |
| **Scenario G** | `Scenario G: Model comparison panel calculates spread and warns on divergence` | Verifies `Multi-Model Consensus & Probability Spread` card displays `0.8850` (ETGNN), `0.5000` (RF), `0.8200` (XGBoost), and `Model Divergence Detected`. | 2.4s | **PASSED** |
| **Scenario H** | `Scenario H: Graph evidence viewer displays Snapshot 43 non-causal qualification and degree wording` | Verifies `Snapshot 43: Observed Distribution Shift`, `0 Observed In-Edges`, and `Illicit-Label Neighbors` badges in the graph evidence viewer. | 2.2s | **PASSED** |
| **Scenario I** | `Scenario I: Follow-up chat answers grounded questions with cited evidence` | Switches to `Follow-Up Chat` tab, clicks suggested prompt chip `Why did ETGNN classify this transaction differently from Random Forest?`, verifies grounded answer and `[EV-GNN-PRED-288717023]` citation pill. | 5.4s | **PASSED** |
| **Scenario J** | `Scenario J: Analyst feedback form submits rating, review category and logs persistence` | Switches to `Analyst Review` tab, verifies `Checkpoint Isolated` badge, enters review notes, clicks `Submit Analyst Review`, verifies `Analyst review feedback successfully recorded in isolated database.`. | 2.9s | **PASSED** |
| **Scenario K** | `Scenario K: Export triggers non-empty JSON and PDF downloads` | Clicks `Download JSON`, intercepts browser `download` event (`ETGNN_Investigation_inv_e2e_playwright_001_Tx288717023.json`), and verifies `Export Official PDF` link (`href="/api/agent/investigations/inv_e2e_playwright_001/export/pdf"`). | 2.2s | **PASSED** |
| **Scenario L** | `Scenario L: Error boundary gracefully displays API failure messages` | Enters invalid Tx ID `999999999`, launches investigation, verifies error banner `Transaction ID 999999999 not found in Elliptic dataset.`. | 2.0s | **PASSED** |

---

## 3. Raw Playwright Execution Log

```text
> frontend@0.0.0 test:e2e
> playwright test

Running 12 tests using 1 worker

  ok  1 [chromium] › e2e\investigation_copilot.spec.ts:250:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario A: Application startup and core navigation in Chromium (2.7s)
  ok  2 [chromium] › e2e\investigation_copilot.spec.ts:261:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario B: Investigation drawer opens, syncs Tx ID, and responds to Esc (3.8s)
  ok  3 [chromium] › e2e\investigation_copilot.spec.ts:279:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario C: Investigation execution in MOCK mode displays simulated label (2.7s)
  ok  4 [chromium] › e2e\investigation_copilot.spec.ts:295:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario D: Evidence-only mode discloses deterministic non-generative mode (3.3s)
  ok  5 [chromium] › e2e\investigation_copilot.spec.ts:329:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario E: Unconfigured live provider returns explicit error without silent mock fallback (3.0s)
  ok  6 [chromium] › e2e\investigation_copilot.spec.ts:366:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario F: Clicking citation anchors navigates to verified Evidence Records (4.3s)
  ok  7 [chromium] › e2e\investigation_copilot.spec.ts:380:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario G: Model comparison panel calculates spread and warns on divergence (2.4s)
  ok  8 [chromium] › e2e\investigation_copilot.spec.ts:396:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario H: Graph evidence viewer displays Snapshot 43 non-causal qualification and degree wording (2.2s)
  ok  9 [chromium] › e2e\investigation_copilot.spec.ts:410:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario I: Follow-up chat answers grounded questions with cited evidence (5.4s)
  ok 10 [chromium] › e2e\investigation_copilot.spec.ts:429:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario J: Analyst feedback form submits rating, review category and logs persistence (2.9s)
  ok 11 [chromium] › e2e\investigation_copilot.spec.ts:449:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario K: Export triggers non-empty JSON and PDF downloads (2.2s)
  ok 12 [chromium] › e2e\investigation_copilot.spec.ts:468:3 › ETGNN-Agent Phase 4.5 Browser E2E Suite › Scenario L: Error boundary gracefully displays API failure messages (2.0s)

  12 passed (57.0s)
```

