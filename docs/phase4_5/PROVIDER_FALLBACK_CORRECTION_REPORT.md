# Provider Fallback Correction & Live Execution Reliability Report (Phase 4.5)

**Project:** ETGNN-Agent — Evidence-Grounded Agentic AI for Explainable Bitcoin Transaction Fraud Detection  
**Phase:** 4.5 Stabilization & Scientific Integrity Audit  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

During the Phase 4 baseline audit, an architectural reliability issue was identified: when an unconfigured or failing live LLM provider (`gemini`, `anthropic`, `ollama`) was invoked without explicit fallback configuration, or when an unsupported provider identifier was supplied to `get_llm_provider()`, the system could either silently default to `MockLLMProvider` (`generation_mode = "SIMULATED"`) or fail to distinguish between a verified live LLM execution and an unverified configuration state.

In **Phase 4.5**, this behavior has been strictly eliminated and verified across the backend provider layer, LangGraph orchestration nodes, FastAPI HTTP transport layer, and React frontend copilot UI.

---

## 2. Enforced Provider Execution Contract

### 2.1 Mode 1: `MOCK` Mode (Development & Offline Testing Only)
- **Trigger:** Strictly when `llm_provider = "mock"` (or omitted in offline dev default).
- **Metadata Contract:**
  - `generation_mode = "SIMULATED"`
  - `live_llm_verified = False`
  - `provider_type = "mock"`
  - `provider_model_id = "deterministic_mock_v3"`
- **UI Presentation:** Amber warning badge: `SIMULATED: Deterministic Offline Mock (Zero Hallucination)`.

### 2.2 Mode 2: `LIVE` Mode (`gemini`, `anthropic`, `ollama`)
- **Trigger:** Explicitly selected by user or API caller (`llm_provider` in `{"gemini", "anthropic", "ollama"}`).
- **Strict Failure Policy:**
  - **Missing Credentials:** Raises `LLMAuthenticationError` (HTTP `401 Unauthorized`).
  - **Rate Limit (HTTP 429):** Raises `LLMRateLimitError` (HTTP `429 Too Many Requests`).
  - **Network / Upstream Timeout:** Raises `LLMTimeoutError` (HTTP `504 Gateway Timeout`).
  - **Unsupported / Unreachable Provider:** Raises `LLMProviderError` (HTTP `502 Bad Gateway`).
  - **No Silent Mock Fallback:** Under zero circumstances does a failing live provider fall back to `MockLLMProvider` (`SIMULATED`).
- **Verified Live Metadata Contract:**
  - `generation_mode = "LIVE"`
  - `live_llm_verified = True` (set strictly after `provider.generate()` and JSON validation succeed).
- **UI Presentation:**
  - If `generation_mode === "LIVE" && live_llm_verified === true`: Emerald badge `LIVE LLM: <model_id> (Verified)`.
  - If `generation_mode === "LIVE" && !live_llm_verified`: Red alert badge `LIVE LLM: <model_id> (Unverified Live Execution)`.

### 2.3 Mode 3: `DETERMINISTIC_EVIDENCE_ONLY` Fallback Mode
- **Trigger:** Either explicitly selected (`llm_provider = "evidence_only"`) OR triggered when a live provider fails AND `allow_evidence_only_fallback = True` is explicitly enabled.
- **Metadata Contract:**
  - `generation_mode = "DETERMINISTIC_EVIDENCE_ONLY"`
  - `live_llm_verified = False`
  - `provider_type = "evidence_only"`
  - `provider_model_id = "deterministic_rules_v1"`
- **Limitation Disclosure:** Appends the exact upstream exception (`LLMAuthenticationError`, `LLMTimeoutError`, etc.) to `report["limitations"]` and `report["evidence_limitations"]`.
- **UI Presentation:** Cyan badge `EVIDENCE-ONLY: Deterministic Factual Mode (Non-Generative Fallback)`.

---

## 3. Code Modifications Implemented

| File | Modification Summary |
| :--- | :--- |
| `src/agent/llm/provider.py` | Removed silent fallback to `MockLLMProvider` in `get_llm_provider()`; raises `LLMProviderError` for unknown providers and `LLMAuthenticationError` when API keys are missing. |
| `src/agent/orchestration/nodes.py` | In `generate_explanation_node`, verified live execution sets `live_llm_verified = True`; when `allow_evidence_only_fallback=False`, emits failure telemetry and re-raises typed exception; when `True`, routes strictly to `EvidenceOnlyProvider`. |
| `src/web/app.py` | Mapped `LLMAuthenticationError` -> 401, `LLMRateLimitError` -> 429, `LLMTimeoutError` -> 504, `LLMProviderError` -> 502 in `/api/agent/investigate` and `/api/agent/chat`. |
| `web/frontend/src/components/agent/AgentProviderStatus.tsx` | Added explicit check for `generationMode === 'LIVE' && !liveVerified` to render a red warning badge rather than a green verified badge. |
| `web/frontend/src/components/agent/InvestigationCopilotDrawer.tsx` | Added user-controlled `Fallback on failure` checkbox (`allowFallback` state) passed as `allow_evidence_only_fallback` in API payload. |
| `web/frontend/src/services/agentApi.ts` | Added `formatErrorMessage()` helper to parse structured FastAPI error details (`error_code`, `message`). |

---

## 4. Verification Evidence

1. **Python Unit & Integration Tests (`tests/test_phase4_5_reliability_and_integrity.py` & `tests/test_llm_reliability_and_fallbacks.py`):**
   - `test_missing_gemini_credentials_raises_authentication_error` — **PASSED**
   - `test_missing_claude_credentials_raises_authentication_error` — **PASSED**
   - `test_unsupported_provider_raises_provider_error` — **PASSED**
   - `test_explicit_mock_mode_metadata` — **PASSED**
   - `test_generate_explanation_node_strict_live_failure` — **PASSED**
   - `test_generate_explanation_node_evidence_only_fallback` — **PASSED**
   - `test_live_provider_timeout_raises_llm_timeout_error` — **PASSED**
   - `test_live_provider_rate_limit_raises_llm_rate_limit_error` — **PASSED**
2. **Real-Browser Playwright E2E Verification (`web/frontend/e2e/investigation_copilot.spec.ts`):**
   - `Scenario C`: Verifies MOCK mode renders `SIMULATED: Deterministic Offline Mock`. — **PASSED (2.7s)**
   - `Scenario D`: Verifies EVIDENCE_ONLY mode renders `EVIDENCE-ONLY: Deterministic Factual Mode` and never displays `LIVE LLM`. — **PASSED (3.3s)**
   - `Scenario E`: Verifies selecting unconfigured `gemini` with fallback unchecked displays `Gemini API key not found in GEMINI_API_KEY environment variable` and never substitutes a simulated report. — **PASSED (3.0s)**

---

## 5. Unresolved Limitations

- Live API execution against Google Gemini (`gemini-2.0-flash`) and Anthropic Claude (`claude-3-5-sonnet-20241022`) requires external user-supplied API keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) and outbound internet connectivity. In offline/sandboxed environments without credentials, the system now deterministically rejects live requests with `401 LLMAuthenticationError` (or routes to `DETERMINISTIC_EVIDENCE_ONLY` when fallback is enabled).

