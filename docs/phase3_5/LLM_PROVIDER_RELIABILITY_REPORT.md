# ETGNN-Agent — Phase 3.5: LLM Provider Reliability & Fallback Architecture Report

**Document ID:** `ETGNN-REP-P3.5-002`  
**Date:** November 9, 2026  
**Status:** COMPLETE & VERIFIED  
**Author:** Principal Agentic AI Engineer, LLM Reliability Engineer, Systems Architect  
**Repository:** `D:\ETGNN-Fraud-Detection`  
**Runtime:** Python 3.13.3 (Windows x64)

---

## 1. Executive Summary

Previous agent iterations exhibited a critical scientific ambiguity: external LLM calls that failed or lacked credentials could silently fall back to synthetic mock generators, blurring the distinction between actual model inferences and deterministic simulations.

In **Phase 3.5**, a strict **3-Mode Provider Architecture** and a **Typed Exception Hierarchy** were implemented in `src/agent/llm/provider.py`. The system guarantees total epistemic honesty:
1. Simulated outputs are permanently tagged with `generation_mode="SIMULATED"` and `live_llm_verified=false`.
2. Deterministic factual summaries without LLM generation are permanently tagged with `generation_mode="DETERMINISTIC_EVIDENCE_ONLY"`.
3. Live generative calls are tagged with `generation_mode="LIVE"` and `live_llm_verified=true`.
4. **Zero Silent Fallbacks:** If a live provider fails or lacks credentials, the system never silently mimics an LLM. It either raises an explicit typed error or gracefully degrades to `EvidenceOnlyProvider` with an explicit notice appended to `evidence_limitations`.

---

## 2. The 3-Mode Provider Architecture

```
                          ┌──────────────────────────┐
                          │   get_llm_provider()     │
                          └─────────────┬────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  ┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
  │  MockLLMProvider  │        │EvidenceOnlyProvider│       │   LiveProvider    │
  │                   │        │                   │        │(Gemini, Claude,   │
  │ • Mode: SIMULATED │        │ • Mode: EVIDENCE  │        │ Ollama)           │
  │ • live_llm: False │        │ • live_llm: False │        │ • Mode: LIVE      │
  │ • Deterministic   │        │ • Non-generative  │        │ • live_llm: True  │
  │   mock template   │        │   factual summary │        │ • Strict typed err│
  └───────────────────┘        └───────────────────┘        └─────────┬─────────┘
                                                                      │ (On Failure)
                                                      ┌───────────────┴───────────────┐
                                                      ▼                               ▼
                                             [allow_fallback=True]          [allow_fallback=False]
                                                      │                               │
                                            Degrades explicitly to          Raises typed error:
                                            EvidenceOnlyProvider            • LLMAuthenticationError
                                            + evidence_limitations          • LLMTimeoutError
                                              annotation                    • LLMRateLimitError
```

### Detailed Mode Specifications:

| Property | Mode 1: Mock Provider | Mode 2: Evidence-Only Provider | Mode 3: Live Provider |
|---|---|---|---|
| **Class** | `MockLLMProvider` | `EvidenceOnlyProvider` | `GeminiProvider` / `AnthropicProvider` / `OllamaProvider` |
| **`generation_mode`** | `"SIMULATED"` | `"DETERMINISTIC_EVIDENCE_ONLY"` | `"LIVE"` |
| **`live_llm_verified`** | `False` | `False` | `True` |
| **External Network** | None (Offline) | None (Offline) | HTTPS REST API / Local Socket |
| **Generative Risk** | Zero (Fixed logic) | Zero (Non-generative) | Monitored by citation & claim validators |
| **Use Case** | Local unit tests, mock demos | Safe production fallbacks, air-gapped runtimes | Live analyst interactive investigations |

---

## 3. Typed Error Hierarchy

All LLM failure modes are mapped to explicit, typed exceptions inheriting from `LLMProviderError`:

```python
class LLMProviderError(Exception):
    """Base exception for all LLM inference errors."""
    pass

class LLMTimeoutError(LLMProviderError):
    """Raised when an external LLM provider call exceeds deadline."""
    pass

class LLMRateLimitError(LLMProviderError):
    """Raised when an external provider returns HTTP 429 Too Many Requests."""
    pass

class LLMAuthenticationError(LLMProviderError):
    """Raised when API credentials are missing, rejected, or expired."""
    pass

class LLMOutputFormatError(LLMProviderError):
    """Raised when provider returns output that cannot be parsed into target schema."""
    pass
```

### Elimination of Silent Fallback:
In `src/agent/orchestration/nodes.py` (`generate_explanation_node`):

```python
    try:
        raw_text = provider.generate(prompt, SYSTEM_PROMPT_FORENSIC_INVESTIGATOR, context_data=context_data)
        report_dict = parse_and_validate_json(raw_text)
    except Exception as e:
        if isinstance(provider, MockLLMProvider):
            raise
        if allow_evidence_fallback:
            fallback = EvidenceOnlyProvider()
            raw_text = fallback.generate(prompt, SYSTEM_PROMPT_FORENSIC_INVESTIGATOR, context_data=context_data)
            report_dict = parse_and_validate_json(raw_text)
            lims = report_dict.get("evidence_limitations", [])
            lims.append(
                f"Live LLM provider '{provider.provider_name}' failed ({type(e).__name__}: {str(e)}); "
                f"fallen back to deterministic evidence-only mode."
            )
            report_dict["evidence_limitations"] = lims
            report_dict["limitations"] = lims
        else:
            _emit_event(
                inv_id,
                InvestigationEventType.EXPLANATION_GENERATED,
                status=ExecutionStatus.FAILED,
                error_code=type(e).__name__,
                message=f"LLM provider failed: {str(e)}",
            )
            raise
```

---

## 4. Live Provider Test & Configuration Audit

In accordance with strict research integrity policies, all live providers in local test runs without credentials are formally marked **NOT TESTED / UNCONFIGURED**, rather than asserting false passes.

| Provider | Model Identifier | Default Timeout | Environment Variable | Status in Local Sandbox |
|---|---|---|---|---|
| **Deterministic Mock** | `deterministic_mock_v3` | 0.0s | N/A | **PASS** (100% offline) |
| **Evidence-Only** | `deterministic_rules_v1` | 0.0s | N/A | **PASS** (100% offline) |
| **Google Gemini** | `gemini-1.5-pro` | 30.0s | `GEMINI_API_KEY` | **NOT CONFIGURED** (Verified raises `LLMAuthenticationError`) |
| **Anthropic Claude** | `claude-3-5-sonnet-20241022` | 30.0s | `ANTHROPIC_API_KEY` | **NOT CONFIGURED** (Verified raises `LLMAuthenticationError`) |
| **Local Ollama** | `llama3:8b-instruct` | 60.0s | `OLLAMA_ENDPOINT` | **NOT CONFIGURED** (Requires external daemon) |

---

## 5. Verification Test Suite

The test suite in `tests/test_llm_reliability_and_fallbacks.py` verifies:
1. `test_mock_provider_metadata_and_simulated_mode`: Asserts `generation_mode="SIMULATED"`, `live_llm_verified=False`.
2. `test_evidence_only_provider_factual_mode`: Asserts `generation_mode="DETERMINISTIC_EVIDENCE_ONLY"`, `live_llm_verified=False`.
3. `test_live_provider_missing_credentials_raises_typed_error`: Confirms Gemini and Anthropic raise `LLMAuthenticationError` when keys are omitted.
4. `test_live_provider_timeout_raises_llm_timeout_error`: Confirms socket timeouts raise `LLMTimeoutError`.
5. `test_live_provider_rate_limit_raises_llm_rate_limit_error`: Confirms HTTP 429 status codes raise `LLMRateLimitError`.
6. `test_runner_fallback_to_evidence_only_when_allowed`: Asserts runner gracefully degrades to `EvidenceOnlyProvider` with limitation notices when live API fails.
7. `test_runner_raises_when_fallback_disabled`: Asserts runner raises typed error without falling back when fallback is disabled.

