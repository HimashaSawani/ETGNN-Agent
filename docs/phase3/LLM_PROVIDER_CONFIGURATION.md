# LLM Provider Configuration & Multi-Backend Architecture

## 1. Overview

The `ETGNN-Agent` system decouples core graph reasoning and evidence validation from specific LLM vendors. The provider subsystem in `src/agent/llm/provider.py` supports four distinct execution environments:
1. `MockLLMProvider` (Default / CI / Offline)
2. `GeminiProvider` (Production Cloud)
3. `AnthropicProvider` (Claude Messages API)
4. `OllamaProvider` (Self-Hosted Local Inference)

Provider selection is governed by the environment variable `ETGNN_LLM_PROVIDER`. If unspecified or if authentication credentials are not found, the system defaults cleanly to `MockLLMProvider` without runtime crashes.

---

## 2. Supported Providers

### 2.1 Mock Provider (`mock`)
- **Use Case**: Unit tests, integration test suites, air-gapped CI/CD, deterministic demonstrations.
- **Characteristics**: Zero latency, zero external API costs, 100% reproducible fact generation.
- **Behavior**: Inspects `evidence_bundle` and `provenance_chain` directly, synthesizing valid forensic reports referencing strictly verified evidence IDs.

### 2.2 Google Gemini (`gemini`)
- **Use Case**: Primary cloud production deployment.
- **Recommended Model**: `gemini-1.5-pro` or `gemini-2.0-flash`.
- **Environment Variables**:
  - `ETGNN_LLM_PROVIDER=gemini`
  - `GEMINI_API_KEY=<your-google-api-key>`
  - `GEMINI_MODEL=gemini-1.5-pro` (Optional, default: `gemini-1.5-pro`)
- **Features**: Automatic markdown fence stripping (` ```json ... ``` `) and schema repair.

### 2.3 Anthropic Claude (`anthropic`)
- **Use Case**: Alternative cloud production deployment.
- **Recommended Model**: `claude-3-5-sonnet-20241022` or `claude-3-opus-20240229`.
- **Environment Variables**:
  - `ETGNN_LLM_PROVIDER=anthropic`
  - `ANTHROPIC_API_KEY=<your-anthropic-key>`
  - `ANTHROPIC_MODEL=claude-3-5-sonnet-20241022` (Optional)

### 2.4 Ollama Local Server (`ollama`)
- **Use Case**: Air-gapped deployments, privacy-sensitive financial enterprise environments, zero-cost local research.
- **Recommended Model**: `qwen2.5:7b-instruct` or `llama3.1:8b-instruct`.
- **Environment Variables**:
  - `ETGNN_LLM_PROVIDER=ollama`
  - `OLLAMA_BASE_URL=http://localhost:11434` (Optional, default: `http://localhost:11434`)
  - `OLLAMA_MODEL=qwen2.5:7b` (Optional, default: `qwen2.5:7b`)

---

## 3. Configuration & Factory Resolution

```python
from src.agent.llm.provider import get_llm_provider

# Automatically resolves based on ETGNN_LLM_PROVIDER
provider = get_llm_provider()

# Invoke text generation
response = provider.generate(
    prompt="Investigate transaction 232629023...",
    system_prompt="You are a senior forensic investigator...",
    context_data={...}
)
```

### Fallback Guarantee
If an external API call to Gemini, Anthropic, or Ollama times out (default timeout: 30.0s) or fails due to network/rate-limit errors, `generate_explanation_node` catches the exception and immediately invokes `MockLLMProvider` as a safety net, ensuring investigations complete without system failure.

