# ETGNN-Agent — Local Deployment & Demonstration Runbook

**Repository:** `D:\ETGNN-Fraud-Detection`  
**Platform Version:** 1.0.0 (Phase 6 Hardened)  
**Readiness Classification:** **`LOCAL_DEMO_READY`**  

---

### 1. Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** Python 3.10+ (Current verified: Python 3.13.3)
- **Node.js:** Node.js v18+ (Current verified: Node v22.15.1)
- **Git:** Git 2.30+

---

### 2. Quick-Start Instructions (PowerShell)

#### Step 1: Environment Setup
```powershell
# Navigate to repository root
cd D:\ETGNN-Fraud-Detection

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Create local environment configuration from template
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
```

#### Step 2: Build Frontend Assets
```powershell
# Install frontend dependencies if needed and compile SPA
cd web\frontend
npm run build
cd ..\..
```
*Verification:* Builds 1,957 modules to `web/frontend/dist` in under 10 seconds.

#### Step 3: Start the Backend & Demonstration Platform
```powershell
# Run the FastAPI server via Uvicorn
.\.venv\Scripts\python.exe -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --reload
```

#### Step 4: Open in Web Browser
Navigate to:
- **Application Dashboard:** `http://127.0.0.1:8000/`
- **Interactive API Documentation:** `http://127.0.0.1:8000/docs`
- **Health Check Endpoint:** `http://127.0.0.1:8000/api/health`

---

### 3. Verification & Diagnostic Commands

```powershell
# Run complete Python automated test suite
.\.venv\Scripts\python.exe -m pytest tests\test_phase5_evaluation_and_drift.py tests\test_phase6_security.py -v

# Run independent 16-scenario adversarial stress suite
.\.venv\Scripts\python.exe scripts\run_adversarial.py

# Verify baseline checkpoint SHA-256 hashes
.\.venv\Scripts\python.exe -c "import json, hashlib; from pathlib import Path; m = json.load(open('artifacts/baseline_manifest.json')); all_ok = all(hashlib.sha256(open(f, 'rb').read()).hexdigest() == h for f, h in m['checkpoints'].items()); print('Checkpoints Verified:', all_ok)"
```

---

### 4. Readiness Classification Definitions

| Classification | Meaning | Platform Status |
| :--- | :--- | :--- |
| **`LOCAL_DEMO_READY`** | Complete, deterministic, self-contained local execution with frozen datasets, checkpoints, and mock/evidence fallbacks. Zero external API keys needed. | **CONFIRMED (CURRENT)** |
| **`STAGING_READY`** | Requires live LLM API keys (`GEMINI_API_KEY`), real network egress, and containerized deployment (Docker/Compose). | **PENDING_API_KEYS** |
| **`PRODUCTION_READY`** | Requires multi-tenant auth, KMS key management, persistent external database, and continuous prospective monitoring. | **FUTURE_PHASE** |

