# ETGNN Web Platform Setup & Execution Guide

This document describes how to install, configure, test, and run the ETGNN Research Demonstration Platform locally on Windows, Linux, or macOS.

---

## 1. Prerequisites

- **Python**: 3.10 to 3.13
- **Node.js**: v18.0.0 or higher (v22.x recommended)
- **Git**: Installed and available in PATH
- **Disk Space**: ~2 GB (includes repository artifacts and Elliptic tensors)
- **RAM**: Minimum 4 GB (8 GB recommended for memory-mapped feature arrays)
- **GPU**: Optional (CPU execution is fully supported and optimized)

---

## 2. Environment Installation

### 2.1. Backend Python Virtual Environment

From the project root:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install httpx pytest
```

```bash
# Linux / macOS Bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install httpx pytest
```

### 2.2. Frontend Dependencies Installation & Build

```powershell
# Navigate to web/frontend
cd web/frontend
npm install
npm run build
cd ../..
```

The production bundle will be built into `web/frontend/dist/`. The FastAPI backend automatically mounts this directory to serve the frontend single-page application at the root URL.

---

## 3. Running the Platform

### Option A: Unified Production Mode (Recommended)

Start the FastAPI application. It serves both the backend API (`/api/*`) and the compiled React frontend (`/`):

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

```bash
# Linux / macOS
source .venv/bin/activate
uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

### Option B: Development Mode (Vite Hot-Reload)

If developing the frontend with hot module replacement (HMR):

1. **Start Backend API Server** (Terminal 1):
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Start Vite Dev Server** (Terminal 2):
   ```powershell
   cd web/frontend
   npm run dev
   ```
   Open `http://localhost:5173/`. Vite automatically proxies `/api` calls to `http://127.0.0.1:8000`.

---

## 4. Running Automated Tests

To run the automated backend test suite covering all 11 API endpoints, checkpoint loading, artifact parity, and SPA serving:

```powershell
# Windows PowerShell
.\.venv\Scripts\pytest.exe tests/test_web_api.py -v
```

```bash
# Linux / macOS
pytest tests/test_web_api.py -v
```

Expected output:
```
============================= 13 passed in ~10s =============================
```

To validate the frontend build:
```powershell
npm --prefix web/frontend run build
```

---

## 5. API Verification via Curl

Test that the backend server is responding:

```powershell
# Health Check
curl http://127.0.0.1:8000/api/health

# Benchmark Metrics
curl http://127.0.0.1:8000/api/metrics

# Query Single Transaction (Connected TP Exemplar)
curl http://127.0.0.1:8000/api/transactions/288717023

# Extract 2-Hop Subgraph
curl http://127.0.0.1:8000/api/transactions/288717023/subgraph?hops=2
```
