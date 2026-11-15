# ETGNN Web Platform Security & Reliability Audit Report

## 1. Security Architecture & Threat Model

The ETGNN Web Platform is an academic research demonstration designed strictly for local or air-gapped evaluation. Because financial and machine learning demonstration platforms are exposed to adversarial inputs, the system was hardened against common web application vulnerabilities (OWASP Top 10) and ML-specific attack vectors.

---

## 2. Security Controls & Defenses Implemented

### 2.1. Path Traversal & File Disclosure Defense
- **Vulnerability Surface**: The SPA static file handler serves files from `web/frontend/dist/`. Unsanitized path inputs could theoretically allow attackers to traverse out of the distribution directory.
- **Defense Implemented**:
  ```python
  @app.get("/{full_path:path}")
  def serve_spa(full_path: str):
      if ".." in full_path or "%2e%2e" in full_path.lower():
          raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
      file_path = (DIST_DIR / full_path).resolve()
      try:
          file_path.relative_to(DIST_DIR.resolve())
      except ValueError:
          raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
  ```
- **Automated Verification**: Requests targeting `/..%2f..%2fsecret.txt` are rejected with HTTP 403 Forbidden.

### 2.2. Strict Input Validation & Type Safety
- **Vulnerability Surface**: Malformed, negative, or excessively large transaction IDs, or non-numeric feature overrides.
- **Defense Implemented**:
  - `transaction_id`: Pydantic field constraint `ge=0, le=2**63 - 1`. Negative IDs return HTTP 422 Unprocessable Entity.
  - `features_override`: Enforces maximum dictionary length $\le 165$ dimensions. Every value is validated via `math.isfinite(val)` to reject `NaN`, `Infinity`, and string injections.
- **Automated Verification**: Verified via unit tests in `tests/test_web_api.py`.

### 2.3. Safe Model Deserialization
- **Vulnerability Surface**: Arbitrary code execution via unsafe PyTorch / Python pickle deserialization.
- **Defense Implemented**:
  - PyTorch checkpoints (`etgnn.pt`, `matched_gat.pt`, `test_graph.pt`) are loaded with `weights_only=True` and `map_location="cpu"`.
  - Checkpoint paths are hardcoded to local repository artifacts (`results/checkpoints/`) and never accept user-supplied filenames.

### 2.4. Cross-Origin Resource Sharing (CORS) Hardening
- **Initial Policy**: `allow_origins=["*"]` (Permissive).
- **Hardened Policy**: Restricted exclusively to local loopback origins:
  ```python
  LOCAL_ORIGINS = [
      "http://127.0.0.1:8000",
      "http://localhost:8000",
      "http://127.0.0.1:5173",
      "http://localhost:5173",
  ]
  ```
- Prevents cross-site request forgery and unauthorized browser-based scraping from external websites.

### 2.5. Resource Exhaustion & CPU Denial of Service
- **Thread Capping**: Set `torch.set_num_threads(2)` on process startup to prevent GNN forward passes from starving the host operating system.
- **Memory Mapping**: Features are memory-mapped (`numpy.load(..., mmap_mode="r")`), maintaining a constant resident memory footprint without loading the 134 MB array into RAM repeatedly.
- **Graph Neighborhood Limits**: Subgraph extraction restricts dense neighborhoods to $k \le 2$ hops and $N \le 150$ nodes, preventing client-side canvas lockups.

---

## 3. Automated Security Test Results

From `tests/test_web_api.py`:

| Security Test Case | Payload | Expected Response | Observed Response | Status |
|---|---|---|---|---|
| Negative Transaction ID | `{"transaction_id": -5}` | `422 Unprocessable` | `422 Unprocessable` | **PASS** |
| Non-Numeric Feature Override | `{"features_override": {"feat_0": "bad"}}` | `422 Unprocessable` | `422 Unprocessable` | **PASS** |
| Oversized Override (>165 keys) | `{"features_override": {feat_0..feat_199}}` | `422 Unprocessable` | `422 Unprocessable` | **PASS** |
| Path Traversal Attack | `GET /..%2f..%2fsecret.txt` | `403 Forbidden` | `403 Forbidden` | **PASS** |
| Invalid Transaction Lookup | `GET /api/transactions/999999999999` | `404 Not Found` | `404 Not Found` | **PASS** |

---

## 4. Security Recommendations for Viva Demonstration

1. Run the uvicorn server bound strictly to `127.0.0.1` (do not use `--host 0.0.0.0` on untrusted networks).
2. Keep the backend in production mode without `--reload` during formal evaluation.
