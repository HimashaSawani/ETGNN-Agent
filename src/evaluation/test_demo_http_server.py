"""
test_demo_http_server.py
========================
Tests the FastAPI demo backend over actual HTTP network requests.
Spawns the uvicorn server on localhost:8000, sends HTTP requests via `requests`,
validates status codes and response bodies, and terminates the server cleanly.
"""

import subprocess
import sys
import time
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_http_verification():
    print("=== Launching FastAPI Backend over HTTP on 127.0.0.1:8000 ===")
    server_cmd = [
        sys.executable,
        "-m", "uvicorn",
        "src.demo.backend:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--log-level", "warning"
    ]
    
    proc = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT))
    base_url = "http://127.0.0.1:8000"
    
    try:
        # 1. Wait for server readiness
        max_wait = 60
        ready = False
        start_t = time.time()
        print("Waiting for server startup and state caching...")
        while time.time() - start_t < max_wait:
            try:
                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(1)
                
        if not ready:
            raise TimeoutError("FastAPI backend failed to start within 60 seconds.")
        print(f"Backend is ready (startup time: {time.time() - start_t:.2f}s)")
        
        # 2. Test GET /health
        r_health = requests.get(f"{base_url}/health", timeout=10)
        print(f"HTTP GET /health -> Status {r_health.status_code}, Body: {r_health.json()}")
        assert r_health.status_code == 200
        assert r_health.json() == {"status": "ok", "scope": "existing Elliptic transactions"}
        
        # 3. Test GET /examples
        r_examples = requests.get(f"{base_url}/examples", timeout=10)
        print(f"HTTP GET /examples -> Status {r_examples.status_code}, Count: {len(r_examples.json())}")
        assert r_examples.status_code == 200
        examples = r_examples.json()
        assert len(examples) == 4
        print(f"  Examples received: {[e['case'] + ':' + str(e['transaction_id']) for e in examples]}")
        
        # 4. Test GET /transactions/{tx_id} for a valid example
        tx_tp = [e for e in examples if e["case"] == "TP"][0]
        tx_id = tx_tp["transaction_id"]
        r_tx = requests.get(f"{base_url}/transactions/{tx_id}", timeout=10)
        print(f"HTTP GET /transactions/{tx_id} (TP) -> Status {r_tx.status_code}")
        assert r_tx.status_code == 200
        data = r_tx.json()
        assert data["transaction_id"] == tx_id
        assert "random_forest" in data and "illicit_probability" in data["random_forest"]
        assert "etgnn" in data and "illicit_probability" in data["etgnn"]
        assert data["explanation"] is not None
        print(f"  RF p(illicit) = {data['random_forest']['illicit_probability']:.4f}, ETGNN p(illicit) = {data['etgnn']['illicit_probability']:.4f}")
        
        # 5. Test GET /transactions/{invalid_id} (expect HTTP 404)
        missing_id = 999999999999
        r_missing = requests.get(f"{base_url}/transactions/{missing_id}", timeout=10)
        print(f"HTTP GET /transactions/{missing_id} -> Status {r_missing.status_code}, Body: {r_missing.json()}")
        assert r_missing.status_code == 404
        assert "Transaction ID not in Elliptic dataset" in r_missing.json()["detail"]
        
        print("\n=== HTTP END-TO-END VERIFICATION: 100% SUCCESSFUL ===")
        return True
    finally:
        print("Terminating background uvicorn server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server terminated cleanly.")

if __name__ == "__main__":
    success = run_http_verification()
    sys.exit(0 if success else 1)
