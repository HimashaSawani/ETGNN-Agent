import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting diagnosis script")

t0 = time.time()
log("Step 1: Testing raw torch import...")
import torch
log(f"Step 1 complete in {time.time()-t0:.2f}s (torch {torch.__version__})")

t0 = time.time()
log("Step 2: Testing data_loader import...")
from src.web.data_loader import get_data_loader
log(f"Step 2 complete in {time.time()-t0:.2f}s")

t0 = time.time()
log("Step 3: Instantiating get_data_loader()...")
loader = get_data_loader()
log(f"Step 3 complete in {time.time()-t0:.2f}s, loaded {len(loader.classes_df)} transactions")

t0 = time.time()
log("Step 4: Calling health() function directly (bypassing HTTP/TestClient)...")
from src.web.app import health
h = health()
log(f"Step 4 complete in {time.time()-t0:.2f}s: {h['status']}")

t0 = time.time()
log("Step 5: Testing starlette / httpx / anyio blocking...")
import httpx
import anyio
log(f"Step 5 httpx/anyio imported in {time.time()-t0:.2f}s")

t0 = time.time()
log("Step 6: Creating TestClient on app...")
from fastapi.testclient import TestClient
from src.web.app import app
client = TestClient(app)
log(f"Step 6 TestClient created in {time.time()-t0:.2f}s")

t0 = time.time()
log("Step 7: Making client.get('/api/health')...")
resp = client.get("/api/health")
log(f"Step 7 response received in {time.time()-t0:.2f}s, status: {resp.status_code}")
log(f"Response body: {resp.json()}")

log("ALL DIAGNOSTIC STEPS COMPLETED SUCCESSFULLY!")
