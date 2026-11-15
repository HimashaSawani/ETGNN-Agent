import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

def p(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

p("1. Importing app...")
t0 = time.time()
from src.web.app import app, health
p(f"1 done in {time.time()-t0:.2f}s")

p("2. Calling health() directly...")
t0 = time.time()
res = health()
p(f"2 done in {time.time()-t0:.2f}s: status={res['status']}, system={res['system']}")

p("3. Importing TestClient...")
t0 = time.time()
from fastapi.testclient import TestClient
p(f"3 done in {time.time()-t0:.2f}s")

p("4. Creating TestClient(app)...")
t0 = time.time()
client = TestClient(app)
p(f"4 done in {time.time()-t0:.2f}s")

p("5. Calling client.get('/api/health')...")
t0 = time.time()
resp = client.get("/api/health")
p(f"5 done in {time.time()-t0:.2f}s: status_code={resp.status_code}, json={resp.json()}")

p("SUCCESS!")
