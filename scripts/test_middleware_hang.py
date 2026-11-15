import time
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import MutableHeaders

print("Testing Hypothesis: BaseHTTPMiddleware deadlock...", flush=True)

# Test 1: Minimal app without middleware
app1 = FastAPI()
@app1.get("/ping")
def ping():
    return {"status": "ok"}

c1 = TestClient(app1)
t0 = time.time()
r1 = c1.get("/ping")
print(f"Test 1 (no middleware): {r1.status_code} in {time.time()-t0:.4f}s", flush=True)

# Test 2: App with app.middleware("http")
app2 = FastAPI()
@app2.middleware("http")
async def sec_mid(request, call_next):
    resp = await call_next(request)
    resp.headers["X-Frame-Options"] = "DENY"
    return resp

@app2.get("/ping")
def ping2():
    return {"status": "ok"}

c2 = TestClient(app2)
t0 = time.time()
try:
    r2 = c2.get("/ping")
    print(f"Test 2 (BaseHTTPMiddleware): {r2.status_code} in {time.time()-t0:.4f}s", flush=True)
except Exception as e:
    print(f"Test 2 failed: {e}", flush=True)

