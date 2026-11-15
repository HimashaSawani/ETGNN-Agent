import os
import sys
import time
import asyncio

sys.path.insert(0, os.path.abspath("."))

def p(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

async def test_asgi():
    p("1. Importing app...")
    from src.web.app import app
    p("1. Imported app")

    p("2. Testing direct ASGI call via httpx.AsyncClient...")
    import httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        p("Sending GET /api/health...")
        try:
            resp = await asyncio.wait_for(client.get("/api/health"), timeout=10.0)
            p(f"AsyncClient response: {resp.status_code}, {resp.json()}")
        except asyncio.TimeoutError:
            p("AsyncClient TIMED OUT after 10s!")
        except Exception as e:
            p(f"AsyncClient Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_asgi())

