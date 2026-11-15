import anyio
import time

print("Testing anyio blocking portal...", flush=True)
try:
    with anyio.from_thread.start_blocking_portal() as portal:
        print("Portal started successfully!", flush=True)
        res = portal.call(anyio.sleep, 0.01)
        print(f"Portal call succeeded: {res}", flush=True)
except Exception as e:
    print(f"Portal failed: {e}", flush=True)
print("Finished anyio test", flush=True)

