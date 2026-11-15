import time
import sys
import os

sys.path.insert(0, os.path.abspath("."))

def debug_step(name, fn):
    t0 = time.time()
    print(f"Starting {name} ...", flush=True)
    res = fn()
    print(f"Finished {name} in {time.time() - t0:.2f}s", flush=True)
    return res

debug_step("import fastapi", lambda: __import__("fastapi"))
debug_step("import torch", lambda: __import__("torch"))
debug_step("import src.web.app", lambda: __import__("src.web.app"))
from src.web.app import app
print("App object ready:", app.title, flush=True)
from src.web.data_loader import get_data_loader
debug_step("get_data_loader()", lambda: get_data_loader())
print("DataLoader initialized successfully!", flush=True)
