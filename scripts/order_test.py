import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

def mem():
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return f"AvailPhys: {s.ullAvailPhys/(1024**2):.1f}MB, AvailPage: {s.ullAvailPageFile/(1024**2):.1f}MB"

print(f"[0s] Start: {mem()}", flush=True)

t0 = time.time()
import pandas as pd
print(f"[{time.time()-t0:.2f}s] pandas imported: {mem()}", flush=True)

t0 = time.time()
import numpy as np
print(f"[{time.time()-t0:.2f}s] numpy imported: {mem()}", flush=True)

t0 = time.time()
import torch
print(f"[{time.time()-t0:.2f}s] torch imported: {mem()}", flush=True)

t0 = time.time()
from src.web.data_loader import get_data_loader
print(f"[{time.time()-t0:.2f}s] data_loader imported: {mem()}", flush=True)

t0 = time.time()
loader = get_data_loader()
print(f"[{time.time()-t0:.2f}s] loader instantiated: {mem()}", flush=True)

