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

stat = MEMORYSTATUSEX()
stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

print(f"Memory load: {stat.dwMemoryLoad}%")
print(f"Total Physical: {stat.ullTotalPhys / (1024**3):.2f} GB")
print(f"Available Physical: {stat.ullAvailPhys / (1024**3):.2f} GB")
print(f"Total PageFile: {stat.ullTotalPageFile / (1024**3):.2f} GB")
print(f"Available PageFile: {stat.ullAvailPageFile / (1024**3):.2f} GB")

