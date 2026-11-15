"""
Pytest configuration and environment preflight checks for D:\\ETGNN-Fraud-Detection.
Detects whether the execution environment blocks Windows 127.0.0.1 TCP loopback connections
(which causes Python 3.13 socket.socketpair() -> _fallback_socketpair() -> lsock.accept()
and asyncio event loop creation to block indefinitely in network-isolated sandboxes).
"""

import socket
import pytest


def check_loopback_socketpair_supported(timeout_sec: float = 0.75) -> bool:
    """
    Safely probe whether 127.0.0.1 loopback TCP accept() succeeds within timeout_sec.
    Does not monkey-patch socket internals.
    """
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    csock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(1)
        lsock.settimeout(timeout_sec)
        addr = lsock.getsockname()
        csock.setblocking(False)
        try:
            csock.connect(addr)
        except (BlockingIOError, InterruptedError):
            pass
        conn, _ = lsock.accept()
        conn.close()
        return True
    except OSError:
        return False
    finally:
        csock.close()
        lsock.close()


LOOPBACK_SOCKETPAIR_AVAILABLE = check_loopback_socketpair_supported()
ENVIRONMENT_BLOCKED_REASON = (
    "ENVIRONMENT-BLOCKED: Execution sandbox blocks 127.0.0.1 TCP loopback connections, "
    "causing Python 3.13 socket.socketpair() (_fallback_socketpair -> accept) and "
    "asyncio/Starlette TestClient event loop creation to block."
)

