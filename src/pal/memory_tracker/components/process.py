"""How much memory this process uses, as best the platform allows. Never raises: anything unavailable is None."""
from __future__ import annotations

import sys


def process_memory() -> dict:
    """{"rss_mb", "peak_rss_mb", "source"}. Tries psutil, then the Unix `resource` module, then Windows' own API (ctypes)."""
    try:
        import psutil
        p = psutil.Process()
        rss = p.memory_info().rss
        return {"rss_mb": round(rss / 1048576, 1), "peak_rss_mb": None, "source": "psutil"}
    except Exception:  # noqa: BLE001  (not installed, or not allowed)
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / 1048576 if sys.platform == "darwin" else peak / 1024        # macOS reports bytes, Linux kilobytes
        return {"rss_mb": None, "peak_rss_mb": round(peak_mb, 1), "source": "resource"}
    except Exception:  # noqa: BLE001  (Windows has no `resource`)
        pass
    if sys.platform == "win32":                                                      # not tested outside Windows
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = Counters()
            c.cb = ctypes.sizeof(c)
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):  # type: ignore[attr-defined]
                return {"rss_mb": round(c.WorkingSetSize / 1048576, 1), "peak_rss_mb": round(c.PeakWorkingSetSize / 1048576, 1), "source": "win32"}
        except Exception:  # noqa: BLE001
            pass
    return {"rss_mb": None, "peak_rss_mb": None, "source": "unavailable"}


def dir_bytes(path, limit_files: int = 20000) -> int:
    """The size of a folder in bytes, counting at most `limit_files` files (a stats call must stay quick)."""
    import os
    total = seen = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
            seen += 1
            if seen >= limit_files:
                return total
    return total
