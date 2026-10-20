"""Read-only facts from the OS kernel. On Linux (Ubuntu) they come from /proc and /etc/os-release."""
from __future__ import annotations

import os
import platform
from pathlib import Path


def _distro() -> str:
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.system()


def _meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                out[key] = int(rest.split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return {"total_mb": out.get("MemTotal"), "available_mb": out.get("MemAvailable")} if out else {}


def system_info() -> dict:
    info: dict = {"system": platform.system(), "distro": _distro(), "kernel": platform.release(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "memory": _meminfo()}
    if hasattr(os, "getloadavg"):
        info["load_average"] = [round(x, 2) for x in os.getloadavg()]
    return info
