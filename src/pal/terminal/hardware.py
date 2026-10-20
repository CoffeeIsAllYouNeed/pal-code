"""What this computer can run: its GPUs and their video memory (VRAM), Apple Silicon's shared memory, and system RAM.

Read without installing anything: `nvidia-smi` for NVIDIA cards (it ships with the driver), sysfs for AMD on Linux, the registry on Windows, `sysctl`
on macOS. Every probe is optional and short; a probe that fails simply finds nothing, and Pal then plans for running on the CPU.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GPU:
    name: str
    vram_gb: float
    vendor: str            # "nvidia" | "amd" | "intel" | "apple" | "other"


@dataclass
class Hardware:
    system: str = ""                         # "Windows" | "Darwin" | "Linux"
    ram_gb: float = 0.0
    gpus: list[GPU] = field(default_factory=list)
    apple_silicon: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def discrete(self) -> list[GPU]:
        """Cards with their own memory that Ollama can use (NVIDIA, AMD). Intel graphics share system memory and are treated as CPU."""
        return [g for g in self.gpus if g.vendor in ("nvidia", "amd") and g.vram_gb >= 1.5]

    @property
    def vram_gb(self) -> float:
        """Memory a model can use on the GPU: all NVIDIA (or all AMD) cards together, since Ollama splits a model across cards; Apple: shared memory."""
        if self.apple_silicon:
            return round(self.ram_gb * 0.70, 1)          # macOS lets the GPU use about two thirds to three quarters of unified memory
        by_vendor: dict[str, float] = {}
        for g in self.discrete:
            by_vendor[g.vendor] = by_vendor.get(g.vendor, 0.0) + g.vram_gb
        return round(max(by_vendor.values(), default=0.0), 1)

    @property
    def kind(self) -> str:
        return "apple" if self.apple_silicon else ("gpu" if self.vram_gb else "cpu")

    def summary(self) -> str:
        if self.apple_silicon:
            chip = self.gpus[0].name if self.gpus else "Apple Silicon"
            return f"{chip} · {self.ram_gb:g} GB unified memory (about {self.vram_gb:g} GB for models)"
        if self.discrete:
            names = ", ".join(f"{g.name} {g.vram_gb:g} GB" for g in self.discrete)
            return f"{names} · {self.ram_gb:g} GB RAM"
        return f"no GPU · {self.ram_gb:g} GB RAM"


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    exe = shutil.which(cmd[0])
    if not exe:
        return ""
    try:
        r = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia() -> list[GPU]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        name, _, mb = line.rpartition(",")
        try:
            gpus.append(GPU(name.strip() or "NVIDIA GPU", round(float(mb) / 1024, 1), "nvidia"))
        except ValueError:
            continue
    return gpus


def _amd_linux(root: Path = Path("/sys/class/drm")) -> list[GPU]:
    gpus = []
    for card in sorted(root.glob("card[0-9]*")) if root.is_dir() else []:
        dev = card / "device"
        vram = dev / "mem_info_vram_total"
        if "-" in card.name or not vram.is_file():
            continue
        try:
            gb = round(int(vram.read_text().strip()) / 1024**3, 1)
        except (OSError, ValueError):
            continue
        name = "AMD GPU"
        for f in ("product_name", "model"):
            try:
                name = (dev / f).read_text().strip() or name
                break
            except OSError:
                continue
        gpus.append(GPU(name, gb, "amd"))
    return gpus


def _windows_gpus() -> list[GPU]:
    """Every display adapter from the registry, with its real memory size (qwMemorySize; WMI's AdapterRAM stops at 4 GB)."""
    if sys.platform != "win32":
        return []
    import winreg
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return []
    for i in range(32):
        try:
            sub = winreg.OpenKey(root, f"{i:04d}")
        except OSError:
            break
        try:
            name = str(winreg.QueryValueEx(sub, "DriverDesc")[0])
        except OSError:
            continue
        size = 0
        for value in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
            try:
                raw = winreg.QueryValueEx(sub, value)[0]
                size = int.from_bytes(raw, "little") if isinstance(raw, bytes) else int(raw)
                break
            except (OSError, ValueError, TypeError):
                continue
        low = name.lower()
        vendor = "nvidia" if "nvidia" in low else "amd" if ("amd" in low or "radeon" in low) else "intel" if "intel" in low else "other"
        gpus.append(GPU(name, round(size / 1024**3, 1), vendor))
    return gpus


def _ram_gb(system: str) -> float:
    try:
        if system == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
        if system == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return round(int(out.strip()) / 1024**3, 1) if out.strip() else 0.0
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1024**3, 1)
    except (OSError, ValueError, AttributeError):
        pass
    if sys.platform == "win32":
        return 0.0
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def detect() -> Hardware:
    """Look at this computer. Never raises."""
    system = platform.system()
    hw = Hardware(system=system, ram_gb=_ram_gb(system))
    if system == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        hw.apple_silicon = True
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or "Apple Silicon"
        hw.gpus.append(GPU(chip, hw.ram_gb, "apple"))
        return hw
    hw.gpus = _nvidia()
    if system == "Linux":
        hw.gpus += _amd_linux()
    if system == "Windows":
        known = {g.name for g in hw.gpus}
        hw.gpus += [g for g in _windows_gpus() if g.name not in known]
    if not hw.discrete and any(g.vendor == "nvidia" for g in hw.gpus):
        hw.notes.append("an NVIDIA card was found but not its memory: install or update the NVIDIA driver so nvidia-smi works")
    return hw


def from_spec(text: str) -> Hardware:
    """A pretend computer for trying out recommendations: "cpu:16", "gpu:12" (VRAM GB, RAM 32), "gpu:24:64", "mac:16"."""
    m = re.fullmatch(r"\s*(cpu|gpu|mac):(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))?\s*", text.lower())
    if not m:
        raise ValueError("use cpu:<RAM GB>, gpu:<VRAM GB>[:<RAM GB>] or mac:<memory GB>")
    kind, a, b = m.group(1), float(m.group(2)), float(m.group(3)) if m.group(3) else None
    if kind == "cpu":
        return Hardware("Linux", a)
    if kind == "mac":
        return Hardware("Darwin", a, [GPU("Apple Silicon", a, "apple")], apple_silicon=True)
    return Hardware("Linux", b or max(16.0, a * 2), [GPU("GPU", a, "nvidia")])
