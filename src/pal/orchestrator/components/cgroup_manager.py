from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")
log = logging.getLogger(__name__)
_SLICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_LIMIT_VALUE = re.compile(r"^[0-9A-Za-z ]{1,32}$")


def _check_name(name: str) -> str:
    """Slice names become directory names under /sys/fs/cgroup and are used in privileged commands,
    so only plain names are accepted."""
    if not _SLICE_NAME.match(name or "") or name.endswith("."):
        raise CgroupError(f"invalid slice name: {name!r}")
    return name


def _check_value(value: str) -> str:
    if not _LIMIT_VALUE.match(str(value)):
        raise CgroupError(f"invalid limit value: {value!r}")
    return str(value)


@dataclass
class ResourceLimits:
    memory_max: str | None = None   # e.g. "256M", "1G"
    cpu_max: str | None = None      # e.g. "50000 100000" = 50% of one CPU
    pids_max: str | None = None     # e.g. "50"


class CgroupError(RuntimeError):
    pass


class CgroupManager:
    """
    Creates and manages cgroup v2 slices, one per agent, under a single
    parent slice (default: /sys/fs/cgroup/pal.slice/<agent_name>).

    Requires root (via sudo) on this host -- WSL2's systemd user-session
    delegation wasn't reliably attached to interactive shells here, so
    operations run through `sudo` explicitly rather than assuming direct
    write access. This is isolated behind this module so a future port
    to a real delegated slice (or to a custom kernel) only changes
    _run(), not any caller.
    """

    enforced = True

    def __init__(self, parent_slice: str = "pal2.slice"):
        self.parent_path = CGROUP_ROOT / _check_name(parent_slice)
        self._ensure_parent()

    @staticmethod
    def supported() -> bool:
        """Linux with cgroup v2 mounted, and either root or sudo to write to it."""
        if not sys.platform.startswith("linux") or not (CGROUP_ROOT / "cgroup.controllers").exists():
            return False
        return _is_root() or shutil.which("sudo") is not None

    def _privileged(self, cmd: list[str]) -> list[str]:
        return cmd if _is_root() else ["sudo", *cmd]

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        result = subprocess.run(self._privileged(cmd), capture_output=True, text=True)
        if result.returncode != 0:
            raise CgroupError(f"command failed: {' '.join(cmd)}\n{result.stderr}")
        return result

    def _write(self, path: Path, value: str) -> None:
        # value goes in on stdin to `tee`: no shell, so nothing in `value` or `path` can inject a command
        result = subprocess.run(self._privileged(["tee", str(path)]), input=f"{value}\n", capture_output=True,
                                text=True)
        if result.returncode != 0:
            raise CgroupError(f"failed writing {value!r} to {path}: {result.stderr}")

    def _ensure_parent(self) -> None:
        if not self.parent_path.exists():
            self._run(["mkdir", "-p", str(self.parent_path)])

        # A cgroup only passes controllers down to its own children once
        # they're explicitly enabled in *its* cgroup.subtree_control --
        # inheriting them at the root level isn't enough. Without this,
        # child slices exist as directories but reject writes to
        # memory.max / cpu.max / pids.max with Permission denied.
        root_subtree = CGROUP_ROOT / "cgroup.subtree_control"
        if root_subtree.exists():
            enabled = root_subtree.read_text(encoding="utf-8").split()
            missing = [c for c in ("cpu", "memory", "pids") if c not in enabled]
            if missing:
                flags = " ".join(f"+{c}" for c in missing)
                self._write(root_subtree, flags)

        parent_subtree = self.parent_path / "cgroup.subtree_control"
        if parent_subtree.exists():
            enabled = parent_subtree.read_text(encoding="utf-8").split()
            missing = [c for c in ("cpu", "memory", "pids") if c not in enabled]
            if missing:
                flags = " ".join(f"+{c}" for c in missing)
                self._write(parent_subtree, flags)

    def create_slice(self, agent_name: str, limits: ResourceLimits) -> Path:
        slice_path = self.parent_path / _check_name(agent_name)
        if slice_path.exists():
            raise CgroupError(f"slice already exists: {agent_name}")

        self._run(["mkdir", "-p", str(slice_path)])

        if limits.memory_max:
            self._write(slice_path / "memory.max", _check_value(limits.memory_max))
        if limits.cpu_max:
            self._write(slice_path / "cpu.max", _check_value(limits.cpu_max))
        if limits.pids_max:
            self._write(slice_path / "pids.max", _check_value(limits.pids_max))

        return slice_path

    def assign_pid(self, agent_name: str, pid: int) -> None:
        slice_path = self.parent_path / agent_name
        if not slice_path.exists():
            raise CgroupError(f"no such slice: {agent_name}")
        self._write(slice_path / "cgroup.procs", str(pid))

    def current_pids(self, agent_name: str) -> list[int]:
        slice_path = self.parent_path / agent_name
        procs_file = slice_path / "cgroup.procs"
        if not procs_file.exists():
            raise CgroupError(f"no such slice: {agent_name}")
        content = procs_file.read_text(encoding="utf-8").strip()
        return [int(p) for p in content.splitlines() if p]

    def stats(self, agent_name: str) -> dict:
        slice_path = self.parent_path / agent_name
        stats = {}
        for fname in ("memory.current", "memory.max", "pids.current"):
            fpath = slice_path / fname
            if fpath.exists():
                stats[fname] = fpath.read_text(encoding="utf-8").strip()
        return stats

    def delete_slice(self, agent_name: str) -> None:
        slice_path = self.parent_path / agent_name
        if not slice_path.exists():
            return
        # cgroups can only be removed once empty of processes.
        pids = self.current_pids(agent_name)
        if pids:
            raise CgroupError(f"cannot delete slice {agent_name}: still has PIDs {pids}")
        self._run(["rmdir", str(slice_path)])

    def list_slices(self) -> list[str]:
        if not self.parent_path.exists():
            return []
        return [p.name for p in self.parent_path.iterdir() if p.is_dir()]


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


class NullResourceManager:
    """Used where cgroups are unavailable (macOS, Windows, Linux without cgroup v2 or sudo). Agents still
    run as separate processes, but memory/CPU/process limits are NOT enforced. The orchestrator logs
    that once, at construction."""

    enforced = False

    def __init__(self, parent_slice: str = "pal.slice"):
        self.parent_slice = parent_slice
        self._slices: dict[str, ResourceLimits] = {}

    def create_slice(self, agent_name: str, limits: ResourceLimits):
        if agent_name in self._slices:
            raise CgroupError(f"slice already exists: {agent_name}")
        self._slices[agent_name] = limits
        return

    def assign_pid(self, agent_name: str, pid: int) -> None:
        if agent_name not in self._slices:
            raise CgroupError(f"no such slice: {agent_name}")

    def current_pids(self, agent_name: str) -> list[int]:
        return []

    def stats(self, agent_name: str) -> dict:
        return {"enforced": False}

    def delete_slice(self, agent_name: str) -> None:
        self._slices.pop(agent_name, None)

    def list_slices(self) -> list[str]:
        return sorted(self._slices)


def create_resource_manager(parent_slice: str = "pal.slice", manager_cls=CgroupManager):
    """A real cgroup manager where the OS supports it, otherwise a no-op one (with a warning)."""
    if manager_cls is not CgroupManager or CgroupManager.supported():
        return manager_cls(parent_slice)
    log.warning("cgroup v2 is not available here (needs Linux with cgroup v2 and root or sudo): agent "
                "resource limits will NOT be enforced")
    return NullResourceManager(parent_slice)
