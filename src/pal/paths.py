"""Where Pal keeps its data, the same way on Linux, macOS and Windows.

Default is `~/.pal` (Path.home() resolves correctly on every OS). Set PAL_HOME to move it.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def pal_home() -> Path:
    env = os.environ.get("PAL_HOME")
    return Path(env).expanduser() if env else Path.home() / ".pal"


def data_dir(*parts: str) -> Path:
    return pal_home().joinpath(*parts)


def safe_id(value: str, default: str = "default", max_len: int = 64) -> str:
    """Turn any string into something safe as one path component on every OS."""
    out = _SAFE_ID.sub("", value or "")[:max_len]
    if not out or out.lower() in _WINDOWS_RESERVED:
        return default if not out else f"_{out}"
    return out


def is_safe_relative_key(key: str) -> bool:
    """True when `key` is a relative path that stays inside its folder on every OS: no `..`, no drive
    letters or absolute paths, no backslashes, no characters Windows forbids, no reserved device names."""
    if not key or key != key.strip() or "\\" in key or key.startswith("/") or "\x00" in key:
        return False
    for part in key.split("/"):
        if part in ("", ".", "..") or re.search(r'[<>:"|?*\x00-\x1f]', part) or part.endswith((".", " ")):
            return False
        if part.split(".")[0].lower() in _WINDOWS_RESERVED:
            return False
    return True


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file and rename, so a crash never leaves a half-written file (os.replace is
    atomic on POSIX and replaces an existing file on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))
