"""Saving and loading an index snapshot as one JSON file, written atomically and checked with a checksum when loaded."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from localrag.core.errors import IndexingError

SNAPSHOT_VERSION = 1


def _digest(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def save_snapshot(path: str | Path, payload: dict[str, Any]) -> int:
    """Write the payload to path via a temporary file, so a crash never leaves a half-written snapshot. Returns the bytes written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    document = json.dumps({"version": SNAPSHOT_VERSION, "sha256": _digest(body), "payload": payload}, sort_keys=True, separators=(",", ":"))
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(document)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return len(document.encode("utf-8"))


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Read a snapshot back, refusing a file that is not valid JSON, has another version, or fails its checksum."""
    p = Path(path)
    if not p.is_file():
        raise IndexingError(f"no snapshot at {p}")
    try:
        document = json.loads(p.read_text(encoding="utf-8"))
        payload = document["payload"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise IndexingError(f"{p} is not a LocalRAG snapshot: {e}") from e
    if document.get("version") != SNAPSHOT_VERSION:
        raise IndexingError(f"snapshot version {document.get('version')} is not supported (this is version {SNAPSHOT_VERSION})")
    if document.get("sha256") != _digest(json.dumps(payload, sort_keys=True, separators=(",", ":"))):
        raise IndexingError(f"{p} is corrupt: its checksum does not match")
    return payload
