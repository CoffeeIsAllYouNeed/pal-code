"""
Local-file storage backend, with one root subdirectory PER AGENT so different agents' files never
collide. Values are stored as JSON when JSON-serializable and as raw bytes otherwise.

Portable and safe on Linux, macOS and Windows:
- agent ids and keys are validated (no `..`, absolute paths, drive letters, backslashes, characters
  or device names Windows forbids), and the resolved path must stay inside the agent's folder;
- writes are atomic (temp file + rename), so a crash never leaves a half-written value;
- keys are always reported with forward slashes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from pal.errors import UserInputError
from pal.paths import atomic_write_bytes, is_safe_relative_key, safe_id


class InvalidStorageKey(UserInputError):
    pass


class LocalFileBackend:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _agent_dir(self, agent_id: str) -> Path:
        if not agent_id or safe_id(agent_id) != agent_id:
            raise InvalidStorageKey(f"invalid agent id: {agent_id!r}")
        d = self.root / agent_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, agent_id: str, key: str) -> Path:
        base = self._agent_dir(agent_id)
        if not is_safe_relative_key(key):
            raise InvalidStorageKey(f"invalid storage key: {key!r}")
        path = (base / key).resolve()
        if base.resolve() not in path.parents:
            raise InvalidStorageKey(f"storage key escapes its folder: {key!r}")
        return path

    def write(self, agent_id: str, key: str, data: Any) -> None:
        path = self._path(agent_id, key)
        raw = bytes(data) if isinstance(data, (bytes, bytearray)) else json.dumps(data).encode("utf-8")
        atomic_write_bytes(path, raw)

    def read(self, agent_id: str, key: str) -> Optional[Any]:
        path = self._path(agent_id, key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return path.read_bytes()

    def delete(self, agent_id: str, key: str) -> None:
        path = self._path(agent_id, key)
        if path.is_file():
            path.unlink()

    def exists(self, agent_id: str, key: str) -> bool:
        return self._path(agent_id, key).is_file()

    def list_keys(self, agent_id: str, prefix: str = "") -> list[str]:
        agent_dir = self._agent_dir(agent_id)
        keys = []
        for p in agent_dir.rglob("*"):
            if not p.is_file() or p.name.endswith(".tmp"):
                continue
            rel = p.relative_to(agent_dir).as_posix()
            if rel.startswith(prefix):
                keys.append(rel)
        return sorted(keys)

    def purge(self, agent_id: str) -> None:
        """Delete everything stored for one agent."""
        d = self.root / agent_id
        if d.is_dir() and safe_id(agent_id) == agent_id:
            shutil.rmtree(d, ignore_errors=True)
