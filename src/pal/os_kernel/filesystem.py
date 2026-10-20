"""File operations: the part of the OS kernel's filesystem an agent may touch.

Every agent gets its own workspace folder. A path must be relative and stay inside it: no `..`, absolute paths,
drive letters, backslashes or names Windows forbids, and a symlink that points outside is refused. Writes are atomic.
"""
from __future__ import annotations

from pathlib import Path

from pal.errors import UserInputError
from pal.paths import atomic_write_bytes, is_safe_relative_key


class OSKernelError(UserInputError):
    """A request to the OS kernel was refused or failed. The message is safe to show."""


class Workspace:
    def __init__(self, root: Path, max_read: int, max_write: int, max_binary: int = 20_000_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_read = max_read
        self.max_write = max_write
        self.max_binary = max_binary

    def resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or not is_safe_relative_key(rel):
            raise OSKernelError(f"invalid path: {rel!r} (use a relative path inside your workspace)")
        target = (self.root / rel).resolve()
        if self.root.resolve() not in target.parents:
            raise OSKernelError(f"path escapes the workspace: {rel!r}")
        return target

    def exists(self, rel: str) -> bool:
        return self.resolve(rel).is_file()

    def read(self, rel: str) -> str:
        p = self.resolve(rel)
        if not p.is_file():
            raise OSKernelError(f"no such file: {rel}")
        if p.stat().st_size > self.max_read:
            raise OSKernelError(f"{rel} is larger than the {self.max_read:,}-byte read limit")
        return p.read_text(encoding="utf-8", errors="replace")

    def read_bytes(self, rel: str) -> bytes:
        p = self.resolve(rel)
        if not p.is_file():
            raise OSKernelError(f"no such file: {rel}")
        if p.stat().st_size > self.max_binary:
            raise OSKernelError(f"{rel} is larger than the {self.max_binary:,}-byte limit for binary files")
        return p.read_bytes()

    def write_bytes(self, rel: str, data: bytes) -> int:
        if len(data) > self.max_binary:
            raise OSKernelError(f"the data is larger than the {self.max_binary:,}-byte limit for binary files")
        atomic_write_bytes(self.resolve(rel), bytes(data))
        return len(data)

    def write(self, rel: str, content: str) -> int:
        data = str(content).encode("utf-8")
        if len(data) > self.max_write:
            raise OSKernelError(f"content is larger than the {self.max_write:,}-byte write limit")
        atomic_write_bytes(self.resolve(rel), data)
        return len(data)

    def delete(self, rel: str) -> None:
        p = self.resolve(rel)
        if not p.is_file():
            raise OSKernelError(f"no such file: {rel}")
        p.unlink()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root.resolve()
        out = []
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                rel = p.relative_to(base).as_posix()
                if rel.startswith(prefix):
                    out.append(rel)
        return out
