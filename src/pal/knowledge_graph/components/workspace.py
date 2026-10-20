"""Per-session project workspace: safe zip extraction and source bookkeeping."""
from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from pal.knowledge_graph.components.base import GraphError
from pal.paths import is_safe_relative_key

# Files graphify's AST extractor understands (the LLM-free part).
CODE_EXTENSIONS = {
    "py", "js", "jsx", "mjs", "cjs", "ts", "tsx", "go", "rs", "java", "kt", "kts", "scala", "c", "h", "cc", "cpp",
    "hpp", "cs", "rb", "php", "swift", "lua", "zig", "ex", "exs", "sh", "bash", "ps1", "jl", "m", "f90", "v", "sv",
}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "graphify-out", "dist", "build", ".idea",
             ".vscode", ".mypy_cache", ".pytest_cache", "site-packages"}
MAX_FILE_BYTES = 1_000_000


def safe_id(session_id: str) -> str:
    sid = re.sub(r"[^A-Za-z0-9_-]", "", session_id or "")[:64]
    return sid or "default"


def ext(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).suffix.lower().lstrip(".")


def is_code(name: str) -> bool:
    return ext(name) in CODE_EXTENSIONS


class Workspace:
    def __init__(self, root: Path, session_id: str):
        self.dir = root / safe_id(session_id)
        self.src = self.dir / "src"
        self.meta_path = self.dir / "meta.json"

    @property
    def graph_path(self) -> Path:
        return self.src / "graphify-out" / "graph.json"

    def exists(self) -> bool:
        return self.graph_path.is_file()

    def reset(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- writing sources -------------------------------------------------------------
    def _target(self, rel: str) -> Path:
        p = PurePosixPath(rel.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts or not p.parts or any(part in SKIP_DIRS for part in p.parts):
            raise GraphError(f"unsafe or ignored path in archive: {rel}")
        target = (self.src / p).resolve()
        if self.src.resolve() not in target.parents:
            raise GraphError(f"path escapes the project folder: {rel}")
        return target

    def add_file(self, rel: str, raw: bytes) -> None:
        t = self._target(rel)
        try:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(raw)
        except OSError as e:
            raise GraphError(f"could not write {rel}: {e.strerror or e}") from e

    def add_zip(self, raw: bytes, max_files: int, max_bytes: int) -> int:
        """Extract code files only. Skips binaries, huge files, junk dirs and symlinks.
        Returns the number of source files written."""
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile:
            raise GraphError("not a valid zip file") from None
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > max_files * 4:
            raise GraphError(f"archive has too many entries ({len(infos)})") from None
        written = total = 0
        for i in infos:
            parts = PurePosixPath(i.filename.replace("\\", "/")).parts
            if not parts or not is_code(parts[-1]) or any(p in SKIP_DIRS for p in parts):
                continue
            if not is_safe_relative_key("/".join(parts)):  # names Windows cannot store: skipped on every OS,
                continue                                   # so a project indexes identically everywhere
            if (i.external_attr >> 16) & 0o170000 == 0o120000:  # symlink
                continue
            if i.file_size > MAX_FILE_BYTES:
                continue
            total += i.file_size
            if total > max_bytes:
                raise GraphError(f"archive source is over {max_bytes // 1_000_000} MB")
            self.add_file("/".join(parts), zf.read(i))
            written += 1
            if written > max_files:
                raise GraphError(f"archive has more than {max_files} source files")
        if not written:
            raise GraphError("no source code files found in the archive")
        return written

    # -- reading back ------------------------------------------------------------------
    def source_files(self) -> list[Path]:
        return [p for p in self.src.rglob("*")
                if p.is_file() and is_code(p.name) and not any(part in SKIP_DIRS for part in p.relative_to(self.src).parts)]

    def load_meta(self) -> dict:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_meta(self, meta: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta), encoding="utf-8")
