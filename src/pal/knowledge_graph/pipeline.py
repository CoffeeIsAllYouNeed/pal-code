"""The Knowledge Graph component: send the relevant slice of a project, not the project.

A zip of source code (or one large code file) attached to the chat is indexed once,
locally, with graphify (AST only: no LLM call, nothing leaves the machine). After that
each API request carries only the graph nodes and code excerpts that match the current
question, within `budget_tokens`. The excerpts are added to the request but not stored
in the chat history, so they are not resent on every later turn.

Only source code is indexed. graphify's semantic extraction for docs and images needs
an LLM and is not used here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from pal.config import KnowledgeGraphConfig
from pal.knowledge_graph.components import builder, retriever, workspace
from pal.knowledge_graph.components.base import (
    GRAPH_NOTICE,
    GraphError,
    GraphExcerpts,
    ProjectInfo,
    Runner,
)

__all__ = ["KnowledgeGraph", "GraphBlock", "GraphError", "GraphExcerpts", "ProjectInfo", "GRAPH_NOTICE"]

log = logging.getLogger(__name__)


@dataclass
class GraphBlock:
    """What the knowledge graph contributes to one request."""
    text: str            # the excerpts, ready to add to the request
    saved_tokens: int    # what sending the whole project would have cost, minus these excerpts
    info: dict           # nodes, excerpts, tokens and project tokens, for the turn's report


class KnowledgeGraph:
    def __init__(self, cfg=None, runner: Optional[Runner] = None):
        self.cfg = KnowledgeGraphConfig.coerce(cfg)
        self.runner: Runner = runner or builder.default_runner
        self.root = self.cfg.resolved_dir()

    # -- config ------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def _ws(self, session_id: str) -> workspace.Workspace:
        return workspace.Workspace(self.root, session_id)

    def available(self) -> bool:
        return self.runner is not builder.default_runner or builder.graphify_available()

    def wants(self, name: str, raw: bytes) -> bool:
        """Should this upload be indexed instead of pasted into the chat?"""
        if not self.enabled:
            return False
        if workspace.ext(name) == "zip":
            return True  # a zip can only be used as a project; fails clearly if graphify is missing
        return (self.available() and workspace.is_code(name)
                and len(raw) > self.cfg.min_code_file_chars)

    # -- indexing --------------------------------------------------------------------------
    def ingest(self, session_id: str, name: str, raw: bytes) -> dict:
        """Index an upload; returns the attachment dict the chat should show the API model."""
        if not self.available():
            raise GraphError(f"{name}: indexing code needs graphify (pip install graphifyy)")
        ws = self._ws(session_id)
        try:
            if workspace.ext(name) == "zip":
                ws.add_zip(raw, self.cfg.max_zip_files, self.cfg.max_zip_mb * 1_000_000)
            else:
                ws.add_file(Path(name).name, raw)
            info = builder.build(ws, Path(name).stem if workspace.ext(name) == "zip" else Path(name).name, self.runner)
        except GraphError as e:
            raise GraphError(f"{name}: {e}") from e
        return {"name": name, "text": info.summary(), "note": "indexed as a code graph; source not included",
                "kind": "project"}

    def stats(self) -> dict:
        """How many projects are indexed right now and how much disk they take (for the Memory Tracker)."""
        from pal.memory_tracker import dir_bytes
        projects = sum(1 for d in self.root.iterdir() if d.is_dir()) if self.root.is_dir() else 0
        return {"enabled": self.enabled, "projects": projects, "bytes": dir_bytes(self.root) if projects else 0}

    def has_project(self, session_id: str) -> bool:
        return self.enabled and self._ws(session_id).exists()

    # -- per-question ------------------------------------------------------------------------
    def retrieve(self, session_id: str, question: str,
                 count_tokens: Callable[[str], int]) -> Optional[GraphExcerpts]:
        if not self.has_project(session_id) or not question.strip():
            return None
        return retriever.retrieve(self._ws(session_id), question, self.runner, count_tokens,
                                  self.cfg.budget_tokens, self.cfg.max_slice_lines)

    def context_block(self, session_id: str, question: str, count_tokens: Callable[[str], int]) -> Optional[GraphBlock]:
        """The excerpts of the loaded project that match the question, as a block for the request, with the tokens saved against sending the whole
        project. None when there is no project or nothing matches. A failure is logged and gives None: the chat never fails because the graph did."""
        try:
            ex = self.retrieve(session_id, question, count_tokens)
        except Exception:  # noqa: BLE001
            log.exception("graph retrieval failed")
            return None
        if not ex:
            return None
        return GraphBlock(ex.text, max(0, ex.project_tokens - ex.tokens),
                          {"nodes": ex.nodes, "excerpts": ex.excerpts, "tokens": ex.tokens, "project_tokens": ex.project_tokens})

    def read_source(self, session_id: str, path: str, start: int = 1, end: int = 0, max_lines: int = 120) -> str:
        """Read-only view of one indexed source file (used by the read_source tool)."""
        from pal.errors import NotFoundError
        from pal.paths import is_safe_relative_key
        ws = self._ws(session_id)
        rel = path.strip().replace("\\", "/").removeprefix("src/")
        if not ws.exists() or not is_safe_relative_key(rel):
            raise NotFoundError(f"no such file in the project: {path!r}")
        target = (ws.src / rel).resolve()
        if ws.src.resolve() not in target.parents or not target.is_file() or not workspace.is_code(target.name):
            raise NotFoundError(f"no such file in the project: {path!r}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start)
        end = min(len(lines), end if end >= start else start + max_lines - 1, start + max_lines - 1)
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return f"[{rel} lines {start}-{end} of {len(lines)}]\n{body}"

    # -- read-only code navigation, no model involved ---------------------------------------------------------------------
    def _code_files(self, session_id: str) -> list:
        ws = self._ws(session_id)
        if not ws.exists():
            return []
        return sorted(p for p in ws.src.rglob("*") if p.is_file() and workspace.is_code(p.name))

    def list_files(self, session_id: str, prefix: str = "", limit: int = 200) -> str:
        """The project's source files, one per line with the line count."""
        ws = self._ws(session_id)
        rows = []
        for p in self._code_files(session_id):
            rel = p.relative_to(ws.src).as_posix()
            if rel.startswith(prefix.strip().lstrip("/")):
                rows.append(f"{rel} ({sum(1 for _ in p.open(encoding='utf-8', errors='replace'))} lines)")
        if not rows:
            return "no files match" if prefix else "no project"
        more = f"\n[{len(rows) - limit} more not shown: use a prefix]" if len(rows) > limit else ""
        return "\n".join(rows[:limit]) + more

    def search(self, session_id: str, text: str, limit: int = 25) -> str:
        """Lines containing `text` (case-insensitive), as path:line: text."""
        needle = text.strip().lower()
        if len(needle) < 2:
            return "give at least two characters to search for"
        return self._grep(session_id, lambda line: needle in line.lower(), limit, f"search for {text.strip()!r}")

    def find_usages(self, session_id: str, symbol: str, limit: int = 25) -> str:
        """Where a function, class or variable is defined and where its name is used (a word-boundary text match)."""
        import re
        sym = symbol.strip()
        if not re.fullmatch(r"[A-Za-z_$][\w$.]*", sym):
            return "give a plain name: letters, digits, underscore"
        word = re.compile(rf"(?<![\w$]){re.escape(sym.split('.')[-1])}(?![\w$])")
        define = re.compile(rf"^\s*(?:export\s+|async\s+|public\s+|private\s+|static\s+)*(?:def|class|function|func|fn|interface|struct|enum|trait|type)\s+"
                            rf"{re.escape(sym.split('.')[-1])}\b")
        defs, uses = [], []
        for p in self._code_files(session_id):
            rel = p.relative_to(self._ws(session_id).src).as_posix()
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if word.search(line):
                    (defs if define.match(line) else uses).append(f"{rel}:{n}: {line.strip()[:160]}")
        if not defs and not uses:
            return f"no match for {sym!r} in the project"
        files = len({u.split(':')[0] for u in defs + uses})
        out = [f"{sym}: {len(defs)} definition(s), {len(uses)} other use(s), in {files} file(s)"]
        out += [f"definition: {d}" for d in defs[:5]]
        out += uses[:limit]
        if len(uses) > limit:
            out.append(f"[{len(uses) - limit} more uses not shown]")
        return "\n".join(out)

    def _grep(self, session_id: str, test, limit: int, what: str) -> str:
        ws = self._ws(session_id)
        hits = []
        for p in self._code_files(session_id):
            rel = p.relative_to(ws.src).as_posix()
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if test(line):
                    hits.append(f"{rel}:{n}: {line.strip()[:160]}")
        if not hits:
            return f"no results for {what}"
        extra = f"\n[{len(hits) - limit} more results not shown]" if len(hits) > limit else ""
        return "\n".join(hits[:limit]) + extra

    def reset(self, session_id: str) -> None:
        self._ws(session_id).reset()

    def prune_stale(self, max_age_seconds: float) -> int:
        """Delete indexed projects nobody has touched for `max_age_seconds` (left behind by a crash or a
        request that failed after the upload). Returns how many were removed."""
        import shutil
        import time
        removed = 0
        if not self.root.is_dir():
            return 0
        for d in self.root.iterdir():
            try:
                if d.is_dir() and time.time() - d.stat().st_mtime > max_age_seconds:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed
