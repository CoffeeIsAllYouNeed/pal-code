"""Setting up the folder Pal works in, when `pal cli` or `pal web` starts.

Four things are prepared for the current folder and kept in ~/.pal/projects/<folder id>/, and only what changed is redone next time:

  documents    PDF, Word, PowerPoint, Excel, EPUB, RTF and LaTeX files are converted to markdown files (markdown/<path>.md). Markdown and
               text files are used as they are. They are indexed, so each request carries only the passages that match it.
  project map  a short outline of the code: every file with its classes and functions. It is indexed too, so a question about
               "the login handler" brings the lines of the map that name it, not the whole project.
  code graph   built with graphify from the source (AST only: no model, nothing leaves the computer). Each request gets the
               graph nodes and code excerpts that match it. Built in the background, so the session opens at once.
  brief        languages, size, the test command: a few lines the model gets with the first request.

Nothing is written in the folder itself.
"""
from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from pal.knowledge_graph import KnowledgeGraph
from pal.knowledge_graph.components import builder
from pal.knowledge_graph.components.workspace import CODE_EXTENSIONS, SKIP_DIRS, Workspace
from pal.paths import pal_home
from pal.terminal.security import is_secret_file
from pal.terminal.workflow import project_id
from pal.tokens import estimate_tokens

DOC_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".rtf", ".epub", ".tex"}      # converted to markdown
TEXT_DOC_EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}                               # already text: used as they are
IGNORE_FILE = ".palignore"
NOTES_FILE = "PAL.md"           # the project's instructions for Pal, sent with the first request of every session
MAX_NOTES_CHARS = 8000
MAX_CODE_FILES = 5000
MAX_DOCS = 300
MAX_DOC_BYTES = 25_000_000
MAX_CODE_BYTES = 1_000_000
MAP_ENTRIES_PER_FILE = 40
GRAPH_WORKSPACE = "project"
LANGUAGES = {".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
             ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin", ".swift": "Swift", ".rb": "Ruby", ".php": "PHP",
             ".cs": "C#", ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".hpp": "C++", ".scala": "Scala", ".lua": "Lua", ".dart": "Dart",
             ".ex": "Elixir", ".exs": "Elixir", ".jl": "Julia", ".zig": "Zig", ".sh": "Shell"}

# a definition line in languages other than Python: good enough for an outline
_DEF = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|async\s+|abstract\s+|final\s+|open\s+|override\s+|pub(?:\(crate\))?\s+)*"
                  r"(?P<kind>class|interface|struct|enum|trait|type|function|func|fn|fun|def|object|module|impl)\s+(?P<name>[A-Za-z_$][\w$]*)")
_ARROW = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")


def _ext(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def read_ignore(folder: Path) -> list[str]:
    """Patterns from .palignore (gitignore-like: one per line, # for comments, a trailing / for folders)."""
    f = folder / IGNORE_FILE
    try:
        lines = f.read_text(encoding="utf-8").splitlines() if f.is_file() else []
    except OSError:
        return []
    return [x.strip() for x in lines if x.strip() and not x.strip().startswith("#")]


def ignored(rel: str, patterns: list[str], is_dir: bool = False) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for p in patterns:
        dir_only = p.endswith("/")
        p = p.rstrip("/").lstrip("/")
        if dir_only and not is_dir:
            if any(fnmatch.fnmatch(part, p) for part in rel.split("/")[:-1]) or fnmatch.fnmatch(rel.rsplit("/", 1)[0] if "/" in rel else "", p):
                return True
            continue
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) or rel.startswith(p + "/"):
            return True
    return False


@dataclass
class Scan:
    code: dict[str, tuple[int, int]] = field(default_factory=dict)       # path -> (mtime_ns, size)
    docs: dict[str, tuple[int, int]] = field(default_factory=dict)
    skipped: int = 0


def scan(folder: Path, patterns: list[str] | None = None) -> Scan:
    """Code files and documents in the folder, leaving out hidden folders, dependencies, build output, secrets files and .palignore."""
    patterns = read_ignore(folder) if patterns is None else patterns
    out = Scan()
    for root, dirs, files in os.walk(folder):
        rel_root = Path(root).relative_to(folder).as_posix()
        rel_root = "" if rel_root == "." else rel_root
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS and not d.endswith(".egg-info")
                         and not ignored(f"{rel_root}/{d}".lstrip("/"), patterns, is_dir=True))
        for name in sorted(files):
            rel = f"{rel_root}/{name}".lstrip("/")
            if name.startswith(".") or is_secret_file(name) or ignored(rel, patterns):
                continue
            ext = _ext(name)
            try:
                st = (Path(root) / name).stat()
            except OSError:
                continue
            if ext.lstrip(".") in CODE_EXTENSIONS or ext in LANGUAGES:
                if st.st_size <= MAX_CODE_BYTES and len(out.code) < MAX_CODE_FILES:
                    out.code[rel] = (st.st_mtime_ns, st.st_size)
                else:
                    out.skipped += 1
            elif ext in DOC_EXTENSIONS or ext in TEXT_DOC_EXTENSIONS:
                if st.st_size <= MAX_DOC_BYTES and len(out.docs) < MAX_DOCS:
                    out.docs[rel] = (st.st_mtime_ns, st.st_size)
                else:
                    out.skipped += 1
    return out


# -- project map ----------------------------------------------------------------------------------------------------------------
def outline(rel: str, text: str) -> list[str]:
    """The classes and functions in one file, indented by nesting (Python), as short lines."""
    if rel.endswith(".py"):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return []
        rows: list[str] = []

        def walk(nodes, depth: int) -> None:
            for n in nodes:
                if len(rows) >= MAP_ENTRIES_PER_FILE:
                    return
                if isinstance(n, ast.ClassDef):
                    rows.append(f"{'  ' * depth}class {n.name}")
                    walk(n.body, depth + 1)
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not (depth and n.name.startswith("__") and n.name != "__init__"):
                    args = [a.arg for a in n.args.args if a.arg not in ("self", "cls")]
                    rows.append(f"{'  ' * depth}def {n.name}({', '.join(args)})")
        walk(tree.body, 0)
        return rows
    rows = []
    for line in text.splitlines():
        m = _DEF.match(line) or _ARROW.match(line)
        if m:
            indent = "  " if line[:1] in (" ", "\t") else ""
            kind = m.groupdict().get("kind") or "const"
            rows.append(f"{indent}{kind} {m.group('name')}")
            if len(rows) >= MAP_ENTRIES_PER_FILE:
                break
    return rows


def build_map(folder: Path, files: list[str]) -> str:
    out = ["# Project map", "", "Every source file with its classes and functions. Ask for a file to see its code.", ""]
    for rel in files:
        try:
            text = (folder / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows = outline(rel, text)
        out.append(f"## {rel} ({text.count(chr(10)) + 1} lines)")
        out += [f"- {r}" for r in rows]
        out.append("")
    return "\n".join(out).strip() + "\n"


def detect_tests(folder: Path) -> str:
    """How this project runs its tests, when that can be told from its files."""
    pkg = folder / "package.json"
    if pkg.is_file():
        try:
            if json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {}).get("test"):
                return "npm test"
        except (OSError, ValueError):
            pass
    if (folder / "Cargo.toml").is_file():
        return "cargo test"
    if (folder / "go.mod").is_file():
        return "go test ./..."
    if (folder / "pytest.ini").is_file() or (folder / "tests").is_dir() or (folder / "conftest.py").is_file():
        return "python -m pytest -q"
    for build in ("build.gradle", "build.gradle.kts"):
        if (folder / build).is_file():
            return "./gradlew test"
    if (folder / "pom.xml").is_file():
        return "mvn test"
    return ""


# -- the code graph for the folder ----------------------------------------------------------------------------------------------
class ProjectGraph(KnowledgeGraph):
    """The knowledge graph of the folder Pal works in, for every session. A project uploaded in the web chat still gets its own
    graph for that session (and is removed with it); the folder's graph is never reset or pruned by a session."""

    def __init__(self, project_dir: Path, cfg=None, runner=None):
        super().__init__(cfg, runner)
        self.project_ws = Workspace(project_dir, GRAPH_WORKSPACE)
        self._uploading: set[str] = set()

    @property
    def enabled(self) -> bool:
        return True

    def _ws(self, session_id: str) -> Workspace:
        own = Workspace(self.root, session_id)
        return own if session_id in self._uploading or own.exists() else self.project_ws

    def ingest(self, session_id: str, name: str, raw: bytes) -> dict:
        self._uploading.add(session_id)
        try:
            return super().ingest(session_id, name, raw)
        finally:
            self._uploading.discard(session_id)

    def reset(self, session_id: str) -> None:
        Workspace(self.root, session_id).reset()


# -- the whole setup ------------------------------------------------------------------------------------------------------------
@dataclass
class DocResult:
    path: str
    markdown: str = ""        # where the markdown file is
    tokens: int = 0
    converted: bool = False
    error: str = ""


@dataclass
class Report:
    code_files: int = 0
    source_tokens: int = 0
    languages: list[str] = field(default_factory=list)
    docs: list[DocResult] = field(default_factory=list)
    map_tokens: int = 0
    tests: str = ""
    graph: str = "not built"            # "ready: N nodes", "building", "needs graphify", "failed: ...", "no code"
    seconds: float = 0.0
    skipped: int = 0

    @property
    def converted(self) -> list[DocResult]:
        return [d for d in self.docs if d.converted and not d.error]


class Project:
    def __init__(self, folder: Path, home: Path | None = None, runner=None, graph_cfg=None):
        self.folder = Path(folder).resolve()
        self.root = (home or pal_home()) / "projects" / project_id(self.folder)
        self.markdown_dir = self.root / "markdown"
        self.manifest_path = self.root / "project.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.graph = ProjectGraph(self.root / "graph", graph_cfg or {"enabled": True, "dir": str(self.root / "graph-uploads")}, runner)
        self.report = Report()
        self.texts: dict[str, str] = {}          # name -> markdown, what is indexed for every session
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- manifest -------------------------------------------------------------------------------------------------------------
    def _manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_manifest(self, data: dict) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.manifest_path)

    # -- steps ----------------------------------------------------------------------------------------------------------------
    def prepare(self, graph_in_background: bool = True, progress=None) -> Report:
        """Scan, convert documents, write the project map, and build (or update) the code graph. Returns the report."""
        t0 = time.perf_counter()
        say = progress or (lambda _msg: None)
        found = scan(self.folder)
        old = self._manifest()
        r = Report(code_files=len(found.code), skipped=found.skipped, tests=detect_tests(self.folder))
        r.source_tokens = sum(size // 4 for _, size in found.code.values())
        langs = Counter(LANGUAGES[_ext(p)] for p in found.code if _ext(p) in LANGUAGES)
        r.languages = [name for name, _ in langs.most_common(4)]

        say(f"documents: {len(found.docs)}")
        old_docs = old.get("docs", {})
        docs_state = {}
        self.texts = {}
        for rel, stamp in found.docs.items():
            res, state = self._document(rel, stamp, old_docs.get(rel))
            r.docs.append(res)
            docs_state[rel] = state
        for rel in set(old_docs) - set(found.docs):                    # a document that was removed: drop its markdown
            md = old_docs[rel].get("markdown")
            if md:
                Path(md).unlink(missing_ok=True)

        say(f"project map: {len(found.code)} files")
        if found.code:
            text = build_map(self.folder, sorted(found.code))
            (self.root / "project-map.md").write_text(text, encoding="utf-8")
            r.map_tokens = estimate_tokens(text)
            self.texts["project-map.md"] = text
        self.report = r
        self._save_manifest({"folder": str(self.folder), "docs": docs_state, "code": {k: list(v) for k, v in found.code.items()},
                             "graph": old.get("graph", {}), "updated": time.time()})

        if not found.code:
            r.graph = "no code"
        elif not self.graph.available():
            r.graph = "needs graphify"
        else:
            changed = old.get("code") != {k: list(v) for k, v in found.code.items()} or not self.graph.project_ws.exists()
            if not changed:
                r.graph = f"ready: {old.get('graph', {}).get('nodes', 0):,} nodes"
            else:
                r.graph = "building"
                if graph_in_background:
                    self._thread = threading.Thread(target=self._build_graph, args=(found,), daemon=True, name="pal-graph")
                    self._thread.start()
                else:
                    self._build_graph(found)
        r.seconds = time.perf_counter() - t0
        return r

    def _document(self, rel: str, stamp: tuple[int, int], before: dict | None) -> tuple[DocResult, dict]:
        src = self.folder / rel
        ext = _ext(rel)
        md_path = self.markdown_dir / (rel + ".md")
        if before and tuple(before.get("stamp", ())) == tuple(stamp) and before.get("error"):
            return DocResult(rel, error=before["error"]), before          # unchanged since it failed: not tried again
        if before and tuple(before.get("stamp", ())) == tuple(stamp) and (ext in TEXT_DOC_EXTENSIONS or md_path.is_file()):
            try:
                text = md_path.read_text(encoding="utf-8") if ext not in TEXT_DOC_EXTENSIONS else src.read_text(encoding="utf-8", errors="replace")
                self.texts[rel] = text
                return DocResult(rel, str(md_path) if ext not in TEXT_DOC_EXTENSIONS else "", estimate_tokens(text), ext not in TEXT_DOC_EXTENSIONS), before
            except OSError:
                pass
        if ext in TEXT_DOC_EXTENSIONS:
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return DocResult(rel, error=str(e.strerror or e)), {"stamp": list(stamp), "error": "unreadable"}
            self.texts[rel] = text
            return DocResult(rel, "", estimate_tokens(text)), {"stamp": list(stamp)}
        from pal.input_converters import convert_to_markdown
        try:
            text = convert_to_markdown(src.name, src.read_bytes()).text
        except Exception as e:  # noqa: BLE001  (damaged file, a scan with no text, a converter that is missing)
            return DocResult(rel, error=str(e)[:160]), {"stamp": list(stamp), "error": str(e)[:160]}
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text, encoding="utf-8")
        self.texts[rel] = text
        return DocResult(rel, str(md_path), estimate_tokens(text), True), {"stamp": list(stamp), "markdown": str(md_path)}

    def _build_graph(self, found: Scan) -> None:
        with self._lock:
            ws = self.graph.project_ws
            try:
                ws.src.mkdir(parents=True, exist_ok=True)
                keep = set()
                for rel in found.code:                                       # copy what changed; the folder itself is never touched
                    if _ext(rel).lstrip(".") not in CODE_EXTENSIONS:
                        continue
                    src, dst = self.folder / rel, ws.src / rel
                    keep.add(dst)
                    try:
                        if not dst.is_file() or dst.stat().st_size != src.stat().st_size or dst.stat().st_mtime_ns < src.stat().st_mtime_ns:
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)
                    except OSError:
                        continue
                for p in list(ws.src.rglob("*")):                            # files removed from the folder
                    if p.is_file() and "graphify-out" not in p.parts and p not in keep:
                        p.unlink(missing_ok=True)
                info = builder.build(ws, self.folder.name, self.graph.runner)
                self.report.graph = f"ready: {info.nodes:,} nodes"
                data = self._manifest()
                data["graph"] = {"nodes": info.nodes, "edges": info.edges, "files": info.files, "built": time.time()}
                self._save_manifest(data)
            except Exception as e:  # noqa: BLE001  (the graph is a help, never a reason for the session to fail)
                self.report.graph = f"failed: {str(e)[:120]}"

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- for the model --------------------------------------------------------------------------------------------------------
    def notes(self) -> str:
        """PAL.md in the folder: the project's own instructions for Pal (written by you, or started with /init)."""
        f = self.folder / NOTES_FILE
        try:
            return f.read_text(encoding="utf-8", errors="replace").strip()[:MAX_NOTES_CHARS] if f.is_file() else ""
        except OSError:
            return ""

    def brief(self) -> str:
        r = self.report
        parts = [f"Project: {self.folder.name}, {r.code_files} source files" + (f" ({', '.join(r.languages)})" if r.languages else "")]
        if r.tests:
            parts.append(f"Tests run with: {r.tests}")
        if self.texts:
            parts.append("Indexed for you (passages that match each request are added): " + ", ".join(sorted(self.texts)[:12])
                         + (f" and {len(self.texts) - 12} more" if len(self.texts) > 12 else ""))
        notes = self.notes()
        return "[" + ". ".join(parts) + ".]" + (f"\n\nProject instructions ({NOTES_FILE}), follow them:\n{notes}" if notes else "")

    def documents(self) -> list[tuple[str, str]]:
        return sorted(self.texts.items())
