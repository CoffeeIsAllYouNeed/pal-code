"""Build the graph with graphify. `graphify update` is AST-only: no LLM call,
no network, nothing leaves the machine."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter

from pal.knowledge_graph.components.base import GraphError, ProjectInfo, Runner
from pal.knowledge_graph.components.workspace import Workspace

BUILD_TIMEOUT_S = 300


def graphify_available() -> bool:
    return importlib.util.find_spec("graphify") is not None


def default_runner(args: list, cwd: str, timeout: int = 60) -> tuple:
    try:
        p = subprocess.run([sys.executable, "-m", "graphify", *args], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"graphify timed out after {timeout}s"
    return p.returncode, p.stdout, p.stderr


def build(ws: Workspace, name: str, runner: Runner) -> ProjectInfo:
    code, out, err = runner(["update", "src"], str(ws.dir))
    if code != 0 or not ws.graph_path.is_file():
        tail = (err or out).strip().splitlines()
        raise GraphError(f"graphify could not build the graph ({tail[-1] if tail else 'no output'})")
    try:
        g = json.loads(ws.graph_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise GraphError("graphify wrote an unreadable graph.json") from None
    files = ws.source_files()
    source_tokens = sum(max(1, p.stat().st_size // 4) for p in files)
    top = Counter(p.relative_to(ws.src).parts[0] for p in files if len(p.relative_to(ws.src).parts) > 1)
    hubs = []
    code, out, _ = runner(["god-nodes", "--top", "8", "--json", "--graph", str(ws.graph_path)], str(ws.dir))
    if code == 0:
        try:
            hubs = [h["label"] for h in json.loads(out)][:8]
        except (ValueError, KeyError, TypeError):
            pass
    info = ProjectInfo(name=name, files=len(files), nodes=len(g.get("nodes", [])),
                       edges=len(g.get("links", g.get("edges", []))), source_tokens=source_tokens,
                       hubs=hubs, top_dirs=top.most_common(8))
    ws.save_meta({"name": name, "files": info.files, "nodes": info.nodes, "edges": info.edges,
                  "source_tokens": info.source_tokens})
    return info
