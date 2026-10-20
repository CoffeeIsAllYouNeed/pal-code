"""Per-question retrieval: ask graphify which nodes matter, then attach small
code excerpts for them, all within a token budget."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable, Optional

from pal.knowledge_graph.components.base import GraphExcerpts, Runner
from pal.knowledge_graph.components.workspace import Workspace

_NODE = re.compile(r"^NODE (.+) \[src=(.+?) loc=L(\d+)(?: [^\]]*)?\]\s*$")
_QUESTION_CHARS = 300


def parse_nodes(stdout: str) -> list:
    """[(label, src, line)] in the order graphify ranked them."""
    out, seen = [], set()
    for line in stdout.splitlines():
        m = _NODE.match(line.strip())
        if not m:
            continue
        key = (m.group(2), int(m.group(3)))
        if key in seen:
            continue
        seen.add(key)
        out.append((m.group(1), m.group(2), int(m.group(3))))
    return out


def _py_span(text: str, start: int) -> Optional[int]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            if node.lineno == start or first == start:
                return node.end_lineno
    return None


def excerpt(path: Path, start: int, next_start: Optional[int], max_lines: int) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if start < 1 or start > len(lines):
        return None
    end = _py_span(text, start) if path.suffix == ".py" else None
    if end is None:
        end = (next_start - 1) if next_start and next_start > start else start + max_lines - 1
    end = min(end, start + max_lines - 1, len(lines))
    body = "\n".join(lines[start - 1:end])
    return body + ("\n..." if end < (_py_span(text, start) or end) else "")


def _is_file_node(label: str, line: int) -> bool:
    return line == 1 and bool(re.search(r"\.\w{1,5}$", label))


def retrieve(ws: Workspace, question: str, runner: Runner, count_tokens: Callable[[str], int],
             budget_tokens: int, max_slice_lines: int, max_nodes: int = 40) -> Optional[GraphExcerpts]:
    code, out, _ = runner(["query", question.strip()[:_QUESTION_CHARS], "--budget", str(max(400, budget_tokens)),
                           "--graph", str(ws.graph_path)], str(ws.dir))
    if code != 0:
        return None
    nodes = parse_nodes(out)[:max_nodes]
    if not nodes:
        return None
    meta = ws.load_meta()
    map_lines = [f"- {label}  {src.removeprefix('src/')}:{line}" for label, src, line in nodes]
    head = "<knowledge_graph>\nRelevant nodes (most relevant first):\n" + "\n".join(map_lines)
    tail = "\n</knowledge_graph>"
    used = count_tokens(head + tail)

    by_file: dict = {}
    for _, src, line in nodes:
        by_file.setdefault(src, []).append(line)
    blocks, n_ex = [], 0
    for label, src, line in nodes:
        if _is_file_node(label, line):
            continue
        later = sorted(n for n in by_file[src] if n > line)
        body = excerpt(ws.dir / src, line, later[0] if later else None, max_slice_lines)
        if not body:
            continue
        block = f"\n### {label}  {src.removeprefix('src/')}:{line}\n```\n{body}\n```"
        cost = count_tokens(block)
        if used + cost > budget_tokens:
            continue
        blocks.append(block)
        used += cost
        n_ex += 1
    text = head + ("\n\nExcerpts:" + "".join(blocks) if blocks else "") + tail
    return GraphExcerpts(text=text, nodes=len(nodes), excerpts=n_ex, tokens=count_tokens(text),
                         project_tokens=int(meta.get("source_tokens", 0)))
