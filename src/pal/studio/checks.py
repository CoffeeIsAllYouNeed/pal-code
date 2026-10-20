"""Static checks on the files a build wrote. Nothing is run: a file is read, never executed.

These catch the mistakes small models make most (a half-closed tag, a missing brace, a page that links to a file that was never written). A problem is
reported in plain words so it can be sent back to the model once for a fix.
"""
from __future__ import annotations

import ast
import json
import re
from html.parser import HTMLParser
from pathlib import PurePosixPath

_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})
_TRACKED = frozenset({"html", "head", "body", "script", "style", "div", "section", "nav", "main", "header", "footer", "form", "table", "ul", "ol", "button"})
_PAIRS = {"{": "}", "(": ")", "[": "]"}


class _Html(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.problems: list[str] = []
        self.refs: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for key in ("src", "href"):
            v = a.get(key)
            if v and tag in ("script", "link", "img", "source") and not re.match(r"^([a-z][a-z0-9+.-]*:|//|#|data:)", v, re.I):
                self.refs.append(v.split("#")[0].split("?")[0])
        if tag in _TRACKED:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag in _TRACKED and self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in _VOID or tag not in _TRACKED:
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.problems.append(f"<{self.stack.pop()}> is never closed")
            self.stack.pop()
        else:
            self.problems.append(f"</{tag}> closes something that was never opened")


def _balance(code: str, comments: bool = True) -> list[str]:
    """Brackets that do not pair up, ignoring strings and comments (a lightweight scan, not a parser)."""
    stack: list[str] = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if comments and code.startswith("//", i):
            i = code.find("\n", i)
            i = n if i < 0 else i
            continue
        if comments and code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "\"'`":
            j = i + 1
            while j < n and code[j] != c:
                j += 2 if code[j] == "\\" else 1
                if c != "`" and j < n and code[j] == "\n":
                    break
            i = j + 1
            continue
        if c in _PAIRS:
            stack.append(c)
        elif c in _PAIRS.values():
            if not stack or _PAIRS[stack[-1]] != c:
                return [f"a '{c}' has no matching opener"]
            stack.pop()
        i += 1
    return [f"a '{stack[-1]}' is never closed"] if stack else []


def check_file(path: str, content: str) -> list[str]:
    ext = PurePosixPath(path).suffix.lower()
    if ext == ".py":
        try:
            ast.parse(content)
        except SyntaxError as e:
            return [f"syntax error on line {e.lineno}: {e.msg}"]
    elif ext == ".json":
        try:
            json.loads(content)
        except ValueError as e:
            return [f"not valid JSON: {e}"]
    elif ext in (".html", ".htm"):
        p = _Html()
        p.feed(content)
        p.close()
        return p.problems[:3] + [f"<{t}> is never closed" for t in p.stack[:3]]
    elif ext in (".js", ".mjs", ".ts", ".tsx", ".jsx", ".css"):
        return _balance(content, comments=True)
    return []


def check_project(files: dict[str, str], only: set[str] | None = None, links: bool = True) -> dict[str, list[str]]:
    """Problems by file. `only` limits the single-file checks to files that just changed. `links` also checks that every file a page links to was
    written: leave it off in the middle of a build, because a page may link to a file a later step is about to write."""
    out: dict[str, list[str]] = {}
    lower = {p.lower() for p in files}
    for path, content in files.items():
        problems = check_file(path, content) if only is None or path in only else []
        if links and PurePosixPath(path).suffix.lower() in (".html", ".htm"):
            p = _Html()
            p.feed(content)
            base = PurePosixPath(path).parent
            for ref in p.refs:
                target = (base / ref).as_posix() if str(base) != "." else ref
                parts: list[str] = []
                for seg in target.split("/"):
                    if seg == "..":
                        if parts:
                            parts.pop()
                    elif seg not in ("", "."):
                        parts.append(seg)
                if "/".join(parts).lower() not in lower:
                    problems.append(f"it refers to {ref}, but no file with that name was written")
        if problems:
            out[path] = problems
    return out
