"""Does a file the model wrote parse? Checked before you review a change, so a missing bracket is fixed by the Build agent, not found by you.

Python, JSON, YAML and TOML are checked with their own parsers. JavaScript, TypeScript, Go, Rust, Java, Kotlin, C, C++, C#, Ruby, PHP, Swift,
Scala, Lua and shell use tree-sitter (installed with Pal for the code graph). Other files are not checked. A check never runs the code.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath

# extension -> (tree-sitter module, the function that returns its language)
TREE_SITTER = {
    ".js": ("tree_sitter_javascript", "language"), ".jsx": ("tree_sitter_javascript", "language"), ".mjs": ("tree_sitter_javascript", "language"),
    ".cjs": ("tree_sitter_javascript", "language"), ".ts": ("tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree_sitter_typescript", "language_tsx"), ".go": ("tree_sitter_go", "language"), ".rs": ("tree_sitter_rust", "language"),
    ".java": ("tree_sitter_java", "language"), ".kt": ("tree_sitter_kotlin", "language"), ".kts": ("tree_sitter_kotlin", "language"),
    ".c": ("tree_sitter_c", "language"), ".h": ("tree_sitter_c", "language"), ".cpp": ("tree_sitter_cpp", "language"),
    ".cc": ("tree_sitter_cpp", "language"), ".hpp": ("tree_sitter_cpp", "language"), ".cs": ("tree_sitter_c_sharp", "language"),
    ".rb": ("tree_sitter_ruby", "language"), ".php": ("tree_sitter_php", "language_php"), ".swift": ("tree_sitter_swift", "language"),
    ".scala": ("tree_sitter_scala", "language"), ".lua": ("tree_sitter_lua", "language"), ".sh": ("tree_sitter_bash", "language"),
    ".bash": ("tree_sitter_bash", "language"),
}


@dataclass(frozen=True)
class SyntaxProblem:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path} line {self.line}: {self.message}"


@lru_cache(maxsize=32)
def _parser(ext: str):
    """A tree-sitter parser for the extension, or None when the grammar is not installed."""
    mod_name, fn = TREE_SITTER[ext]
    try:
        from tree_sitter import Language, Parser
        mod = importlib.import_module(mod_name)
        return Parser(Language(getattr(mod, fn)()))
    except Exception:  # noqa: BLE001  (grammar missing or an incompatible version: the file is just not checked)
        return None


def _first_error(node):
    """The innermost ERROR (or missing) node on the first broken path through the tree."""
    found = None
    while node is not None:
        if node.type == "ERROR" or node.is_missing:
            found = node
        nxt = next((c for c in node.children if c.has_error or c.is_missing or c.type == "ERROR"), None)
        node = nxt
    return found


def check(path: str, text: str) -> SyntaxProblem | None:
    """The first syntax problem in the text, or None when it parses (or the language is not checked)."""
    ext = PurePosixPath(path).suffix.lower()
    try:
        if ext in (".py", ".pyw"):
            compile(text, path, "exec", dont_inherit=True)
        elif ext == ".json":
            json.loads(text)
        elif ext in (".yaml", ".yml"):
            import yaml
            list(yaml.safe_load_all(text))
        elif ext == ".toml":
            try:
                import tomllib
            except ImportError:                         # Python 3.10: not checked
                return None
            tomllib.loads(text)
        elif ext in TREE_SITTER:
            parser = _parser(ext)
            if parser is None:
                return None
            tree = parser.parse(text.encode("utf-8"))
            if not tree.root_node.has_error:
                return None
            bad = _first_error(tree.root_node) or tree.root_node
            what = f"missing {bad.type}" if bad.is_missing else "unexpected code"
            return SyntaxProblem(path, bad.start_point[0] + 1, what)
        else:
            return None
    except SyntaxError as e:
        return SyntaxProblem(path, e.lineno or 1, e.msg)
    except json.JSONDecodeError as e:
        return SyntaxProblem(path, e.lineno, e.msg)
    except Exception as e:  # noqa: BLE001  (yaml and toml errors carry their own types)
        mark = getattr(e, "problem_mark", None)
        line = (mark.line + 1) if mark is not None else getattr(e, "lineno", 1) or 1
        return SyntaxProblem(path, line, str(getattr(e, "problem", None) or e).splitlines()[0][:160])
    return None
