"""A real build task: the local model re-implements functions that were removed from LocalRAG, and the project's own tests judge the result.

The experiment takes the real LocalRAG project, replaces the body of a few functions with `raise NotImplementedError`, and asks the model to
write them again, one at a time, from the docstring and the rest of the project. Each answer is judged by running LocalRAG's real test
suite with only that one function replaced, so "correct" means the project's tests pass, not that someone liked the code.

SAFETY. The model's code is run (by the tests) on this machine. Before that it must pass a static filter (no file, process or network
access, no eval or exec, only a few harmless imports) and it runs in a temporary copy of the project with a time limit. That is a guard
against accidents, not a sandbox: do not point this at a model or a prompt you do not trust.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from experiments.demo.scenario import PROJECT_ROOT, SRC, load_source

ALL = ("core", "embed", "tree", "index", "retrieve", "ingest", "mcp", "eval", "cli")


@dataclass(frozen=True)
class Task:
    name: str            # the function to write
    path: str            # its file, inside localrag/


TASKS = [Task("reciprocal_rank", "eval/metrics.py"), Task("recall_at_k", "eval/metrics.py"), Task("precision_at_k", "eval/metrics.py"),
         Task("cosine", "embed/similarity.py"), Task("jaccard", "ingest/dedupe.py"), Task("min_max", "index/fusion.py"),
         Task("reciprocal_rank_fusion", "index/fusion.py"), Task("weighted_fusion", "index/fusion.py"), Task("average_precision", "eval/metrics.py"),
         Task("ndcg_at_k", "eval/metrics.py"), Task("centroid", "embed/similarity.py"), Task("mmr_select", "retrieve/mmr.py")]

# how many functions the model must write, and which parts of the project it is shown
BUILD_SIZES = {"small": dict(count=4, packages=("core", "embed", "eval")),
               "big": dict(count=9, packages=("core", "embed", "eval", "index", "ingest")),
               "max": dict(count=12, packages=ALL)}

ALLOWED_IMPORTS = {"math", "re", "heapq", "itertools", "functools", "collections", "typing", "operator", "statistics", "__future__", "localrag"}
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}


# -- finding, stubbing and replacing a function -----------------------------------------------------------------------------------------
def find_function(code: str, name: str) -> ast.FunctionDef:
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"no top-level function {name}")


def param_names(node: ast.FunctionDef) -> list:
    a = node.args
    return [x.arg for x in a.posonlyargs + a.args] + ([a.vararg.arg] if a.vararg else []) + [x.arg for x in a.kwonlyargs] + ([a.kwarg.arg] if a.kwarg else [])


def stub_of(code: str, name: str) -> str:
    """The function as a stub: its signature and docstring, with the body replaced by raise NotImplementedError."""
    node, lines = find_function(code, name), code.splitlines(keepends=True)
    first = node.body[0]
    has_doc = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    header = lines[node.lineno - 1:first.lineno - 1]
    doc = lines[first.lineno - 1:first.end_lineno] if has_doc else []
    return "".join(header + doc) + '    raise NotImplementedError("implement this function")\n'


def import_position(code: str, before_line: int) -> int:
    """Where new imports belong: just after the last top-level import above `before_line`, else after the module docstring, else the top.
    (An import dropped into the middle of a file breaks it: `from __future__ import annotations` is only legal at the very top.)"""
    pos = 0
    for i, node in enumerate(ast.parse(code).body):
        if node.lineno >= before_line:
            break
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pos = node.end_lineno or node.lineno
        elif i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pos = node.end_lineno or node.lineno
    return pos


def replace_function(code: str, name: str, new_def: str, imports: list | None = None) -> str:
    """The code with the function's definition replaced by new_def. Imports it brought are added at the top of the module, once, after the existing
    ones; `from __future__` lines are dropped (the files that need one already have it)."""
    node, lines = find_function(code, name), code.splitlines(keepends=True)
    present = {ast.get_source_segment(code, n) for n in ast.parse(code).body if isinstance(n, (ast.Import, ast.ImportFrom))}
    wanted = [i.strip() for i in dict.fromkeys(imports or []) if not i.strip().startswith("from __future__") and i.strip() not in present]
    out = lines[:node.lineno - 1] + [new_def.rstrip("\n") + "\n"] + lines[node.end_lineno:]
    if wanted:
        at = import_position(code, node.lineno)
        out[at:at] = [f"{imp}\n" for imp in wanted]
    return "".join(out)


def apply_stub(code: str, name: str) -> str:
    return replace_function(code, name, stub_of(code, name).rstrip("\n"))


def stubbed_source(tasks: list, packages: tuple) -> dict:
    """The source files of the shown part of the project, with the functions to be written stubbed out: what the model gets to see."""
    files = load_source(packages, packages == ALL)
    for t in tasks:
        if t.path in files:
            files[t.path] = apply_stub(files[t.path], t.name)
    return files


# -- reading the model's reply -----------------------------------------------------------------------------------------------------------
@dataclass
class Extraction:
    ok: bool
    definition: str = ""
    imports: tuple = ()
    error: str = ""


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract(reply: str, name: str, expected_params: list | None = None) -> Extraction:
    """The function `name` from a reply, with the imports that sit beside it. Says exactly what is wrong when it cannot."""
    error = "no code found: reply with the complete function in one ```python block"
    bare = [reply] if re.search(r"^\s*(def|import|from|@)\b", reply, re.M) else []      # an unfenced reply counts only if it looks like code
    for block in (_FENCE.findall(reply) or bare):
        try:
            tree = ast.parse(block)
        except SyntaxError as e:
            error = f"syntax error: {e.msg} (line {e.lineno})"
            continue
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if node is None:
            error = f"the reply does not define a function named {name}"
            continue
        if expected_params is not None and param_names(node) != expected_params:
            error = f"the signature must keep the parameters {expected_params}, not {param_names(node)}"
            continue
        imports = tuple(seg for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for seg in [ast.get_source_segment(block, n)] if seg)
        return Extraction(True, ast.get_source_segment(block, node) or "", imports)
    return Extraction(False, error=error)


def unsafe_reason(definition: str, imports: tuple = ()) -> str:
    """Why this code may not be run, or an empty string. A best-effort filter, not a sandbox."""
    try:
        tree = ast.parse("\n".join(imports) + "\n" + definition)
    except SyntaxError as e:
        return f"syntax error: {e.msg}"
    for node in ast.walk(tree):
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for mod in mods:
            if mod.split(".")[0] not in ALLOWED_IMPORTS:
                return f"importing {mod} is not allowed here (allowed: {', '.join(sorted(ALLOWED_IMPORTS - {'__future__'}))})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            return f"calling {node.func.id} is not allowed here"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            return f"using {node.attr} is not allowed here"
    return ""


# -- judging it ---------------------------------------------------------------------------------------------------------------------------
@dataclass
class TestRun:
    __test__ = False                 # not a pytest test class
    ok: bool
    passed: int
    failed: int
    tail: str                        # the end of pytest's output: what to show the model when it failed
    ran: bool = True                 # False when pytest printed no result at all (a crash, nothing collected): "0 passed, 0 failed" would be a lie
    output: str = ""                 # the last 20,000 characters of pytest's output, saved to a file so a surprise can be investigated


def copy_project(dest: Path) -> Path:
    shutil.copytree(PROJECT_ROOT, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".localrag", "*.pyc"))
    return dest


SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")


def safe_env(project: Path) -> dict:
    """The environment the model's code is tested in: this machine's own (Windows needs SYSTEMROOT and friends just to start Python), minus anything
    that looks like a secret, so an API key in your environment can never be seen by code a model wrote."""
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in SECRET_HINTS)}
    env.update(PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


def pytest_available() -> bool:
    return importlib.util.find_spec("pytest") is not None


def run_tests(project: Path, fail_fast: bool = True, timeout: int = 120) -> TestRun:
    """LocalRAG's own test suite, run on a copy of the project."""
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", *(["-x"] if fail_fast else [])]
    try:
        done = subprocess.run(cmd, cwd=project, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=safe_env(project))
    except subprocess.TimeoutExpired:
        msg = f"the tests did not finish within {timeout} seconds (an infinite loop?)"
        return TestRun(False, 0, 1, msg, True, msg)
    out = done.stdout + done.stderr
    if "No module named pytest" in out:
        msg = f"pytest is not installed for this Python. Install it with:  {sys.executable} -m pip install pytest"
        return TestRun(False, 0, 0, msg, False, msg)
    passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1])
    failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1]) + int((re.search(r"(\d+) errors?\b", out) or [0, 0])[1])
    if not re.search(r"\b\d+ (passed|failed|errors?)\b", out):          # no result line: say so instead of reporting zeros
        msg = f"pytest printed no test result (exit code {done.returncode}). Its output was:\n{out.strip()[-1200:] or '(nothing)'}"
        return TestRun(False, 0, 0, msg, False, out.strip()[-20000:] or msg)
    return TestRun(done.returncode == 0, passed, failed, out.strip()[-1400:], True, out.strip()[-20000:])


def focus_failure(output: str, limit: int = 14) -> str:
    """The lines of pytest's output that say what went wrong (the E lines, FAILED lines and syntax errors), so a model is shown the reason, not a traceback."""
    keep = [ln.rstrip() for ln in output.splitlines() if ln.startswith(("E ", "E\t", "FAILED", "ERROR")) or "SyntaxError" in ln or "ImportError" in ln or "NameError" in ln]
    return "\n".join(list(dict.fromkeys(keep))[:limit]) or output[-900:]


def check_candidate(task: Task, definition: str, imports: tuple = (), timeout: int = 120) -> TestRun:
    """Does the project's test suite pass with only this one function replaced by the model's version?"""
    tmp = Path(tempfile.mkdtemp(prefix="pal-build-"))
    try:
        project = copy_project(tmp / "project")
        path = project / "src" / "localrag" / task.path
        path.write_text(replace_function(path.read_text(encoding="utf-8"), task.name, definition, list(imports)), encoding="utf-8")
        return run_tests(project, timeout=timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def reference_definition(task: Task) -> str:
    """The original function, as written in the project (used to check the tasks themselves, never shown to the model)."""
    code = (SRC / task.path).read_text(encoding="utf-8")
    return ast.get_source_segment(code, find_function(code, task.name)) or ""


def task_message(task: Task, stub: str) -> str:
    return (f"Implement the Python function `{task.name}` in `localrag/{task.path}` of the project. Here is its current stub:\n\n```python\n{stub}```\n\n"
            "Reply with the complete function (the `def` line, the docstring and the body) in one ```python block and nothing else. Keep the "
            "signature exactly. Use only the standard library. If you need an import, put it above the function in the same block.")


def retry_message(problem: str) -> str:
    return (f"That version did not work:\n\n{problem}\n\nReply with the corrected complete function in one ```python block and nothing else.")
