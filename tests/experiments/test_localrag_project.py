"""LocalRAG is the workload of the demo experiment, so it must really be what the report says it is: a real, working, sizeable project."""
import os
import re
import subprocess
import sys
from pathlib import Path

from pal.tokens import estimate_tokens

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "experiments" / "projects" / "localrag"
SRC = PROJECT / "src" / "localrag"


def test_localrag_is_a_sizeable_project_with_documentation():
    files = [p for p in SRC.rglob("*.py") if p.stat().st_size]
    text = "".join(p.read_text(encoding="utf-8") for p in files)
    assert len(files) >= 70 and text.count("\n") >= 3_000 and estimate_tokens(text) >= 30_000
    assert {p.name for p in (PROJECT / "docs").glob("*.md")} == {"ARCHITECTURE.md", "CONFIG.md", "MCP.md", "RETRIEVAL.md"} and (PROJECT / "README.md").is_file()
    assert {d.name for d in SRC.iterdir() if d.is_dir() and d.name != "__pycache__"} == {"core", "ingest", "embed", "index", "retrieve", "tree", "mcp", "eval", "cli"}


def test_localrag_has_no_dependencies_and_does_not_use_pal_nor_the_reverse():
    assert 'dependencies = []' in (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    assert not any(re.search(r"(?m)^\s*(from|import)\s+pal", p.read_text(encoding="utf-8")) for p in SRC.rglob("*.py"))
    assert not any(re.search(r"(?m)^\s*(from|import)\s+localrag", p.read_text(encoding="utf-8")) for p in (ROOT / "src").rglob("*.py"))


def test_localrags_own_test_suite_passes():
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=PROJECT, capture_output=True, text=True, timeout=300)
    passed = int(re.search(r"(\d+) passed", done.stdout).group(1))
    assert done.returncode == 0 and passed >= 90 and "failed" not in done.stdout, done.stdout[-800:]


def test_every_module_imports_and_the_command_line_runs():
    code = ("import importlib, pkgutil, localrag\n"
            "names = [m.name for m in pkgutil.walk_packages(localrag.__path__, 'localrag.')]\n"
            "[importlib.import_module(n) for n in names]\nprint(len(names))")
    env = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0 and int(done.stdout) >= 70, done.stderr[-500:]
    version = subprocess.run([sys.executable, "-m", "localrag", "--version"], capture_output=True, text=True, env=env, timeout=60)
    assert version.returncode == 0 and version.stdout.startswith("localrag ")
