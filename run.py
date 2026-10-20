#!/usr/bin/env python3
"""One command to run Pal:   python run.py        (Windows: py run.py)

First run: creates .venv, installs Pal and its extras, then opens the web interface in your browser.
Later runs: skips straight to starting. `python run.py cli` opens the terminal session instead. Other arguments go to the server (e.g. --port 8080).
Reads API keys from the environment or a .env file next to this script (see .env.example).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
EXTRAS = ("full", "documents")  # tried in this order; the second skips graphify if it will not install


def venv_python(venv_dir: Path = VENV, platform: str = sys.platform) -> Path:
    return venv_dir / ("Scripts/python.exe" if platform == "win32" else "bin/python")


LOCK = Path("requirements") / "litellm.lock"


def stamp_value(root: Path | None = None) -> str:
    """Changes when the project or the hash-locked LiteLLM pin changes, so the next run reinstalls."""
    base = root or ROOT
    h = hashlib.sha256((base / "pyproject.toml").read_bytes())
    if (base / LOCK).is_file():
        h.update((base / LOCK).read_bytes())
    return h.hexdigest()[:16]


def needs_install(stamp_file: Path, root: Path | None = None) -> bool:
    return not stamp_file.is_file() or stamp_file.read_text(encoding="utf-8").strip() != stamp_value(root)


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=ROOT)  # noqa: S603


def _has_module(py: Path, name: str) -> bool:
    """Can this Python import the module? False if the Python cannot even be started."""
    try:
        return subprocess.run([str(py), "-c", f"import {name}"], capture_output=True).returncode == 0
    except OSError:
        return False


def ensure_venv() -> Path:
    py = venv_python()
    if not py.is_file():
        print("Creating .venv (first run only)...")
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    return py


def ensure_installed(py: Path) -> None:
    stamp = VENV / ".pal-installed"
    if not needs_install(stamp):
        return
    print("Installing Pal (first run, or the project changed). This takes a minute...")
    _run([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    # LiteLLM handles users' API keys, so it is installed FIRST from the hash-locked file: pip refuses any package whose hash
    # differs from the one recorded when the pin was reviewed. If that cannot work on this platform, the exact pin below is
    # installed instead (without hash checking) and you are told.
    if (ROOT / LOCK).is_file() and _run([str(py), "-m", "pip", "install", "--quiet", "--require-hashes", "-r", str(LOCK)]) != 0:
        print("Note: the hash-verified LiteLLM install did not work on this platform; the exact pinned version is installed "
              "without hash checking.")
    for extra in EXTRAS:
        if _run([str(py), "-m", "pip", "install", "--quiet", "-e", f".[{extra}]"]) == 0:
            stamp.write_text(stamp_value(), encoding="utf-8")
            if extra != EXTRAS[0]:
                print(f"Note: installed with [{extra}] only; code-project graphs (graphify) are unavailable.")
            return
    sys.exit("Install failed. Run `python -m pip install -e .` yourself to see the error.")


def main(argv: list[str]) -> int:
    if sys.version_info < (3, 10):  # noqa: UP036
        sys.exit(f"Python 3.10 or newer is required (this is {sys.version_info.major}.{sys.version_info.minor}).")
    if not (ROOT / "pyproject.toml").is_file():
        sys.exit("Run this from the Pal folder.")
    in_own_venv = Path(sys.prefix).resolve() == VENV.resolve()
    if in_own_venv or sys.prefix != sys.base_prefix:  # already inside a virtualenv: use it as it is
        py = Path(sys.executable)
    else:
        py = ensure_venv()
    ensure_installed(py)
    if argv and argv[0] in ("experiment", "suite"):      # the demo experiments: not part of the app, one command
        if not _has_module(py, "pytest"):     # the build experiment judges the model's code with pytest
            print("Installing pytest (the experiments use it to judge code) ...")
            _run([str(py), "-m", "pip", "install", "--quiet", "pytest"])
        if argv[0] == "suite":                # 30 experiments (python run.py suite): needs matplotlib for the charts, psutil for memory
            for mod, pkg in (("matplotlib", "matplotlib"), ("psutil", "psutil")):
                if not _has_module(py, mod):
                    print(f"Installing {pkg} (the suite uses it) ...")
                    _run([str(py), "-m", "pip", "install", "--quiet", pkg])
            return _run([str(py), str(ROOT / "experiments" / "run_demo.py"), "suite", *argv[1:]])
        return _run([str(py), str(ROOT / "experiments" / "run_demo.py"), *argv[1:]])
    if argv and argv[0] == "cli":             # the terminal session
        return _run([str(py), "-m", "pal.terminal.cli", "cli", *argv[1:]])
    if argv and argv[0] == "studio":          # the Studio: describe a product, review the plan, watch it being built
        return _run([str(py), "-m", "pal", "serve", "--open", "--page", "studio", *argv[1:]])
    return _run([str(py), "-m", "pal", "serve", "--open", *argv])


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(0) from None
