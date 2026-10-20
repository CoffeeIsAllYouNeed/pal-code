#!/usr/bin/env bash
# Linux / macOS / Git Bash: installs Pal in its own virtual environment, then checks the computer.
#   ./install.sh          install pal-code from PyPI
#   ./install.sh --dev    install this source folder in editable mode, with the test tools
set -e
PY=$(command -v python3 || command -v python)
[ -d venv ] || "$PY" -m venv venv
if [ -f venv/bin/activate ]; then . venv/bin/activate; else . venv/Scripts/activate; fi
python -m pip install --upgrade pip
if [ "$1" = "--dev" ]; then python -m pip install -e ".[dev,full]"; else python -m pip install pal-code; fi
pal doctor || true
echo
echo "Run, in the folder you want to build in:  pal cli    (or  pal web  for the browser)"
