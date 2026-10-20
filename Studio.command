#!/bin/sh
# Double-click (Mac) or run (Linux) to open the Pal Studio. The first run installs what it needs (a few minutes).
cd "$(dirname "$0")" || exit 1
python3 run.py studio
