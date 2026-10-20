"""One command to run the demo experiment:  python experiments/run_demo.py  [--size small|big|max] [--simulate | --real] ..."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.demo.cli import main  # noqa: E402

for _stream in (sys.stdout, sys.stderr):      # a Windows console may not be able to print every character; never crash on that
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
