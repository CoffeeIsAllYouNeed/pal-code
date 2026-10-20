"""Logging setup for entry points. Library code only does `logging.getLogger(__name__)`; only the
CLI/server calls configure_logging(). Secrets are never logged: API keys never enter a log call."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("pal")
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
