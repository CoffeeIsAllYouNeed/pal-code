"""Logging setup: one named logger per module, quiet by default, switched up with LOCALRAG_LOG=debug."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """The logger for a module; the first call configures the root handler from the LOCALRAG_LOG environment variable."""
    global _CONFIGURED
    if not _CONFIGURED:
        level = getattr(logging, os.environ.get("LOCALRAG_LOG", "warning").upper(), logging.WARNING)
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        _CONFIGURED = True
    return logging.getLogger(name)
