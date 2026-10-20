"""Moved to pal.terminal.files; this name is kept so the Studio and older imports keep working."""
import sys

from pal.terminal import files as _moved

sys.modules[__name__] = _moved
