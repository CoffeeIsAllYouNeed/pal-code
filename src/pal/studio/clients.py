"""Moved to pal.terminal.clients; this name is kept so the Studio and older imports keep working."""
import sys

from pal.terminal import clients as _moved

sys.modules[__name__] = _moved
