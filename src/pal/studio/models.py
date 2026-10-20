"""Moved to pal.terminal.models; this name is kept so the Studio and older imports keep working."""
import sys

from pal.terminal import models as _moved

sys.modules[__name__] = _moved
