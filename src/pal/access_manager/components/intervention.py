"""
User intervention interface (AIOS section 3.8): irreversible operations
(deletion, overwrite, privilege modification) require explicit user
confirmation before proceeding, even when privilege-based access
control alone would permit them.

The actual UI a real deployment asks the user through (a CLI prompt, a
web confirmation dialog, ...) is out of scope for this project -- what
this implements is the gate itself: a pluggable confirmation_handler
callback the surrounding application registers. The default handler
denies everything, safely, rather than silently approving destructive
operations when no real confirmation UI has been wired up.
"""
from typing import Callable, Optional

from pal.access_manager.components.base import AccessRequest


def _default_deny_handler(request: AccessRequest) -> bool:
    return False


class UserInterventionGate:
    def __init__(self, confirmation_handler: Optional[Callable[[AccessRequest], bool]] = None):
        self.confirmation_handler = confirmation_handler or _default_deny_handler

    def set_handler(self, handler: Callable[[AccessRequest], bool]) -> None:
        self.confirmation_handler = handler

    def confirm(self, request: AccessRequest) -> bool:
        return bool(self.confirmation_handler(request))
