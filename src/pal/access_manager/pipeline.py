"""
Access Manager (AIOS section 3.8): privilege-based access control for
cross-agent resource access (components/privilege_registry.py,
components/access_control.py), plus a user-intervention gate for
irreversible operations (components/intervention.py).
"""
from typing import Callable, Optional

from pal.access_manager.components.access_control import AccessController
from pal.access_manager.components.base import AccessDecision, AccessRequest, OperationType
from pal.access_manager.components.intervention import UserInterventionGate
from pal.access_manager.components.privilege_registry import PrivilegeRegistry


class AccessManagerPipeline:
    def __init__(self, confirmation_handler: Optional[Callable[[AccessRequest], bool]] = None):
        self.registry = PrivilegeRegistry()
        self.controller = AccessController(self.registry)
        self.intervention = UserInterventionGate(confirmation_handler)

    def assign_privilege_group(self, agent_id: str, group: str) -> None:
        self.registry.assign(agent_id, group)

    def set_confirmation_handler(self, handler: Callable[[AccessRequest], bool]) -> None:
        self.intervention.set_handler(handler)

    def check_access(
        self, requesting_agent_id: str, target_agent_id: str, operation: OperationType, resource: str = ""
    ) -> AccessDecision:
        """Privilege check only -- does not trigger user intervention,
        even for a destructive operation. Use authorize() for the full
        gate."""
        request = AccessRequest(
            requesting_agent_id=requesting_agent_id,
            target_agent_id=target_agent_id,
            operation=operation,
            resource=resource,
        )
        return self.controller.check(request)

    def authorize(
        self, requesting_agent_id: str, target_agent_id: str, operation: OperationType, resource: str = ""
    ) -> AccessDecision:
        """Full authorization: privilege check first, then -- only if
        that passed AND the operation is destructive -- user
        confirmation. Returns one AccessDecision reflecting both
        checks: a privilege-permitted destructive op the user declines
        to confirm comes back allowed=False, not the privilege-only
        result."""
        request = AccessRequest(
            requesting_agent_id=requesting_agent_id,
            target_agent_id=target_agent_id,
            operation=operation,
            resource=resource,
        )
        decision = self.controller.check(request)

        if not decision.allowed:
            return decision

        if decision.requires_confirmation and not self.intervention.confirm(request):
            return AccessDecision(
                allowed=False,
                requires_confirmation=True,
                reason="user declined confirmation for an irreversible operation",
            )

        return decision
