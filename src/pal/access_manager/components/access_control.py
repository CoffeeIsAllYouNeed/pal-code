"""
Privilege-based access control (AIOS section 3.8): validates a
cross-agent access request against the privilege-group hashmap BEFORE
execution -- an agent may only access another agent's resources if
both are assigned to the same shared privilege domain. An agent acting
on its OWN resources is always permitted regardless of privilege group
(3.8 is specifically about cross-agent data access; a resource owner
always has access to what it owns).
"""
from pal.access_manager.components.base import DESTRUCTIVE_OPS, AccessDecision, AccessRequest
from pal.access_manager.components.privilege_registry import PrivilegeRegistry


class AccessController:
    def __init__(self, registry: PrivilegeRegistry):
        self.registry = registry

    def check(self, request: AccessRequest) -> AccessDecision:
        requires_confirmation = request.operation in DESTRUCTIVE_OPS

        if request.requesting_agent_id == request.target_agent_id:
            return AccessDecision(allowed=True, requires_confirmation=requires_confirmation)

        if self.registry.same_group(request.requesting_agent_id, request.target_agent_id):
            return AccessDecision(allowed=True, requires_confirmation=requires_confirmation)

        group_a = self.registry.group_of(request.requesting_agent_id)
        group_b = self.registry.group_of(request.target_agent_id)
        if group_a is None or group_b is None:
            unassigned = [
                aid for aid, g in ((request.requesting_agent_id, group_a), (request.target_agent_id, group_b))
                if g is None
            ]
            reason = f"agent(s) not assigned a privilege group: {unassigned}"
        else:
            reason = (
                f"'{request.requesting_agent_id}' (group '{group_a}') is not in the same "
                f"privilege group as '{request.target_agent_id}' (group '{group_b}')"
            )

        return AccessDecision(allowed=False, requires_confirmation=requires_confirmation, reason=reason)
