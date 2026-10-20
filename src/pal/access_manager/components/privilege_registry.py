"""
Privilege-group hashmap (AIOS section 3.8): agent_id -> privilege
group, enforced by AccessController before any cross-agent access is
permitted. Each agent is assigned to exactly one group; agents sharing
a group are in the same "privilege domain" and may access each other's
resources.
"""
from typing import Optional


class PrivilegeRegistry:
    def __init__(self):
        self._groups: dict[str, str] = {}

    def assign(self, agent_id: str, group: str) -> None:
        self._groups[agent_id] = group

    def group_of(self, agent_id: str) -> Optional[str]:
        return self._groups.get(agent_id)

    def same_group(self, agent_a: str, agent_b: str) -> bool:
        group_a = self.group_of(agent_a)
        group_b = self.group_of(agent_b)
        return group_a is not None and group_a == group_b

    def revoke(self, agent_id: str) -> None:
        self._groups.pop(agent_id, None)

    def all_assignments(self) -> dict[str, str]:
        return dict(self._groups)
