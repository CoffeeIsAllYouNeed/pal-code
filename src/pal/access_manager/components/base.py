"""
Access request/decision shapes for the Access Manager (AIOS section
3.8). An AccessRequest always names both the requesting agent and the
agent whose resource is being touched -- when they're the same agent,
there's nothing cross-agent to regulate, only user-intervention (for
destructive ops) applies.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OperationType(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    OVERWRITE = "overwrite"
    PRIVILEGE_MODIFY = "privilege_modify"


# "irreversible operations (deletion, overwrite, privilege modification)"
DESTRUCTIVE_OPS = {OperationType.DELETE, OperationType.OVERWRITE, OperationType.PRIVILEGE_MODIFY}


@dataclass
class AccessRequest:
    requesting_agent_id: str
    target_agent_id: str   # whose resource is being accessed -- equal to
                            # requesting_agent_id for an agent acting on
                            # its own data
    operation: OperationType
    resource: str = ""     # a key, tool name, or other resource descriptor -- for logging/audit only


@dataclass
class AccessDecision:
    allowed: bool
    requires_confirmation: bool
    reason: Optional[str] = None
