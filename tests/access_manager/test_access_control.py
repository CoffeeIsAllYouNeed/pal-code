from pal.access_manager.components.access_control import AccessController
from pal.access_manager.components.base import OperationType
from pal.access_manager.components.privilege_registry import PrivilegeRegistry


def test_privilege_registry_hashmap_basics():
    registry = PrivilegeRegistry()
    registry.assign("agent-a", "team-x")
    registry.assign("agent-b", "team-x")
    registry.assign("agent-c", "team-y")

    assert registry.group_of("agent-a") == "team-x"
    assert registry.same_group("agent-a", "agent-b")
    assert not registry.same_group("agent-a", "agent-c")
    assert registry.group_of("unassigned-agent") is None


def test_own_resource_always_allowed_regardless_of_privilege_group():
    from pal.access_manager.components.base import AccessRequest
    controller = AccessController(PrivilegeRegistry())  # nobody assigned any group
    decision = controller.check(AccessRequest(requesting_agent_id="agent-a", target_agent_id="agent-a", operation=OperationType.READ))
    assert decision.allowed


def test_cross_agent_same_group_allowed():
    from pal.access_manager.components.base import AccessRequest
    registry = PrivilegeRegistry()
    registry.assign("agent-a", "team-x")
    registry.assign("agent-b", "team-x")
    controller = AccessController(registry)

    decision = controller.check(AccessRequest(requesting_agent_id="agent-a", target_agent_id="agent-b", operation=OperationType.READ))
    assert decision.allowed


def test_cross_agent_different_group_denied():
    from pal.access_manager.components.base import AccessRequest
    registry = PrivilegeRegistry()
    registry.assign("agent-a", "team-x")
    registry.assign("agent-c", "team-y")
    controller = AccessController(registry)

    decision = controller.check(AccessRequest(requesting_agent_id="agent-a", target_agent_id="agent-c", operation=OperationType.READ))
    assert not decision.allowed
    assert "team-x" in decision.reason and "team-y" in decision.reason


def test_cross_agent_unassigned_group_denied():
    from pal.access_manager.components.base import AccessRequest
    controller = AccessController(PrivilegeRegistry())  # nobody assigned
    decision = controller.check(AccessRequest(requesting_agent_id="agent-a", target_agent_id="agent-b", operation=OperationType.WRITE))
    assert not decision.allowed


def test_destructive_ops_flagged_for_confirmation_even_when_allowed():
    from pal.access_manager.components.base import DESTRUCTIVE_OPS, AccessRequest
    controller = AccessController(PrivilegeRegistry())

    for op in DESTRUCTIVE_OPS:
        decision = controller.check(AccessRequest(requesting_agent_id="agent-a", target_agent_id="agent-a", operation=op))
        assert decision.allowed  # own resource
        assert decision.requires_confirmation


def test_non_destructive_ops_not_flagged():
    from pal.access_manager.components.base import AccessRequest
    controller = AccessController(PrivilegeRegistry())
    decision = controller.check(AccessRequest(requesting_agent_id="agent-a", target_agent_id="agent-a", operation=OperationType.READ))
    assert not decision.requires_confirmation
