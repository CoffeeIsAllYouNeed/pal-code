from pal.access_manager.components.base import AccessRequest, OperationType
from pal.access_manager.components.intervention import UserInterventionGate
from pal.access_manager.pipeline import AccessManagerPipeline


def test_default_handler_denies():
    gate = UserInterventionGate()
    req = AccessRequest(requesting_agent_id="a", target_agent_id="a", operation=OperationType.DELETE)
    assert gate.confirm(req) is False


def test_custom_handler_approves():
    gate = UserInterventionGate(confirmation_handler=lambda req: True)
    req = AccessRequest(requesting_agent_id="a", target_agent_id="a", operation=OperationType.DELETE)
    assert gate.confirm(req) is True


def test_set_handler_replaces_it():
    gate = UserInterventionGate()
    req = AccessRequest(requesting_agent_id="a", target_agent_id="a", operation=OperationType.DELETE)
    assert gate.confirm(req) is False

    gate.set_handler(lambda req: True)
    assert gate.confirm(req) is True


def test_authorize_non_destructive_never_asks_confirmation():
    am = AccessManagerPipeline()  # default handler denies everything
    decision = am.authorize("agent-a", "agent-a", OperationType.READ)
    assert decision.allowed  # never went through the (denying) confirmation gate


def test_authorize_destructive_denied_by_default_handler():
    am = AccessManagerPipeline()
    decision = am.authorize("agent-a", "agent-a", OperationType.DELETE)
    assert not decision.allowed
    assert "declined confirmation" in decision.reason


def test_authorize_destructive_approved_by_custom_handler():
    am = AccessManagerPipeline(confirmation_handler=lambda req: True)
    decision = am.authorize("agent-a", "agent-a", OperationType.DELETE)
    assert decision.allowed


def test_authorize_privilege_denial_short_circuits_before_confirmation():
    """A cross-agent request denied by privilege check should never
    reach the confirmation gate at all -- even a handler that always
    approves shouldn't matter."""
    am = AccessManagerPipeline(confirmation_handler=lambda req: True)
    am.assign_privilege_group("agent-a", "team-x")
    am.assign_privilege_group("agent-c", "team-y")

    decision = am.authorize("agent-a", "agent-c", OperationType.DELETE)
    assert not decision.allowed
    assert "privilege group" in decision.reason


def test_authorize_cross_agent_same_group_destructive_still_needs_confirmation():
    am = AccessManagerPipeline()  # default handler denies
    am.assign_privilege_group("agent-a", "team-x")
    am.assign_privilege_group("agent-b", "team-x")

    decision = am.authorize("agent-a", "agent-b", OperationType.DELETE)
    assert not decision.allowed  # privilege check passed, but confirmation denied
    assert "declined confirmation" in decision.reason

    am.set_confirmation_handler(lambda req: True)
    decision2 = am.authorize("agent-a", "agent-b", OperationType.DELETE)
    assert decision2.allowed
