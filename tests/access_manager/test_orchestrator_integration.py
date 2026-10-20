from unittest.mock import patch


def _build_orchestrator(**kwargs):
    class _FakeCgroupManager:
        def __init__(self, parent_slice):
            pass

    with patch("pal.orchestrator.pipeline.CgroupManager", _FakeCgroupManager):
        from pal.orchestrator.pipeline import OrchestratorPipeline
        return OrchestratorPipeline(**kwargs)


def test_orchestrator_wires_access_manager_into_storage_manager(tmp_path):
    orch = _build_orchestrator(storage_root=str(tmp_path / "storage"))
    assert orch.storage_manager.access_manager is orch.access_manager


def test_assign_privilege_group_enables_cross_agent_access(tmp_path):
    orch = _build_orchestrator(storage_root=str(tmp_path / "storage"))

    orch.assign_privilege_group("agent-a", "team-x")
    orch.assign_privilege_group("agent-b", "team-x")

    sdk_a = orch.storage_sdk_for("agent-a")
    sdk_b = orch.storage_sdk_for("agent-b")

    sdk_a.write("shared.json", "team data")
    assert sdk_b.read_from("agent-a", "shared.json") == "team data"


def test_confirmation_handler_gates_destructive_ops(tmp_path):
    orch = _build_orchestrator(storage_root=str(tmp_path / "storage"))
    sdk = orch.storage_sdk_for("agent-a")
    sdk.write("f.json", "v1")

    # default handler denies -- overwrite fails
    try:
        sdk.write("f.json", "v2")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    orch.set_confirmation_handler(lambda req: True)
    sdk.write("f.json", "v2")
    assert sdk.read("f.json") == "v2"
