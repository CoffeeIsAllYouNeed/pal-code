from unittest.mock import patch


def _build_orchestrator(**kwargs):
    class _FakeCgroupManager:
        def __init__(self, parent_slice):
            pass

    with patch("pal.orchestrator.pipeline.CgroupManager", _FakeCgroupManager):
        from pal.orchestrator.pipeline import OrchestratorPipeline
        return OrchestratorPipeline(**kwargs)


def test_sdk_for_shares_orchestrator_managers(tmp_path):
    from pal.llm_wrapper.components.base import WrapperConfig

    orch = _build_orchestrator(storage_root=str(tmp_path / "storage"))
    orch.register_builtin_tools()

    sdk = orch.sdk_for("agent-a", WrapperConfig(backend="ollama", model="llama3"))

    assert sdk._tool_manager is orch.tool_manager
    assert "calculator" in sdk.list_tools()
    assert sdk.call_tool("calculator", expression="6 * 7") == 42

    sdk.write("f.json", {"v": 1})
    assert sdk.read("f.json") == {"v": 1}


def test_sdk_for_two_agents_share_access_manager(tmp_path):
    from pal.llm_wrapper.components.base import WrapperConfig

    orch = _build_orchestrator(storage_root=str(tmp_path / "storage"))
    orch.assign_privilege_group("agent-a", "team-x")
    orch.assign_privilege_group("agent-b", "team-x")

    sdk_a = orch.sdk_for("agent-a", WrapperConfig(backend="ollama", model="llama3"))
    sdk_b = orch.sdk_for("agent-b", WrapperConfig(backend="ollama", model="llama3"))

    sdk_a.write("shared.json", "team data")
    assert sdk_b.read_from("agent-a", "shared.json") == "team data"
