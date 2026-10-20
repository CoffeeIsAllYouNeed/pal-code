from unittest.mock import patch

from pal.tool_manager.components.base import ToolParam


def test_orchestrator_register_and_invoke_tool():
    # OrchestratorPipeline.__init__ constructs a real CgroupManager,
    # which needs sudo/real cgroups -- not available in this sandbox.
    # Patch it out; nothing tool-manager related depends on it.
    class _FakeCgroupManager:
        def __init__(self, parent_slice):
            pass

    with patch("pal.orchestrator.pipeline.CgroupManager", _FakeCgroupManager):
        from pal.orchestrator.pipeline import OrchestratorPipeline

        orch = OrchestratorPipeline()
        orch.register_tool("add", lambda: (lambda x, y: x + y), params=[ToolParam("x", int), ToolParam("y", int)])

        result = orch.invoke_tool("agent-a", "add", {"x": 2, "y": 3})
        assert result.ok
        assert result.result == 5

        assert orch.tool_device.tool_manager is orch.tool_manager
        assert "add" in orch.tool_manager.list_tools()
