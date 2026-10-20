from pal.sdk.components.tool_catalog import register_builtin_tools
from pal.tool_manager.pipeline import ToolManagerPipeline


def test_builtin_tools_registered():
    tm = ToolManagerPipeline()
    register_builtin_tools(tm)
    tools = tm.list_tools()
    assert "calculator" in tools
    assert "current_time" in tools
    assert "word_count" in tools


def test_calculator_evaluates_simple_expression():
    tm = ToolManagerPipeline()
    register_builtin_tools(tm)
    result = tm.invoke("agent-a", "calculator", {"expression": "2 + 3 * 4"})
    assert result.ok
    assert result.result == 14


def test_calculator_rejects_disallowed_characters():
    tm = ToolManagerPipeline()
    register_builtin_tools(tm)
    result = tm.invoke("agent-a", "calculator", {"expression": "__import__('os').system('echo hacked')"})
    assert not result.ok


def test_word_count():
    tm = ToolManagerPipeline()
    register_builtin_tools(tm)
    result = tm.invoke("agent-a", "word_count", {"text": "the quick brown fox"})
    assert result.ok
    assert result.result == 4


def test_current_time_returns_string():
    tm = ToolManagerPipeline()
    register_builtin_tools(tm)
    result = tm.invoke("agent-a", "current_time", {})
    assert result.ok
    assert isinstance(result.result, str)
