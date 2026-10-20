"""
A small built-in tool catalog (AIOS section 3.9's "Tool Integration"):
a minimal, dependency-free set of example tools covering a few
different input/output modalities (free text, structured expressions,
plain numbers), registered into a ToolManagerPipeline so a fresh
AgentSDK has something callable out of the box.

This is intentionally small. AIOS's own SDK ships a much larger
catalog of tools sourced from real external platforms, which needs
network access and API keys this project doesn't assume are available.
Real deployments register their own tools alongside (or instead of)
these via ToolManagerPipeline.register_tool() -- this catalog is a
starting point, not a ceiling.
"""
import time

from pal.tool_manager.builtin import safe_eval
from pal.tool_manager.components.base import ToolParam
from pal.tool_manager.pipeline import ToolManagerPipeline


def _calculator():
    # AST-based evaluator: never eval(), so it cannot execute code and cannot be made to hang.
    return safe_eval


def _current_time():
    def current_time() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    return current_time


def _word_count():
    def word_count(text: str) -> int:
        return len(text.split())

    return word_count


def register_builtin_tools(tool_manager: ToolManagerPipeline) -> None:
    tool_manager.register_tool("calculator", _calculator, params=[ToolParam("expression", str)])
    tool_manager.register_tool("current_time", _current_time, params=[])
    tool_manager.register_tool("word_count", _word_count, params=[ToolParam("text", str)])
