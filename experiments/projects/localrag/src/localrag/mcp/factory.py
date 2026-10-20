"""One call that wires an engine into a ready-to-run MCP server."""
from __future__ import annotations

from pathlib import Path

from localrag.engine import LocalRAG
from localrag.mcp.prompts import build_prompts
from localrag.mcp.resources import build_resources
from localrag.mcp.server import McpServer
from localrag.mcp.tools import build_tools


def build_server(engine: LocalRAG, allowed_roots: list[Path] | None = None) -> McpServer:
    """An MCP server exposing the engine's tools, documents (as resources) and prompts."""
    return McpServer(build_tools(engine, allowed_roots), build_resources(engine), build_prompts(engine), engine.config.server)
