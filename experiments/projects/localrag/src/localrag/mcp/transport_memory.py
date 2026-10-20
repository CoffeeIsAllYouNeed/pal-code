"""An in-process client for tests and embedding: it talks to a server directly, without any pipes."""
from __future__ import annotations

import itertools
import json
from typing import Any

from localrag.core.errors import ProtocolError
from localrag.mcp import jsonrpc
from localrag.mcp.server import McpServer


class MemoryClient:
    def __init__(self, server: McpServer):
        self.server = server
        self._ids = itertools.count(1)

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a request and return its result; an error response is raised as ProtocolError with the server's code."""
        raw = jsonrpc.encode(jsonrpc.Request(next(self._ids), method, params))
        reply = self.server.handle_raw(raw)
        assert reply is not None
        data = json.loads(reply)
        if "error" in data:
            raise ProtocolError(data["error"]["message"], data["error"]["code"], data["error"].get("data"))
        return data["result"]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.server.handle_raw(jsonrpc.encode(jsonrpc.Notification(method, params)))

    def initialize(self, version: str = "2025-06-18", **extra: Any) -> dict[str, Any]:
        """Run the handshake: initialize, then the initialized notification."""
        result = self.request("initialize", {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "memory-client", "version": "0"}, **extra})
        self.notify("notifications/initialized")
        return result

    def call_tool(self, name: str, /, **arguments: Any) -> dict[str, Any]:
        """Call a tool by name; the name is positional-only so a tool may have an argument called name."""
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def tool_text(self, name: str, /, **arguments: Any) -> str:
        """The text of a tool call's first content block."""
        return self.call_tool(name, **arguments)["content"][0]["text"]
