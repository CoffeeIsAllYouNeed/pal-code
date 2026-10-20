"""What an MCP server offers: tools the model can call, resources it can read, and prompts it can ask for."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from localrag.core.errors import LocalRAGError, ProtocolError
from localrag.mcp.jsonrpc import INVALID_PARAMS, METHOD_NOT_FOUND, RESOURCE_NOT_FOUND
from localrag.mcp.schema import apply_defaults, validate


@dataclass
class ToolResult:
    """What a tool call returns: content blocks for the model, whether it failed, and the same result as structured data."""
    content: list[dict[str, Any]]
    is_error: bool = False
    structured: Any = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"content": self.content, "isError": self.is_error}
        if self.structured is not None:
            out["structuredContent"] = self.structured
        return out


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    read_only: bool = True

    def descriptor(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema,
                "annotations": {"readOnlyHint": self.read_only, "destructiveHint": not self.read_only}}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, schema: dict[str, Any], read_only: bool = True) -> Callable[[Callable], Callable]:
        """A decorator that registers a function as a tool."""
        def decorate(fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
            self.register(Tool(name, description, schema, fn, read_only))
            return fn
        return decorate

    def names(self) -> list[str]:
        return sorted(self._tools)

    def descriptors(self) -> list[dict[str, Any]]:
        return [self._tools[n].descriptor() for n in self.names()]

    def call(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        """Validate the arguments against the tool's schema and run it. A failure inside the tool is reported as an error result, not raised."""
        tool = self._tools.get(name)
        if tool is None:
            raise ProtocolError(f"unknown tool: {name}", INVALID_PARAMS)
        args = apply_defaults(arguments or {}, tool.input_schema)
        problems = validate(args, tool.input_schema)
        if problems:
            raise ProtocolError("invalid arguments: " + "; ".join(problems), INVALID_PARAMS, problems)
        try:
            value = tool.handler(args)
        except LocalRAGError as e:
            return ToolResult([{"type": "text", "text": f"{type(e).__name__}: {e}"}], True)
        except Exception as e:  # noqa: BLE001  (a bug in a tool must not take the server down)
            return ToolResult([{"type": "text", "text": f"internal error in {name}: {type(e).__name__}"}], True)
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        return ToolResult([{"type": "text", "text": text}], False, None if isinstance(value, str) else value)


@dataclass
class Resource:
    uri: str
    name: str
    mime_type: str
    reader: Callable[[], str]
    description: str = ""

    def descriptor(self) -> dict[str, Any]:
        return {"uri": self.uri, "name": self.name, "mimeType": self.mime_type, "description": self.description}


@dataclass
class ResourceRegistry:
    """Resources are listed by a function (so the list can change as documents come and go) and read by uri."""
    lister: Callable[[], list[Resource]] = field(default=lambda: [])
    templates: list[dict[str, str]] = field(default_factory=list)

    def list(self) -> list[dict[str, Any]]:
        return [r.descriptor() for r in self.lister()]

    def read(self, uri: str) -> dict[str, Any]:
        for r in self.lister():
            if r.uri == uri:
                return {"uri": uri, "mimeType": r.mime_type, "text": r.reader()}
        raise ProtocolError(f"resource not found: {uri}", RESOURCE_NOT_FOUND)


@dataclass
class Prompt:
    name: str
    description: str
    arguments: list[dict[str, Any]]
    builder: Callable[[dict[str, str]], list[dict[str, Any]]]

    def descriptor(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "arguments": self.arguments}


class PromptRegistry:
    def __init__(self) -> None:
        self._prompts: dict[str, Prompt] = {}

    def register(self, prompt: Prompt) -> None:
        self._prompts[prompt.name] = prompt

    def descriptors(self) -> list[dict[str, Any]]:
        return [self._prompts[n].descriptor() for n in sorted(self._prompts)]

    def get(self, name: str, arguments: dict[str, str] | None) -> dict[str, Any]:
        prompt = self._prompts.get(name)
        if prompt is None:
            raise ProtocolError(f"unknown prompt: {name}", METHOD_NOT_FOUND)
        args = arguments or {}
        missing = [a["name"] for a in prompt.arguments if a.get("required") and a["name"] not in args]
        if missing:
            raise ProtocolError(f"missing prompt arguments: {', '.join(missing)}", INVALID_PARAMS)
        return {"description": prompt.description, "messages": prompt.builder(args)}
