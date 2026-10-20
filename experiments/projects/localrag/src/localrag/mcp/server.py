"""The MCP server: it speaks JSON-RPC, runs the initialize handshake, and answers tool, resource and prompt requests."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from localrag.core.config import ServerConfig
from localrag.core.errors import ProtocolError
from localrag.core.log import get_logger
from localrag.mcp import jsonrpc
from localrag.mcp.capabilities import negotiate_version, server_capabilities, server_info
from localrag.mcp.jsonrpc import Message, Notification, Request, Response
from localrag.mcp.middleware import RateLimiter, Timing, check_token
from localrag.mcp.registry import PromptRegistry, ResourceRegistry, ToolRegistry

log = get_logger(__name__)


class McpServer:
    def __init__(self, tools: ToolRegistry, resources: ResourceRegistry, prompts: PromptRegistry, config: ServerConfig | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.tools, self.resources, self.prompts = tools, resources, prompts
        self.config = config or ServerConfig()
        self.limiter = RateLimiter(self.config.rate_limit_per_minute, clock)
        self.timing = Timing()
        self.initialized = False
        self.protocol_version = ""
        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "ping": lambda p: {}, "tools/list": self._tools_list, "tools/call": self._tools_call,
            "resources/list": lambda p: {"resources": self.resources.list()},
            "resources/templates/list": lambda p: {"resourceTemplates": self.resources.templates},
            "resources/read": self._resources_read, "prompts/list": lambda p: {"prompts": self.prompts.descriptors()},
            "prompts/get": self._prompts_get}

    # -- entry points ------------------------------------------------------------------------------------------------------------
    def handle_raw(self, raw: str | bytes) -> str | None:
        """Handle one line from a transport. Returns the JSON to send back, or None when nothing should be sent."""
        if len(raw) > self.config.max_message_bytes:
            return jsonrpc.encode(jsonrpc.error_response(None, jsonrpc.INVALID_REQUEST, "message is too large"))
        try:
            parsed = jsonrpc.parse_message(raw)
        except ProtocolError as e:
            return jsonrpc.encode(jsonrpc.error_response(None, e.code, str(e)))
        if isinstance(parsed, list):
            replies = [r for r in (self.handle(m) for m in parsed) if r is not None]
            return jsonrpc.encode(replies) if replies else None
        reply = self.handle(parsed)
        return jsonrpc.encode(reply) if reply is not None else None

    def handle(self, message: Message) -> Response | None:
        """Handle one parsed message: requests get a response, notifications and stray responses get none."""
        if isinstance(message, Notification):
            if message.method == "notifications/initialized" and self.initialized:
                log.debug("client finished initialisation")
            return None
        if isinstance(message, Response):
            return None
        started = time.perf_counter()
        try:
            return Response(message.id, self._dispatch(message))
        except ProtocolError as e:
            return jsonrpc.error_response(message.id, e.code, str(e), e.data)
        except Exception as e:  # noqa: BLE001  (never let one bad request stop the server)
            log.exception("internal error handling %s", message.method)
            return jsonrpc.error_response(message.id, jsonrpc.INTERNAL_ERROR, f"internal error: {type(e).__name__}")
        finally:
            self.timing.record(message.method, time.perf_counter() - started)

    # -- dispatch ----------------------------------------------------------------------------------------------------------------
    def _dispatch(self, request: Request) -> Any:
        params = request.params if isinstance(request.params, dict) else {}
        if request.method == "initialize":
            return self._initialize(params)
        if request.method == "ping":
            return {}
        if not self.initialized:
            raise ProtocolError("send initialize first", jsonrpc.NOT_INITIALIZED)
        if not self.limiter.allow():
            raise ProtocolError("rate limit exceeded; slow down", jsonrpc.RATE_LIMITED)
        handler = self._methods.get(request.method)
        if handler is None:
            raise ProtocolError(f"method not found: {request.method}", jsonrpc.METHOD_NOT_FOUND)
        return handler(params)

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        if not check_token(params.get("authToken") or (params.get("clientInfo") or {}).get("authToken"), self.config.auth_token):
            raise ProtocolError("authentication failed", jsonrpc.UNAUTHORIZED)
        self.protocol_version = negotiate_version(params.get("protocolVersion"))
        self.initialized = True
        return {"protocolVersion": self.protocol_version, "capabilities": server_capabilities(), "serverInfo": server_info(self.config),
                "instructions": "Search the indexed documents with the search tool; use tree_search to browse by section."}

    # -- methods -----------------------------------------------------------------------------------------------------------------
    def _tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": self.tools.descriptors()}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise ProtocolError("tools/call needs a tool name", jsonrpc.INVALID_PARAMS)
        arguments = params.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            raise ProtocolError("arguments must be an object", jsonrpc.INVALID_PARAMS)
        return self.tools.call(name, arguments).to_dict()

    def _resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise ProtocolError("resources/read needs a uri", jsonrpc.INVALID_PARAMS)
        return {"contents": [self.resources.read(uri)]}

    def _prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise ProtocolError("prompts/get needs a prompt name", jsonrpc.INVALID_PARAMS)
        return self.prompts.get(name, params.get("arguments"))
