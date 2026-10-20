"""JSON-RPC 2.0, the message format MCP is built on: requests, notifications, responses and the standard error codes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Union

from localrag.core.errors import ProtocolError

JSONRPC_VERSION = "2.0"
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603
NOT_INITIALIZED, UNAUTHORIZED, RATE_LIMITED, RESOURCE_NOT_FOUND = -32002, -32001, -32003, -32004


@dataclass
class Request:
    """A call that expects an answer; id is how the answer is matched to it."""
    id: Union[int, str]
    method: str
    params: Any = None


@dataclass
class Notification:
    """A message that expects no answer (it has no id)."""
    method: str
    params: Any = None


@dataclass
class ErrorObject:
    code: int
    message: str
    data: Any = None


@dataclass
class Response:
    """The answer to a request: either a result or an error, never both."""
    id: Union[int, str, None]
    result: Any = None
    error: ErrorObject | None = None


Message = Union[Request, Notification, Response]


def parse_message(raw: str | bytes) -> Message | list[Message]:
    """Parse one JSON-RPC message, or a batch (a JSON array) of them. Raises ProtocolError carrying the right error code."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProtocolError(f"parse error: {e}", PARSE_ERROR) from e
    if isinstance(data, list):
        if not data:
            raise ProtocolError("an empty batch is not allowed", INVALID_REQUEST)
        return [_from_object(item) for item in data]
    return _from_object(data)


def _from_object(obj: Any) -> Message:
    if not isinstance(obj, dict) or obj.get("jsonrpc") != JSONRPC_VERSION:
        raise ProtocolError('a message must be an object with "jsonrpc": "2.0"', INVALID_REQUEST)
    if "method" in obj:
        if not isinstance(obj["method"], str):
            raise ProtocolError("method must be a string", INVALID_REQUEST)
        params = obj.get("params")
        if params is not None and not isinstance(params, (dict, list)):
            raise ProtocolError("params must be an object or an array", INVALID_PARAMS)
        if "id" in obj:
            if isinstance(obj["id"], bool) or not isinstance(obj["id"], (int, str)):
                raise ProtocolError("id must be a string or a number", INVALID_REQUEST)
            return Request(obj["id"], obj["method"], params)
        return Notification(obj["method"], params)
    if "result" in obj or "error" in obj:
        err = obj.get("error")
        return Response(obj.get("id"), obj.get("result"), ErrorObject(err["code"], err["message"], err.get("data")) if err else None)
    raise ProtocolError("a message needs a method, a result or an error", INVALID_REQUEST)


def to_object(message: Message) -> dict[str, Any]:
    """The JSON-ready dictionary form of a message."""
    out: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION}
    if isinstance(message, Request):
        out.update(id=message.id, method=message.method)
        if message.params is not None:
            out["params"] = message.params
    elif isinstance(message, Notification):
        out["method"] = message.method
        if message.params is not None:
            out["params"] = message.params
    else:
        out["id"] = message.id
        if message.error is not None:
            out["error"] = {"code": message.error.code, "message": message.error.message,
                            **({"data": message.error.data} if message.error.data is not None else {})}
        else:
            out["result"] = message.result
    return out


def encode(message: Message | list[Message]) -> str:
    """One line of compact JSON for a message or a batch, ready to write to a transport."""
    payload = [to_object(m) for m in message] if isinstance(message, list) else to_object(message)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def error_response(request_id: Union[int, str, None], code: int, message: str, data: Any = None) -> Response:
    """A response that reports an error."""
    return Response(request_id, error=ErrorObject(code, message, data))
