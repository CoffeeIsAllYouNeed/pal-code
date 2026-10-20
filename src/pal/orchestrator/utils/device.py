"""Tool transport between agent processes and the orchestrator.

Portable: a Unix domain socket where the OS has one (Linux, macOS), a loopback TCP socket where it
does not (Windows). Requests and responses are one JSON object per line, so a message of any size
is read completely (the old single recv() truncated anything over 64 KB). The Unix socket file is
created owner-only (0600); the TCP fallback binds 127.0.0.1 only.

Stands in for a future /dev/pal-tools character device: porting later means swapping this
transport, not the protocol or how tool calls are handled.
"""
from __future__ import annotations

import logging
import os
import re
import socket
import tempfile
import threading
from collections.abc import Callable
from typing import Optional

from pal.orchestrator.utils.protocol import ToolRequest, ToolResponse
from pal.tool_manager.pipeline import ToolManagerPipeline

log = logging.getLogger(__name__)

HAS_UNIX_SOCKETS = hasattr(socket, "AF_UNIX")
DEFAULT_SOCKET_PATH = (
    os.path.join(tempfile.gettempdir(), "pal-tools.sock") if HAS_UNIX_SOCKETS else "127.0.0.1:47613")
MAX_MESSAGE_BYTES = 1_000_000
_TCP = re.compile(r"^[A-Za-z0-9.\-]+:\d+$")


def _is_tcp(address: str) -> bool:
    return bool(_TCP.match(address))


def _recv_line(conn: socket.socket) -> bytes:
    buf = bytearray()
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
        if len(buf) > MAX_MESSAGE_BYTES:
            raise ValueError("message too large")
    return bytes(buf)


def _connect(address: str, timeout: float) -> socket.socket:
    if _is_tcp(address) or not HAS_UNIX_SOCKETS:
        host, _, port = address.rpartition(":")
        return socket.create_connection((host or "127.0.0.1", int(port)), timeout=timeout)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(address)
    return sock


class ToolDevice:
    """Agents (in their own OS process) connect here to invoke tools; the orchestrator's process
    executes them, giving one place to apply permissions and auditing per tool call.

    With a ToolManagerPipeline, register_tool() registers a full ToolSpec (loader, params,
    parallel_limit, dependencies). Without one it falls back to a plain name -> callable dict.
    `address` is a Unix socket path or "host:port"; after start(), `.address` is the real one
    (port 0 picks a free port)."""

    def __init__(self, address: Optional[str] = None, tool_manager: Optional[ToolManagerPipeline] = None,
                 socket_path: Optional[str] = None):
        self.address = address or socket_path or DEFAULT_SOCKET_PATH
        self.tool_manager = tool_manager
        self._handlers: dict[str, Callable[[dict], dict]] = {}
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def socket_path(self) -> str:  # backwards-compatible name
        return self.address

    def register_tool(self, name: str, handler: Callable[[dict], dict], **tool_spec_kwargs) -> None:
        if self.tool_manager is not None:
            self.tool_manager.register_tool(name, handler, **tool_spec_kwargs)
        else:
            self._handlers[name] = handler

    def start(self) -> None:
        if self._running:
            return
        if _is_tcp(self.address) or not HAS_UNIX_SOCKETS:
            host, _, port = self.address.rpartition(":")
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.bind((host or "127.0.0.1", int(port or 0)))
            self.address = f"{self._server.getsockname()[0]}:{self._server.getsockname()[1]}"
        else:
            if os.path.exists(self.address):
                os.remove(self.address)
            self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server.bind(self.address)
            os.chmod(self.address, 0o600)  # owner only
        self._server.listen(5)
        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _serve_loop(self) -> None:
        while self._running:
            try:
                self._server.settimeout(0.5)
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(30)
            try:
                raw = _recv_line(conn)
                if not raw:
                    return
                req = ToolRequest.decode(raw)
                if self.tool_manager is not None:
                    result = self.tool_manager.invoke(req.agent_name, req.tool_name, req.args)
                    resp = ToolResponse(ok=result.ok, result=result.result, error=result.error)
                else:
                    handler = self._handlers.get(req.tool_name)
                    if handler is None:
                        resp = ToolResponse(ok=False, error=f"unknown tool: {req.tool_name}")
                    else:
                        resp = ToolResponse(ok=True, result=handler(req.args))
            except Exception as e:  # noqa: BLE001  (any failure becomes an error response)
                log.warning("tool request failed: %s", e)
                resp = ToolResponse(ok=False, error=str(e))
            try:
                conn.sendall(resp.encode())
            except OSError:
                pass

    def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
        if HAS_UNIX_SOCKETS and not _is_tcp(self.address) and os.path.exists(self.address):
            os.remove(self.address)


def call_tool(agent_name: str, tool_name: str, args: dict, socket_path: Optional[str] = None,
              timeout: float = 30, address: Optional[str] = None) -> dict:
    """Client-side helper: what an agent process calls to invoke a tool."""
    req = ToolRequest(agent_name=agent_name, tool_name=tool_name, args=args)
    sock = _connect(address or socket_path or DEFAULT_SOCKET_PATH, timeout)
    try:
        sock.sendall(req.encode())
        raw = _recv_line(sock)
    finally:
        sock.close()
    resp = ToolResponse.decode(raw)
    if not resp.ok:
        raise RuntimeError(f"tool call failed: {resp.error}")
    return resp.result
