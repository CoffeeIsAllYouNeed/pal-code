import os
import sys

import pytest

from pal.orchestrator.utils import device
from pal.orchestrator.utils.device import ToolDevice, call_tool
from pal.tool_manager.builtin import register_builtin_tools
from pal.tool_manager.pipeline import ToolManagerPipeline

needs_unix = pytest.mark.skipif(not device.HAS_UNIX_SOCKETS, reason="no Unix sockets on this OS")


def run(dev, **kw):
    dev.register_tool("echo", lambda a: a)
    dev.start()
    try:
        yield dev
    finally:
        dev.stop()


@pytest.fixture
def tcp():
    yield from run(ToolDevice("127.0.0.1:0"))


@pytest.fixture
def unix(tmp_path):
    yield from run(ToolDevice(str(tmp_path / "t.sock")))


def test_tcp_transport_round_trip_and_ephemeral_port(tcp):
    host, port = tcp.address.rsplit(":", 1)
    assert host == "127.0.0.1" and int(port) > 0
    assert call_tool("a", "echo", {"x": 1}, address=tcp.address) == {"x": 1}


def test_tcp_binds_loopback_only(tcp):
    assert tcp._server.getsockname()[0] == "127.0.0.1"


def test_large_messages_are_read_completely(tcp):
    blob = {"blob": "é" * 100_000}   # ~600 KB on the wire: over 64 KB, under the 1 MB cap
    assert call_tool("a", "echo", blob, address=tcp.address) == blob


def test_unknown_tool_and_tool_errors_come_back_as_errors(tcp):
    with pytest.raises(RuntimeError, match="unknown tool"):
        call_tool("a", "missing", {}, address=tcp.address)
    tcp.register_tool("boom", lambda a: 1 / 0)
    with pytest.raises(RuntimeError, match="division by zero"):
        call_tool("a", "boom", {}, address=tcp.address)


def test_oversized_request_is_refused_and_the_server_stays_healthy(tcp):
    port = int(tcp.address.rsplit(":", 1)[1])
    with pytest.raises(RuntimeError, match="too large"):
        call_tool("a", "echo", {"blob": "x" * (device.MAX_MESSAGE_BYTES + 10)}, address=tcp.address)
    assert call_tool("a", "echo", {"ok": 1}, address=f"127.0.0.1:{port}") == {"ok": 1}


def test_tool_manager_is_used_when_supplied():
    tm = ToolManagerPipeline()
    register_builtin_tools(tm)
    dev = ToolDevice("127.0.0.1:0", tool_manager=tm)
    dev.start()
    try:
        assert call_tool("a", "calculator", {"expression": "6*7"}, address=dev.address) == 42
        with pytest.raises(RuntimeError, match="missing required parameter"):
            call_tool("a", "calculator", {}, address=dev.address)
    finally:
        dev.stop()


@needs_unix
def test_unix_transport_round_trip_and_socket_is_owner_only(unix):
    assert call_tool("a", "echo", {"y": 2}, address=unix.address) == {"y": 2}
    if sys.platform != "win32":
        assert (os.stat(unix.address).st_mode & 0o777) == 0o600


@needs_unix
def test_stop_removes_the_socket_file(tmp_path):
    d = ToolDevice(str(tmp_path / "s.sock"))
    d.start()
    assert os.path.exists(d.address)
    d.stop()
    assert not os.path.exists(d.address)


def test_address_kinds_are_told_apart():
    assert device._is_tcp("127.0.0.1:5000") and device._is_tcp("localhost:80")
    assert not device._is_tcp("/tmp/x.sock") and not device._is_tcp("C:\\temp\\x.sock") and not device._is_tcp("C:/x")


def test_without_unix_sockets_the_default_is_loopback_tcp(monkeypatch):
    monkeypatch.setattr(device, "HAS_UNIX_SOCKETS", False)
    d = ToolDevice("/tmp/whatever.sock")  # a path is ignored on such an OS
    d.register_tool("echo", lambda a: a)
    d.address = "127.0.0.1:0"
    d.start()
    try:
        assert call_tool("a", "echo", {"w": 1}, address=d.address) == {"w": 1}
    finally:
        d.stop()
