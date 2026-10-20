import json
import os
import platform
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from pal.access_manager.pipeline import AccessManagerPipeline
from pal.config import OsKernelConfig
from pal.os_kernel import OSKernel, OSKernelError, network, system_info


def kernel(tmp_path, access=None, **cfg):
    return OSKernel(OsKernelConfig.coerce(cfg), access=access, root=tmp_path / "ws")


# -- files ----------------------------------------------------------------------------------------------------
def test_files_roundtrip_inside_a_per_agent_workspace(tmp_path):
    k = kernel(tmp_path)
    assert k.handle("a", "file_write", {"path": "docs/readme.md", "content": "héllo"}) == {"path": "docs/readme.md", "bytes": 6}
    assert k.handle("a", "file_read", {"path": "docs/readme.md"}) == "héllo"
    k.handle("a", "file_write", {"path": "top.txt", "content": "x"})
    assert k.handle("a", "file_list", {}) == ["docs/readme.md", "top.txt"]
    assert k.handle("a", "file_list", {"prefix": "docs/"}) == ["docs/readme.md"]
    assert k.handle("b", "file_list", {}) == []                                   # another agent sees nothing
    assert (tmp_path / "ws" / "a" / "docs" / "readme.md").read_text(encoding="utf-8") == "héllo"


@pytest.mark.parametrize("bad", ["../x", "/etc/passwd", "a/../../x", "C:\\x", "con.txt", "a\x00b", "", "trail.", "a/./b", "x:y"])
def test_paths_outside_the_workspace_or_unsafe_on_any_os_are_refused(tmp_path, bad):
    k = kernel(tmp_path)
    for op in ("file_read", "file_write", "file_delete"):
        with pytest.raises(OSKernelError, match="invalid path"):
            k.handle("a", op, {"path": bad, "content": "x"})
    assert not (tmp_path / "x").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_a_symlink_pointing_outside_is_refused(tmp_path):
    k = kernel(tmp_path)
    k.handle("a", "file_write", {"path": "ok.txt", "content": "x"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    try:
        os.symlink(outside, tmp_path / "ws" / "a" / "link")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    with pytest.raises(OSKernelError, match="escapes"):
        k.handle("a", "file_read", {"path": "link/secret.txt"})


def test_size_limits(tmp_path):
    k = kernel(tmp_path, max_read_bytes=10, max_write_bytes=20)
    with pytest.raises(OSKernelError, match="write limit"):
        k.handle("a", "file_write", {"path": "big.txt", "content": "x" * 21})
    k.handle("a", "file_write", {"path": "ok.txt", "content": "x" * 20})
    with pytest.raises(OSKernelError, match="read limit"):
        k.handle("a", "file_read", {"path": "ok.txt"})
    with pytest.raises(OSKernelError, match="no such file"):
        k.handle("a", "file_read", {"path": "missing.txt"})


def test_overwrite_and_delete_need_confirmation_and_default_to_no(tmp_path):
    k = kernel(tmp_path)
    k.handle("a", "file_write", {"path": "n.txt", "content": "one"})            # a new file asks nobody
    with pytest.raises(OSKernelError, match="not permitted"):
        k.handle("a", "file_write", {"path": "n.txt", "content": "two"})        # overwrite: declined by default
    with pytest.raises(OSKernelError, match="not permitted"):
        k.handle("a", "file_delete", {"path": "n.txt"})
    assert k.handle("a", "file_read", {"path": "n.txt"}) == "one"
    asked = []
    access = AccessManagerPipeline(lambda req: asked.append((req.operation.value, req.resource)) or True)
    k2 = kernel(tmp_path / "k2", access=access)
    k2.handle("a", "file_write", {"path": "n.txt", "content": "one"})
    k2.handle("a", "file_write", {"path": "n.txt", "content": "two"})
    k2.handle("a", "file_delete", {"path": "n.txt"})
    assert asked == [("overwrite", "n.txt"), ("delete", "n.txt")]
    assert k2.handle("a", "file_list", {}) == []


# -- keyboard, mouse, unknown, disabled, audit ------------------------------------------------------------------
def test_nobody_is_asked_to_confirm_an_invalid_or_missing_target(tmp_path):
    asked = []
    k = kernel(tmp_path, access=AccessManagerPipeline(lambda req: asked.append(req) or True))
    for op, path in (("file_delete", "../x"), ("file_delete", "missing.txt"), ("file_write", "../x")):
        with pytest.raises(OSKernelError):
            k.handle("a", op, {"path": path, "content": "x"})
    assert asked == []


def test_keyboard_and_mouse_are_clearly_unsupported(tmp_path):
    k = kernel(tmp_path)
    for op in ("keyboard", "mouse"):
        with pytest.raises(OSKernelError, match="desktop session"):
            k.handle("a", op, {})
    with pytest.raises(OSKernelError, match="unknown OS operation"):
        k.handle("a", "reboot", {})


def test_the_whole_interface_can_be_turned_off(tmp_path):
    with pytest.raises(OSKernelError, match="turned off"):
        kernel(tmp_path, enabled=False).handle("a", "sys_info", {})


def test_every_call_is_audited_without_contents(tmp_path):
    k = kernel(tmp_path)
    k.handle("a", "file_write", {"path": "p.txt", "content": "TOP SECRET"})
    with pytest.raises(OSKernelError):
        k.handle("a", "file_read", {"path": "../x"})
    assert [(e["op"], e["ok"]) for e in k.audit] == [("file_write", True), ("file_read", False)]
    assert "TOP SECRET" not in json.dumps(list(k.audit))


# -- system facts from the Ubuntu kernel ---------------------------------------------------------------------------
def test_system_info_reports_the_os_kernel():
    info = system_info()
    assert info["system"] == platform.system() and info["kernel"] == platform.release() and info["cpu_count"]


@pytest.mark.skipif(platform.system() != "Linux", reason="reads /proc")
def test_system_info_reads_proc_on_linux():
    info = system_info()
    assert info["memory"]["total_mb"] > 0 and len(info["load_average"]) == 3
    assert open("/proc/meminfo", encoding="utf-8").read().startswith("MemTotal") or "MemTotal" in open("/proc/meminfo", encoding="utf-8").read()


# -- network -------------------------------------------------------------------------------------------------------
class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/redir":
            return self._send(302, b"", {"Location": "/echo"})
        if self.path == "/loop":
            return self._send(302, b"", {"Location": "/loop"})
        return self._send(200, b"x" * 5000 if self.path == "/big" else b"hello", {"Content-Type": "text/plain"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self._send(200, self.rfile.read(n), {"Content-Type": "application/json"})


@pytest.fixture
def server():
    s = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def test_get_and_post_work_when_private_addresses_are_allowed(tmp_path, server):
    k = kernel(tmp_path, allow_private_network=True)
    r = k.handle("a", "net_get", {"url": server + "/echo"})
    assert (r["status"], r["text"], r["truncated"]) == (200, "hello", False)
    assert k.handle("a", "net_get", {"url": server + "/redir"})["text"] == "hello"      # redirects are followed by hand
    p = k.handle("a", "net_post", {"url": server + "/x", "json": {"a": 1}})
    assert json.loads(p["text"]) == {"a": 1}


def test_private_and_loopback_addresses_are_blocked_by_default(tmp_path, server):
    k = kernel(tmp_path)
    for url in (server + "/echo", "http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest/meta-data/",
                "http://10.0.0.5/", "http://[::1]/"):
        with pytest.raises(OSKernelError, match="private or reserved"):
            k.handle("a", "net_get", {"url": url})


@pytest.mark.parametrize("url,msg", [("file:///etc/passwd", "only http"), ("ftp://x.test/", "only http"),
                                     ("http://user:pw@example.test/", "credentials"), ("not a url", "only http")])
def test_bad_urls_are_refused(tmp_path, url, msg):
    with pytest.raises(OSKernelError, match=msg):
        kernel(tmp_path).handle("a", "net_get", {"url": url})


def test_allowed_hosts_limit_where_requests_may_go(tmp_path, server):
    k = kernel(tmp_path, allow_private_network=True, allowed_hosts=["api.example.test"])
    with pytest.raises(OSKernelError, match="not on the allowed list"):
        k.handle("a", "net_get", {"url": server + "/echo"})


def test_response_size_is_capped_and_redirect_loops_stop(tmp_path, server):
    k = kernel(tmp_path, allow_private_network=True, net_max_bytes=1000)
    r = k.handle("a", "net_get", {"url": server + "/big"})
    assert len(r["text"]) == 1000 and r["truncated"] is True
    with pytest.raises(OSKernelError, match="too many redirects"):
        k.handle("a", "net_get", {"url": server + "/loop"})


def test_a_redirect_to_a_private_address_is_blocked_at_the_hop(monkeypatch):
    monkeypatch.setattr(network.socket, "getaddrinfo", lambda host, port, **kw: [(2, 1, 6, "", ("127.0.0.1" if host == "evil.test" else "93.184.216.34", port))])

    class Resp:
        is_redirect = True
        headers = {"Location": "http://evil.test/admin"}
        def close(self): pass

    monkeypatch.setattr("requests.request", lambda *a, **kw: Resp())
    with pytest.raises(OSKernelError, match="private or reserved"):
        network.request("GET", "http://public.test/", allow_private=False, allowed_hosts=[], timeout=5, max_bytes=100)


def test_unresolvable_hosts_and_network_failures_are_clean_errors(tmp_path, monkeypatch):
    with pytest.raises(OSKernelError, match="could not resolve"):
        kernel(tmp_path).handle("a", "net_get", {"url": "http://no-such-host.invalid/"})
    monkeypatch.setattr(network.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 80))])
    import requests
    monkeypatch.setattr("requests.request", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("boom secret")))
    with pytest.raises(OSKernelError) as e:
        kernel(tmp_path).handle("a", "net_get", {"url": "http://public.test/"})
    assert "ConnectionError" in str(e.value) and "secret" not in str(e.value)


def test_config_is_validated():
    with pytest.raises(Exception, match="must be > 0"):
        OsKernelConfig.coerce({"max_read_bytes": 0})
    with pytest.raises(Exception, match="list of host names"):
        OsKernelConfig.coerce({"allowed_hosts": "example.com"})
    assert SimpleNamespace  # keep import used
