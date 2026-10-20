"""net_request: auth headers from an allow-list, binary data in both directions through the workspace, and no key in any log."""
import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pal.access_manager.pipeline import AccessManagerPipeline
from pal.config import OsKernelConfig
from pal.os_kernel import OSKernel, OSKernelError

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8
SECRET = "sk-live-secret-value-123456"


class H(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _record(self, body=b""):
        H.seen.append({"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})

    def do_GET(self):
        self._record()
        if self.path.startswith("/image"):
            self._send(200, PNG, "image/png")
        elif self.path.startswith("/audio"):
            self._send(200, b"fLaC" + bytes(300), "audio/flac")
        elif self.path.startswith("/huge"):
            self._send(200, bytes(5000), "image/png")
        elif self.path.startswith("/redirect"):
            self._send(302, b"", "text/plain", {"Location": self.path.replace("/redirect", "/echo-headers")})
        elif self.path.startswith("/echo-headers"):
            self._send(200, json.dumps(H.seen[-1]["headers"]).encode(), "application/json")
        elif self.path.startswith("/error"):
            self._send(401, json.dumps({"error": "bad key"}).encode(), "application/json")
        elif self.path.startswith("/binary-error"):
            self._send(500, b"\x00\x01\x02", "application/octet-stream")
        else:
            self._send(200, json.dumps({"ok": True, "path": self.path}).encode(), "application/json")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._record(body)
        if self.path.startswith("/audio"):
            self._send(200, b"RIFF" + bytes(200), "audio/wav")
        else:
            self._send(200, json.dumps({"got_bytes": len(body), "type": self.headers.get("Content-Type"),
                                        "json": json.loads(body) if self.headers.get("Content-Type", "").startswith("application/json") else None}).encode(),
                       "application/json")


@pytest.fixture
def server():
    H.seen = []
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def kernel(tmp_path, **cfg):
    return OSKernel(OsKernelConfig.coerce({"allow_private_network": True, **cfg}), access=AccessManagerPipeline(), root=tmp_path / "ws")


# -- headers -----------------------------------------------------------------------------------------------------------------------
def test_allowed_auth_headers_are_sent_and_json_posts_work(tmp_path, server):
    k = kernel(tmp_path)
    r = k.handle("a", "net_request", {"method": "POST", "url": server + "/x", "json": {"q": 1},
                                      "headers": {"X-Goog-Api-Key": SECRET, "Authorization": "Bearer t0k", "X-RapidAPI-Key": "rk",
                                                  "X-RapidAPI-Host": "h.example", "Accept": "application/json"}})
    assert r["status"] == 200 and json.loads(r["text"])["json"] == {"q": 1}
    h = H.seen[-1]["headers"]
    assert h["x-goog-api-key"] == SECRET and h["authorization"] == "Bearer t0k" and h["x-rapidapi-key"] == "rk" and h["user-agent"] == "pal-code"


@pytest.mark.parametrize("headers", [{"Host": "evil.example"}, {"Cookie": "a=b"}, {"X-Forwarded-For": "1.2.3.4"}, {"Proxy-Authorization": "x"},
                                     {"X-Anything": "y"}])
def test_other_headers_are_refused_so_a_request_cannot_pretend_to_be_something_else(tmp_path, server, headers):
    with pytest.raises(OSKernelError, match="is not allowed"):
        kernel(tmp_path).handle("a", "net_request", {"url": server + "/x", "headers": headers})
    assert H.seen == []


@pytest.mark.parametrize("value", ["a\r\nX-Evil: 1", "a\nb", "a\x00b", "x" * 2001])
def test_header_values_cannot_inject_lines_or_be_huge(tmp_path, server, value):
    with pytest.raises(OSKernelError, match="invalid value"):
        kernel(tmp_path).handle("a", "net_request", {"url": server + "/x", "headers": {"Authorization": value}})


def test_headers_must_be_text_and_the_method_must_be_get_or_post(tmp_path, server):
    k = kernel(tmp_path)
    with pytest.raises(OSKernelError, match="must be text"):
        k.handle("a", "net_request", {"url": server + "/x", "headers": {"Authorization": 5}})
    for m in ("DELETE", "PUT", "PATCH"):
        with pytest.raises(OSKernelError, match="only GET and POST"):
            k.handle("a", "net_request", {"method": m, "url": server + "/x"})


def test_keys_are_not_forwarded_to_a_redirect_target(tmp_path, server):
    k = kernel(tmp_path)
    r = k.handle("a", "net_request", {"url": server + "/redirect", "headers": {"Authorization": "Bearer " + SECRET}})
    echoed = json.loads(r["text"])
    assert "authorization" not in echoed and SECRET not in r["text"]                  # the first hop had it, the redirected one did not
    assert H.seen[0]["headers"]["authorization"] == "Bearer " + SECRET


# -- nothing sensitive in logs or the audit trail ------------------------------------------------------------------------------------
def test_the_query_string_with_a_key_never_reaches_the_audit_trail_or_the_log(tmp_path, server, caplog):
    k = kernel(tmp_path)
    caplog.set_level(logging.DEBUG)
    k.handle("a", "net_request", {"url": f"{server}/search?q=cats&api_key={SECRET}"})
    with pytest.raises(OSKernelError):
        k.handle("a", "net_request", {"url": f"http://localhost/search?api_key={SECRET}"})               # refused: still no key in the trail
    assert k.audit[-2]["target"] == server + "/search" and k.audit[-1]["target"] == "http://localhost/search"
    assert SECRET not in json.dumps(list(k.audit)) and SECRET not in caplog.text
    k.handle("a", "net_get", {"url": f"{server}/search?appid={SECRET}"})
    assert SECRET not in json.dumps(list(k.audit))
    assert H.seen[0]["path"].endswith(f"api_key={SECRET}")                                              # but the server did get it


# -- binary data: down into the workspace ---------------------------------------------------------------------------------------------------
def test_a_binary_answer_is_saved_in_the_workspace_with_the_right_extension(tmp_path, server):
    k = kernel(tmp_path)
    r = k.handle("a", "net_request", {"url": server + "/image", "save_as": "generated/pic-1.{ext}"})
    assert r["saved"] == "generated/pic-1.png" and r["bytes"] == len(PNG) and r["binary"] is True and r["content_type"] == "image/png"
    assert k.workspace("a").read_bytes("generated/pic-1.png") == PNG
    a = k.handle("a", "net_request", {"url": server + "/audio", "save_as": "generated/say.{ext}"})
    assert a["saved"] == "generated/say.flac"
    w = k.handle("a", "net_request", {"method": "POST", "url": server + "/audio", "json": {"inputs": "hi"}, "save_as": "generated/w.{ext}"})
    assert w["saved"] == "generated/w.wav"
    other = kernel(tmp_path)
    assert other.workspace("b").list() == []                                                            # another agent sees nothing


def test_a_binary_answer_without_save_as_reports_its_size_instead_of_returning_bytes(tmp_path, server):
    r = kernel(tmp_path).handle("a", "net_request", {"url": server + "/image"})
    assert r["binary"] is True and r["saved"] == "" and r["text"] == "" and r["bytes"] == len(PNG) and "body" not in r


def test_the_binary_size_cap_is_an_error_not_a_corrupt_file(tmp_path, server):
    k = kernel(tmp_path, max_binary_bytes=1000)
    with pytest.raises(OSKernelError, match="larger than the 1,000-byte limit"):
        k.handle("a", "net_request", {"url": server + "/huge", "save_as": "generated/x.{ext}"})
    assert k.workspace("a").list() == []


def test_error_answers_are_returned_not_saved_and_save_as_names_are_validated_first(tmp_path, server):
    k = kernel(tmp_path)
    r = k.handle("a", "net_request", {"url": server + "/binary-error", "save_as": "generated/x.{ext}"})
    assert r["status"] == 500 and r["saved"] == "" and k.workspace("a").list() == []
    e = k.handle("a", "net_request", {"url": server + "/error"})
    assert e["status"] == 401 and json.loads(e["text"]) == {"error": "bad key"}
    before = len(H.seen)
    for bad in ("../escape.{ext}", "/abs.{ext}", "a\\b.{ext}", "C:\\x.{ext}"):
        with pytest.raises(OSKernelError, match="invalid path|escapes"):
            k.handle("a", "net_request", {"url": server + "/image", "save_as": bad})
    assert len(H.seen) == before                                                                     # no request was made for a bad name


def test_overwriting_a_saved_file_goes_through_the_confirmation_gate(tmp_path, server):
    k = kernel(tmp_path)
    k.handle("a", "net_request", {"url": server + "/image", "save_as": "generated/a.png"})
    with pytest.raises(OSKernelError, match="not permitted"):
        k.handle("a", "net_request", {"url": server + "/image", "save_as": "generated/a.png"})        # same name: asks, declined by default


# -- binary data: up from the workspace -------------------------------------------------------------------------------------------------------
def test_a_workspace_file_can_be_sent_as_the_request_body(tmp_path, server):
    k = kernel(tmp_path)
    k.workspace("a").write_bytes("uploads/cat.png", PNG)
    r = k.handle("a", "net_request", {"method": "POST", "url": server + "/up", "body_path": "uploads/cat.png", "content_type": "image/png",
                                      "headers": {"Authorization": "Bearer t"}})
    assert json.loads(r["text"]) == {"got_bytes": len(PNG), "type": "image/png", "json": None}
    assert H.seen[-1]["body"] == PNG
    with pytest.raises(OSKernelError, match="either a JSON body or a file"):
        k.handle("a", "net_request", {"method": "POST", "url": server + "/up", "body_path": "uploads/cat.png", "json": {"a": 1}})
    with pytest.raises(OSKernelError, match="no such file"):
        k.handle("a", "net_request", {"method": "POST", "url": server + "/up", "body_path": "uploads/missing.png"})
    with pytest.raises(OSKernelError, match="invalid path|escapes"):
        k.handle("a", "net_request", {"method": "POST", "url": server + "/up", "body_path": "../other/secret.png"})


def test_file_read_b64_returns_base64_within_the_limit_and_only_from_the_workspace(tmp_path):
    k = kernel(tmp_path, max_binary_bytes=3000)
    k.workspace("a").write_bytes("uploads/p.png", PNG)
    r = k.handle("a", "file_read_b64", {"path": "uploads/p.png"})
    assert base64.b64decode(r["b64"]) == PNG and r["bytes"] == len(PNG) and r["path"] == "uploads/p.png"
    with pytest.raises(OSKernelError, match="no such file"):
        k.handle("a", "file_read_b64", {"path": "uploads/nope.png"})
    with pytest.raises(OSKernelError, match="invalid path|escapes"):
        k.handle("a", "file_read_b64", {"path": "../x"})
    k.workspace("a").max_binary = 10**9
    k.workspace("a").write_bytes("uploads/big.bin", bytes(5000))
    k.workspace("a").max_binary = 3000
    with pytest.raises(OSKernelError, match="larger than"):
        k.handle("a", "file_read_b64", {"path": "uploads/big.bin"})


def test_the_old_net_get_and_net_post_still_return_text_and_the_new_settings_are_validated(tmp_path, server):
    k = kernel(tmp_path)
    assert json.loads(k.handle("a", "net_get", {"url": server + "/x"})["text"])["ok"] is True
    assert k.handle("a", "net_get", {"url": server + "/image"})["status"] == 200               # a binary answer through net_get is still decoded as before
    with pytest.raises(Exception, match="max_binary_bytes"):
        OsKernelConfig.coerce({"max_binary_bytes": 0})
