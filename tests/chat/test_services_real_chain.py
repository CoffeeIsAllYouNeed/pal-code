"""The whole chain with nothing faked except the address: ToolBox -> real OS kernel interface -> real `requests` -> a real HTTP server on
this machine. Only the host name is redirected to localhost, so the headers, bodies, binary files and logs are the real ones."""
import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest
import requests as real_requests

from pal.access_manager.pipeline import AccessManagerPipeline
from pal.chat.tools import ToolBox, ToolCall
from pal.config import OsKernelConfig, ToolsConfig
from pal.os_kernel import OSKernel, network

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(200))
KEYS = {"SERPAPI_API_KEY": "serp-secret-111", "WOLFRAM_APP_ID": "wolf-secret-333", "HF_TOKEN": "hf_secret_555", "RAPIDAPI_KEY": "rapid-secret-444"}
SEEN: list = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _go(self, body=b""):
        SEEN.append({"host": self.headers.get("X-Test-Orig-Host"), "method": self.command, "path": self.path, "body": body,
                     "headers": {k.lower(): v for k, v in self.headers.items()}})
        path = urlsplit(self.path).path
        if path.endswith("blip-image-captioning-large"):
            self._send(json.dumps([{"generated_text": "a cat on a sofa"}]).encode(), "application/json")
        elif path.endswith("vilt-b32-finetuned-vqa"):
            self._send(json.dumps([{"score": 0.9, "answer": "two"}]).encode(), "application/json")
        elif path.endswith("FLUX.1-schnell"):
            self._send(PNG, "image/png")
        elif path.endswith("mms-tts-eng"):
            self._send(b"fLaC" + bytes(100), "audio/flac")
        elif path.endswith("whisper-large-v3"):
            self._send(json.dumps({"text": "hello there"}).encode(), "application/json")
        elif path.endswith("llm-api"):
            self._send(b"Result: osmium", "text/plain")
        elif path.endswith("search.json"):
            self._send(json.dumps({"organic_results": [{"title": "Alpha", "link": "https://a.test", "snippet": "s"}]}).encode(), "application/json")
        else:
            self._send(b"{}", "application/json", 404)

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._go()

    def do_POST(self):
        self._go(self.rfile.read(int(self.headers.get("Content-Length", 0))))


@pytest.fixture
def chain(tmp_path, monkeypatch):
    SEEN.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    local = f"http://127.0.0.1:{server.server_address[1]}"
    original = real_requests.request

    def redirect(method, url, **kw):
        parts = urlsplit(url)
        target = local + parts.path + (f"?{parts.query}" if parts.query else "")
        kw["headers"] = {**kw.get("headers", {}), "X-Test-Orig-Host": parts.hostname}
        return original(method, target, **kw)

    monkeypatch.setattr(real_requests, "request", redirect)
    monkeypatch.setattr(network, "check_url", lambda *a, **k: None)             # the real address is not reachable from a test
    kernel = OSKernel(OsKernelConfig.coerce({}), access=AccessManagerPipeline(), root=tmp_path / "ws")
    box = ToolBox("chat-s", ToolsConfig(services=["all"]), lambda t: len(t.split()), session_id="s", os_kernel=kernel, env=KEYS)
    yield box, kernel
    server.shutdown()


def run(box, name, **args):
    return box.run(ToolCall(name, args))


def test_image_caption_uploads_the_raw_file_with_the_token_in_a_header(chain):
    box, kernel = chain
    kernel.workspace("chat-s").write_bytes("uploads/cat.png", PNG)
    out = run(box, "image_caption", path="uploads/cat.png")
    assert out.ok and out.text == "a cat on a sofa"
    req = SEEN[0]
    assert req["host"] == "router.huggingface.co" and req["body"] == PNG and req["method"] == "POST"
    assert req["headers"]["content-type"] == "image/png" and req["headers"]["authorization"] == "Bearer hf_secret_555"
    assert "hf_secret_555" not in req["path"]


def test_visual_question_answering_sends_the_image_as_base64_inside_json(chain):
    box, kernel = chain
    kernel.workspace("chat-s").write_bytes("uploads/cats.png", PNG)
    out = run(box, "visual_question_answering", path="uploads/cats.png", question="how many cats?")
    assert out.text == "two (90%)"
    sent = json.loads(SEEN[0]["body"])
    assert base64.b64decode(sent["inputs"]["image"]) == PNG and sent["inputs"]["question"] == "how many cats?"


def test_text_to_image_saves_the_binary_answer_in_the_workspace_with_the_right_extension(chain):
    box, kernel = chain
    out = run(box, "text_to_image", prompt="a red fox in snow")
    assert out.ok and len(box.saved_files) == 1
    path = box.saved_files[0]
    assert path.startswith("generated/image-") and path.endswith(".png")
    assert kernel.workspace("chat-s").read_bytes(path) == PNG and f"{len(PNG):,} bytes" in out.text
    assert json.loads(SEEN[0]["body"]) == {"inputs": "a red fox in snow"} and SEEN[0]["headers"]["authorization"] == "Bearer hf_secret_555"
    again = run(box, "text_to_image", prompt="a red fox in snow")                                   # same name: overwrite asks, declined
    assert not again.ok and "not permitted" in again.text


def test_text_to_audio_and_speech_to_text_round_trip_audio_files(chain):
    box, kernel = chain
    assert run(box, "text_to_audio", text="hello there").ok and box.saved_files[0].endswith(".flac")
    audio = kernel.workspace("chat-s").read_bytes(box.saved_files[0])
    assert audio.startswith(b"fLaC")
    out = run(box, "speech_to_text", path=box.saved_files[0])                                        # generated audio can be transcribed
    assert out.text == "hello there" and SEEN[-1]["body"] == audio and SEEN[-1]["headers"]["content-type"] == "audio/flac"


def test_wolfram_sends_the_app_id_as_a_bearer_header_and_never_in_the_address(chain):
    box, _ = chain
    out = run(box, "wolfram_alpha", query="10 densest elemental metals")
    assert out.text == "Result: osmium"
    req = SEEN[0]
    assert req["host"] == "www.wolframalpha.com" and req["headers"]["authorization"] == "Bearer wolf-secret-333"
    assert "wolf-secret-333" not in req["path"] and "input=10+densest+elemental+metals" in req["path"]


def test_serpapi_gets_its_key_in_the_query_but_it_never_reaches_the_audit_trail_or_any_log(chain, caplog):
    box, kernel = chain
    caplog.set_level(logging.DEBUG)
    out = run(box, "google_search", query="cats")
    assert out.text == "1. Alpha - s (https://a.test)"
    assert "api_key=serp-secret-111" in SEEN[0]["path"]                                              # the service did receive it
    assert "serp-secret-111" not in json.dumps(list(kernel.audit)) and "serp-secret-111" not in caplog.text
    assert kernel.audit[-1]["target"].endswith("/search.json") and "?" not in kernel.audit[-1]["target"]


def test_the_logging_level_of_urllib3_is_restored_after_a_request(chain):
    box, _ = chain
    lg = logging.getLogger("urllib3")
    before = lg.level
    run(box, "moon_phase")
    run(box, "wolfram_alpha", query="pi")
    assert lg.level == before
