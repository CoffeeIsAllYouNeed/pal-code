import io
import logging

import pytest
from tests.fakes import FakeAPI
from tests.helpers import client, make_service

BASE = {"provider": "anthropic", "model": "m", "api_key": "sk-test-secret-key-000000", "route": "api"}


@pytest.fixture
def app(tmp_path):
    svc, api = make_service(tmp_path)
    return client(svc), svc, api


def test_index_health_and_security_headers(app):
    c, *_ = app
    r = c.get("/")
    assert r.status_code == 200 and b"Pal" in r.data
    assert c.get("/healthz").get_json() == {"ok": True}
    for k, v in {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                 "Cache-Control": "no-store"}.items():
        assert r.headers[k] == v


def test_config_never_exposes_secrets(app):
    c, *_ = app
    d = c.get("/api/config").get_json()
    assert d["memory"] == {"enabled": True} and "api_key" not in str(d) and d["providers"]["anthropic"] in (True, False)


def test_a_normal_turn_returns_everything_the_ui_needs(app):
    c, svc, api = app
    r = c.post("/api/chat", data={**BASE, "session_id": "s1", "message": "please explain the cache"})
    d = r.get_json()
    assert r.status_code == 200 and d["reply"] == "api reply" and d["handled_by"] == "api"
    assert set(d) >= {"reply", "sent", "input_tokens", "output_tokens", "saver", "file_errors", "handled_by",
                      "route_reason", "graph", "documents", "memory", "context", "tools", "latency"}
    assert d["latency"]["total"] >= 0 and d["saver"]["tokens_saved"] >= 0


def test_json_body_works_too(app):
    c, *_ = app
    r = c.post("/api/chat", json={**BASE, "session_id": "j1", "message": "explain caching"})
    assert r.status_code == 200 and r.get_json()["reply"] == "api reply"


@pytest.mark.parametrize("data,status,text", [
    ({"session_id": "s", "message": "  "}, 400, "empty message"),
    ({"session_id": "../../etc", "message": "hi"}, 400, "invalid session id"),
])
def test_user_mistakes_are_400_with_a_clear_message(app, data, status, text):
    c, *_ = app
    r = c.post("/api/chat", data=data)
    assert r.status_code == status and text in r.get_json()["error"] and r.get_json()["code"] == "UserInputError"


@pytest.mark.parametrize("provider,model,key,text", [
    ("nope", "m", "k", "unsupported provider"), ("anthropic", "", "k", "choose a model"),
    ("anthropic", "m", "", "no API key")])
def test_provider_settings_are_validated_by_the_real_api_core(tmp_path, monkeypatch, provider, model, key, text):
    from pal.chat import ChatService
    for env in ("ANTHROPIC_API_KEY",):
        monkeypatch.delenv(env, raising=False)
    svc, _ = make_service(tmp_path)
    svc.api_core = ChatService.api_core.__get__(svc)  # not the fake
    r = client(svc).post("/api/chat", data={"session_id": "s", "message": "hi", "provider": provider, "model": model,
                                            "api_key": key, "route": "api"})
    assert r.status_code == 400 and text in r.get_json()["error"]


def test_provider_failures_are_502_and_never_contain_the_key(tmp_path, caplog):
    svc, _ = make_service(tmp_path, api=FakeAPI(boom=ConnectionError("bad key sk-test-secret-key-000000")))
    with caplog.at_level(logging.DEBUG):
        r = client(svc).post("/api/chat", data={**BASE, "session_id": "s", "message": "explain caching"})
    assert r.status_code == 502 and r.get_json()["code"] == "ProviderError"
    assert "sk-test-secret-key-000000" not in r.get_data(as_text=True) and "sk-test-secret-key-000000" not in caplog.text


def test_unexpected_errors_are_500_without_a_traceback(app, caplog):
    c, svc, _ = app
    svc.send = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret internals /home/user/x.py"))
    r = c.post("/api/chat", data={**BASE, "session_id": "s", "message": "hi"})
    assert r.status_code == 500 and r.get_json() == {"error": "internal error", "code": "InternalError"}
    assert "internals" not in r.get_data(as_text=True) and "Traceback" not in r.get_data(as_text=True)
    assert "secret internals" in caplog.text  # ... but it is logged for the operator


def test_unknown_routes_and_wrong_methods_are_json(app):
    c, *_ = app
    r = c.get("/nope")
    assert r.status_code == 404 and r.is_json
    r = c.get("/api/chat")
    assert r.status_code == 405 and r.is_json


def test_too_large_uploads_are_413(tmp_path):
    svc, _ = make_service(tmp_path, server={"max_upload_mb": 1})
    r = client(svc).post("/api/chat", data={**BASE, "session_id": "s", "message": "read",
                                            "files": (io.BytesIO(b"x" * 2_000_000), "big.txt")})
    assert r.status_code == 413 and r.get_json()["error"] == "upload too large"


def test_bad_files_are_reported_per_file_and_the_turn_still_runs(app):
    c, svc, api = app
    r = c.post("/api/chat", data={**BASE, "session_id": "s", "message": "read these", "files": [
        (io.BytesIO(b"\xd0\xcf\x11\xe0"), "old.doc"), (io.BytesIO(b"hello file"), "ok.txt")]})
    d = r.get_json()
    assert r.status_code == 200 and len(d["file_errors"]) == 1 and "old.doc" in d["file_errors"][0]
    assert "hello file" in d["sent"]


def test_only_bad_files_and_no_message_is_a_400(app):
    c, *_ = app
    r = c.post("/api/chat", data={**BASE, "session_id": "s", "message": "",
                                  "files": (io.BytesIO(b"\xd0\xcf\x11\xe0"), "old.doc")})
    assert r.status_code == 400 and "old.doc" in r.get_json()["error"]


def test_upload_file_names_are_reduced_to_a_base_name(app):
    c, *_ = app
    r = c.post("/api/chat", data={**BASE, "session_id": "s", "message": "read",
                                  "files": (io.BytesIO(b"body text"), "../../etc/passwd.txt")})
    assert "[file: passwd.txt]" in r.get_json()["sent"] and "../" not in r.get_json()["sent"]


def test_oversized_converted_text_is_rejected(tmp_path):
    svc, _ = make_service(tmp_path, documents={"max_file_chars": 1000})
    r = client(svc).post("/api/chat", data={**BASE, "session_id": "s", "message": "read",
                                            "files": (io.BytesIO(b"x" * 5000), "big.txt")})
    assert "over 1,000 characters" in r.get_json()["file_errors"][0]


def test_metrics_endpoints(app):
    c, *_ = app
    assert c.get("/api/metrics").get_json()["summary"] == {"turns": 0}
    c.post("/api/chat", data={"session_id": "s", "message": "12*12"})
    c.post("/api/chat", data={**BASE, "session_id": "s", "message": "explain caching"})
    d = c.get("/api/metrics").get_json()
    assert d["summary"]["turns"] == 2 and d["summary"]["api_calls_avoided"] == 1 and len(d["recent"]) == 2
    assert c.post("/api/metrics/reset").get_json() == {"ok": True}
    assert c.get("/api/metrics").get_json()["summary"] == {"turns": 0}


def test_memory_endpoints(app):
    c, *_ = app
    c.post("/api/chat", data={**BASE, "session_id": "s", "message": "my dog is called Biscuit and hates storms"})
    body = c.get("/api/memory").get_json()
    assert body["enabled"] is True and body["notes"] == 1 and body["stats"]["notes"] == 1 and {"ram_count", "disk_count", "since_tidy"} <= set(body["stats"])
    assert c.delete("/api/memory").get_json() == {"cleared": 1}
    assert c.get("/api/memory").get_json()["notes"] == 0


def test_memory_endpoints_when_disabled(tmp_path):
    svc, _ = make_service(tmp_path, memory={"enabled": False})
    c = client(svc)
    assert c.get("/api/memory").get_json() == {"enabled": False, "notes": 0, "stats": None}
    assert c.delete("/api/memory").get_json() == {"cleared": 0}


def test_reset_and_experiment(app):
    c, svc, api = app
    c.post("/api/chat", data={**BASE, "session_id": "s", "message": "explain caching"})
    assert c.post("/api/reset", json={"session_id": "s"}).get_json() == {"ok": True}
    assert svc.sessions.get("s") is None
    assert c.post("/api/reset", json={}).status_code == 200
    r = c.post("/api/experiment", json={"text": "please explain the cache", "methods": None})
    d = r.get_json()
    assert r.status_code == 200 and d["saver"]["tokens_original"] >= d["saver"]["tokens_reduced"] and d["api"] is None
    assert c.post("/api/experiment", json={"text": " "}).status_code == 400


def test_the_api_key_is_never_logged(app, caplog):
    c, *_ = app
    with caplog.at_level(logging.DEBUG):
        c.post("/api/chat", data={**BASE, "session_id": "s", "message": "explain caching"})
        c.post("/api/experiment", json={"text": "hello there", "api_key": "sk-test-secret-key-000000"})
    assert "sk-test-secret-key-000000" not in caplog.text


def test_context_window_for_the_tokens_left_counter(app):
    app = app[0]
    r = app.get("/api/context_window?provider=local&model=mistral:7b")
    assert r.status_code == 200 and r.get_json() == {"window": 8192}
