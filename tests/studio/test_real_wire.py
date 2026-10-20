"""The Studio wired the real way: Ollama's model list over HTTP, the planner and the builder going through the real LiteLLM to an Ollama-shaped server.
No test hooks: only the server on the other end is fake (it exists only here). Skipped when LiteLLM is not installed."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pal.llm_wrapper.components.base import LLMSyscall
from pal.web.app import create_app
from tests.fakes import FakeAPI
from tests.helpers import make_service
from tests.studio.conftest import ScriptedCore, planner_json

TAGS = {"models": [
    {"name": "llama3.2:1b", "size": 1_300_000_000, "details": {"parameter_size": "1.2B", "family": "llama"}},
    {"name": "llama3.1:8b-instruct-q4_K_M", "size": 4_900_000_000, "details": {"parameter_size": "8.0B", "family": "llama"}},
    {"name": "nomic-embed-text:latest", "size": 274_000_000, "details": {"parameter_size": "137M", "family": "nomic-bert"}}]}


class FakeOllama(BaseHTTPRequestHandler):
    core = ScriptedCore(planner_json=planner_json(model={"mode": "local", "model": "llama3.1:8b-instruct-q4_K_M"}))
    seen_models: list = []
    seen_options: list = []

    def log_message(self, *a):
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json(TAGS if self.path.startswith("/api/tags") else {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "messages" not in body:
            self._json({})
            return
        FakeOllama.seen_models.append(body["model"])
        FakeOllama.seen_options.append((body["messages"][0]["content"][:30], body.get("options", {}), body.get("keep_alive")))
        r = FakeOllama.core.chat(LLMSyscall(messages=body["messages"]))
        self._json({"model": body["model"], "message": {"role": "assistant", "content": r.text}, "done": True, "prompt_eval_count": r.prompt_tokens, "eval_count": r.completion_tokens})


@pytest.fixture(autouse=True)
def _a_computer_with_a_gpu(monkeypatch):
    """These tests talk to a local (Ollama-shaped) server, which Pal only uses on a computer with a GPU."""
    from pal.terminal import backend
    from pal.terminal.hardware import from_spec
    monkeypatch.setattr(backend, "_HW", from_spec("gpu:8"))


@pytest.fixture
def real_litellm():
    pytest.importorskip("litellm")
    from pal.llm_wrapper.components import litellm_wrapper
    litellm_wrapper._module = None
    yield
    litellm_wrapper._module = None


@pytest.fixture
def ollama():
    FakeOllama.seen_models = []
    FakeOllama.seen_options = []
    FakeOllama.core = ScriptedCore(planner_json=planner_json(model={"mode": "local", "model": "llama3.1:8b-instruct-q4_K_M"}))
    s = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def studio_client(tmp_path, ollama):
    svc, _ = make_service(tmp_path, api=FakeAPI(["ok"]), local_llm={"backend": "ollama", "model": "llama3.2:1b", "host": ollama})
    app = create_app(service=svc)                                           # no test hooks: the real discovery, planner and builder
    return app.test_client(), app.extensions["studio_runs"]


def wait(run, timeout=60):
    t0 = time.time()
    while run.status in ("queued", "running"):
        assert time.time() - t0 < timeout, run.events[-3:]
        time.sleep(0.05)
    return run


def test_the_catalogue_finds_the_strongest_installed_model_over_http(tmp_path, ollama, real_litellm):
    c, _ = studio_client(tmp_path, ollama)
    local = c.get("/api/studio/catalog").get_json()["local"]
    assert local["ok"] and local["recommended"] == "llama3.1:8b-instruct-q4_K_M" and [m["name"] for m in local["models"] if m["embedding"]] == ["nomic-embed-text:latest"]


def test_the_bot_plans_with_the_strongest_model_through_the_real_litellm(tmp_path, ollama, real_litellm):
    c, _ = studio_client(tmp_path, ollama)
    r = c.post("/api/studio/plan", json={"command": "a to-do list web page"}).get_json()
    assert r["source"] == "model" and r["planner_model"] == "llama3.1:8b-instruct-q4_K_M" and r["plan"]["roles"] == ["designer", "coder", "docs", "reviewer"]
    assert set(FakeOllama.seen_models) == {"llama3.1:8b-instruct-q4_K_M"}            # the bot and nothing else used the strongest model, never the small one


def test_a_whole_build_runs_through_the_real_litellm_and_writes_the_project(tmp_path, ollama, real_litellm):
    c, runs = studio_client(tmp_path, ollama)
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    rid = c.post("/api/studio/run", json={"plan": plan}).get_json()["run_id"]
    run = wait(runs.get(rid))
    assert run.status == "done" and sorted(run.arms["main"].build.files) == ["README.md", "app.js", "index.html", "style.css"]
    t = run.arms["main"].totals
    assert t["main_tokens"] > 0 and t["helper_tokens"] > 0 and t["steps_done"] == 4 and t["problems_left"] == 0
    assert (tmp_path / "home" / "studio" / rid / "main" / "todo-app" / "index.html").is_file()


def test_building_twice_through_the_real_litellm_gives_a_comparison(tmp_path, ollama, real_litellm):
    c, runs = studio_client(tmp_path, ollama)
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    plan["metrics"] = True
    run = wait(runs.get(c.post("/api/studio/run", json={"plan": plan}).get_json()["run_id"]), 90)
    cmp = run.comparison
    assert run.status == "done" and cmp and cmp["baseline"]["files"] == cmp["pal"]["files"] == 4 and cmp["pal"]["helper_tokens"] > 0 and cmp["baseline"]["helper_tokens"] == 0
    assert cmp["baseline"]["main_tokens"] > 0 and isinstance(cmp["main_tokens_saved_pct"], float)


def test_a_model_that_is_not_installed_is_replaced_by_the_recommended_one_and_the_person_is_told(tmp_path, ollama, real_litellm):
    c, _ = studio_client(tmp_path, ollama)
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    plan["model"] = {"mode": "local", "model": "gpt-9:999b"}
    r = c.post("/api/studio/validate", json={"plan": plan}).get_json()
    assert r["plan"]["model"]["model"] == "llama3.1:8b-instruct-q4_K_M" and any("gpt-9:999b is not installed" in n for n in r["notes"])


def test_with_ollama_down_the_catalogue_says_so_and_a_run_explains_what_to_do(tmp_path, real_litellm):
    c, runs = studio_client(tmp_path, "http://127.0.0.1:9")
    cat = c.get("/api/studio/catalog").get_json()
    assert not cat["local"]["ok"] and "ollama serve" in cat["local"]["error"]
    plan = c.post("/api/studio/plan", json={"command": "a landing page"}).get_json()
    assert plan["source"] == "fallback" and plan["plan"]["model"]["model"] == ""
    r = c.post("/api/studio/run", json={"plan": plan["plan"]})
    run = wait(runs.get(r.get_json()["run_id"]))
    assert run.status == "error" and "no local model is chosen" in json.dumps(run.events)


def test_every_local_call_carries_a_context_window_and_keep_alive_so_ollama_does_not_silently_cut_the_prompt(tmp_path, ollama, real_litellm):
    c, runs = studio_client(tmp_path, ollama)
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    wait(runs.get(c.post("/api/studio/run", json={"plan": plan}).get_json()["run_id"]))
    planner = [o for s, o, k in FakeOllama.seen_options if s.startswith("You are the planner")]
    others = [o for s, o, k in FakeOllama.seen_options if not s.startswith("You are the planner")]
    assert planner and all(o["num_ctx"] == 8192 for o in planner) and others and all(o["num_ctx"] == 16384 for o in others)
    assert c.get("/api/studio/catalog").get_json()["limits"]["num_ctx"] == 16384


def test_the_context_windows_come_from_the_studio_settings(tmp_path, ollama, real_litellm):
    svc, _ = make_service(tmp_path, api=FakeAPI(["ok"]), local_llm={"backend": "ollama", "model": "llama3.2:1b", "host": ollama}, studio={"num_ctx": 32768, "planner_num_ctx": 4096})
    c = create_app(service=svc).test_client()
    assert c.get("/api/studio/catalog").get_json()["limits"]["num_ctx"] == 32768
    c.post("/api/studio/plan", json={"command": "x"})
    assert [o["num_ctx"] for s_, o, k in FakeOllama.seen_options if s_.startswith("You are the planner")] == [4096]
