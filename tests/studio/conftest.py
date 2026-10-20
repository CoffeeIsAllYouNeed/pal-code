"""Shared fakes for the Studio tests: a scripted model that plays every role, and helpers to build a Studio app around it."""
import re
import threading
import time
from types import SimpleNamespace

import pytest

from pal.llm_wrapper.components.base import (
    DeploymentType,
    LLMResponse,
    LLMSyscall,
    LLMWrapper,
    WrapperCapabilities,
    WrapperConfig,
)
from tests.helpers import make_service

INDEX = '<!doctype html>\n<html>\n<head><link rel="stylesheet" href="style.css"></head>\n<body>\n<div id="app"></div>\n<script src="app.js"></script>\n</body>\n</html>'
GOOD = {"Designer": [("index.html", INDEX), ("style.css", "body { font-family: sans-serif; }")],
        "Coder": [("app.js", "document.getElementById('app').textContent = 'hello';")],
        "Docs writer": [("README.md", "# App\n\nOpen index.html.")],
        "Tester": [("test_main.py", "def test_ok():\n    assert True\n")],
        "Reviewer": [("index.html", INDEX)]}


def files_reply(files, note="done"):
    return "\n\n".join(f"### FILE: {p}\n```{p.rsplit('.', 1)[-1]}\n{c}\n```" for p, c in files) + f"\n\nNOTE: {note}"


class ScriptedCore(LLMWrapper):
    """Plays the planner, the builder's model and Pal's small helper (router, input saver, verifier), told apart by the system prompt."""
    name = "scripted"

    def __init__(self, files=None, delay=0.0, planner_json=None, fail_on_step=None):
        super().__init__(WrapperConfig(backend="ollama", model="scripted"))
        self.files = {**GOOD, **(files or {})}
        self.delay, self.planner_json, self.fail_on_step = delay, planner_json, fail_on_step
        self.calls = []
        self.lock = threading.Lock()

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        msgs = syscall.messages
        system, user = msgs[0]["content"], msgs[-1]["content"]
        with self.lock:
            self.calls.append((system[:40], user[:60]))
        if self.delay:
            time.sleep(self.delay)
        m = re.search(r"<text>\n(.*)\n</text>", user, re.S)
        if system.startswith("You are the planner"):
            pj = self.planner_json
            with self.lock:
                text = (pj.pop(0) if len(pj) > 1 else pj[0]) if isinstance(pj, list) else (pj if pj is not None else "{}")
        elif system.startswith("You route"):
            text = '{"route": "api", "reason": "x"}'
        elif system.startswith("You audit"):
            text = '{"ok": true, "lost": []}'
        elif m:
            text = f"<text>\n{m.group(1)}\n</text>"
        elif user == "Reply with the single word: ready":
            text = "ready"
        elif "problems were found" in user:
            text = files_reply([("style.css", "body { color: black; }"), ("app.js", "let x = 1;")], "fixed")
        else:
            step = re.search(r"Step (\d+) (?:of \d+ )?\(([^)]+)\)", user)
            if self.fail_on_step and step and int(step.group(1)) == self.fail_on_step:
                raise ConnectionError("the model went away")
            role = step.group(2) if step else "Coder"
            text = files_reply(self.files.get(role, GOOD["Coder"]))
        return LLMResponse(text=text, prompt_tokens=sum(len(x["content"]) for x in msgs) // 4, completion_tokens=len(text) // 4, raw=None, meta={})

    def token_count(self, text):
        return max(1, len(text) // 4)

    def capabilities(self):
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER)


def local_info(models=("llama3.2:1b", "llama3.1:8b-instruct-q4_K_M", "nomic-embed-text:latest"), ok=True):
    items = [{"name": n, "params_b": None, "size_gb": 4.9 if "8b" in n else (1.3 if "1b" in n else 0.3), "family": "", "quant": "", "embedding": "embed" in n, "label": n} for n in models]
    if not ok:
        return {"ok": False, "url": "http://x", "models": [], "recommended": None, "error": "Ollama is not reachable"}
    return {"ok": True, "url": "http://x", "models": items, "recommended": "llama3.1:8b-instruct-q4_K_M", "error": None}


def planner_json(**over):
    import json
    plan = {"product": {"name": "Todo app", "summary": "A to-do list page.", "kind": "web_app"}, "roles": ["designer", "coder", "docs", "reviewer"], "tools": ["calculator"],
            "permissions": {"network_access": False}, "model": {"mode": "local", "model": "llama3.1:8b-instruct-q4_K_M"}, "reasons": {"role:designer": "it has a page"}, **over}
    return json.dumps(plan)


@pytest.fixture
def studio(tmp_path):
    """(test client, the scripted core, the run store) with the Studio wired to the scripted model."""
    from pal.web.app import create_app
    from tests.fakes import FakeAPI
    core = ScriptedCore(planner_json=planner_json())
    svc, _ = make_service(tmp_path, api=FakeAPI(["ok"]))
    app = create_app(service=svc, studio={"planner_core": lambda model: core, "core_factory": lambda plan, key: core, "discover_fn": lambda url: local_info()})
    return app.test_client(), core, app.extensions["studio_runs"]


def wait(run, timeout=20):
    t0 = time.time()
    while run.status in ("queued", "running"):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"run still {run.status}")
        time.sleep(0.02)
    return run


NS = SimpleNamespace
