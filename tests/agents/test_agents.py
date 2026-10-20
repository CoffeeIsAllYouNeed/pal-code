import io
import json
import time

import pytest

from pal import cli
from pal.agents import AGENTS
from pal.errors import ProviderError, UserInputError
from pal.sdk.agent import agent_key, get_spec, list_agents
from tests.fakes import FakeAPI
from tests.helpers import client, doc, make_service, make_zip, send

CODE_KEY, DOCS_KEY = "k-coding-111", "k-docs-222"


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_CODING", "GEMINI_API_KEY_DOCS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_CODING", CODE_KEY)
    monkeypatch.setenv("GEMINI_API_KEY_DOCS", DOCS_KEY)


def build(tmp_path, **kw):
    """Service whose API core records how it was built, so the tests can see provider, model and key."""
    kw.setdefault("scheduler", {"api_min_interval": {"gemini": 0.0}})
    svc, api = make_service(tmp_path, **kw)
    built = []
    svc.api_core = lambda provider, model, key=None: built.append((provider, model, key)) or api
    return svc, api, built


def ask(svc, msg, agent, atts=None, sid="s1"):
    return svc.send(sid, "", "", None, msg, atts or [], agent=agent)


def test_two_agents_with_small_specs():
    assert set(AGENTS) == {"coding", "docs"}
    for a in AGENTS.values():
        assert len(a.spec.role) < 400 and a.spec.max_tokens <= 2048 and "read_source" in a.spec.tools
    assert get_spec(AGENTS, "coding").key_env == "GEMINI_API_KEY_CODING"
    assert get_spec(AGENTS, "docs").key_env == "GEMINI_API_KEY_DOCS"
    with pytest.raises(UserInputError, match="unknown agent"):
        get_spec(AGENTS, "chef")


def test_each_agent_uses_its_own_key_and_the_shared_key_is_a_fallback(monkeypatch):
    assert agent_key(AGENTS["coding"].spec) == CODE_KEY and agent_key(AGENTS["docs"].spec) == DOCS_KEY
    monkeypatch.delenv("GEMINI_API_KEY_DOCS")
    assert agent_key(AGENTS["docs"].spec) is None
    monkeypatch.setenv("GEMINI_API_KEY", "shared")
    assert agent_key(AGENTS["docs"].spec) == "shared" and agent_key(AGENTS["coding"].spec) == CODE_KEY
    assert [a["ready"] for a in list_agents(AGENTS, {})] == [False, False]


def test_agent_turn_uses_gemini_its_key_the_api_route_and_its_role(tmp_path):
    svc, api, built = build(tmp_path)
    out = ask(svc, "what is 2+2 in python", "coding")
    assert built[-1] == ("gemini", svc.settings.agents.model, CODE_KEY)
    assert out.handled_by == "api"                       # never answered locally or by the direct answerer
    system = api.seen[-1][0]["content"]
    assert "Role: coding agent" in system and "No fluff" in system     # role + the normal output rules
    ask(svc, "write the readme", "docs", sid="s2")
    assert built[-1][2] == DOCS_KEY and "Role: documentation agent" in api.seen[-1][0]["content"]
    send(svc, "hello there friend", route="api", sid="s3")   # a normal turn after an agent turn has no role
    assert "Role:" not in api.seen[-1][0]["content"]


def test_token_savers_still_apply_to_agent_turns(tmp_path):
    svc, api, _ = build(tmp_path)
    out = ask(svc, "please explain the caching layer", "coding")
    assert out.saver.tokens_saved > 0 and api.seen[-1][-1]["content"] == "explain caching layer"


def test_agent_output_is_capped_per_agent(tmp_path):
    svc, api, _ = build(tmp_path)
    caps = []
    orig = api.chat
    api.chat = lambda syscall: caps.append(syscall.max_tokens) or orig(syscall)
    ask(svc, "explain the cache", "coding")
    ask(svc, "write the docs", "docs", sid="s2")
    assert caps == [AGENTS["coding"].spec.max_tokens, AGENTS["docs"].spec.max_tokens]


def test_missing_key_and_disabled_agents_give_clear_errors(tmp_path, monkeypatch):
    svc, api, _ = build(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY_DOCS")
    with pytest.raises(UserInputError, match="GEMINI_API_KEY_DOCS"):
        ask(svc, "docs please", "docs")
    with pytest.raises(UserInputError, match="unknown agent"):
        ask(svc, "hi there", "chef")
    svc2, _, _ = build(tmp_path / "b", agents={"enabled": False})
    with pytest.raises(UserInputError, match="turned off"):
        ask(svc2, "hi there", "coding")
    assert api.seen == []


def call(tool, /, **args):
    return f'<tool>{json.dumps({"name": tool, "args": args})}</tool>'


def test_agents_get_only_their_read_tools(tmp_path):
    api = FakeAPI([call("calculator", expression="1+1"), "done"])
    svc, api, _ = build(tmp_path, api=api)
    out = ask(svc, "compute 1 plus 1", "coding")        # the digit-free message offers no tools at all
    assert out.tools == [] and "Tools:" not in api.seen[-1][0]["content"]
    # a tool outside the allow-list is refused even if the model asks for it
    box = svc.sessions.get("s1").toolbox
    box.allow = AGENTS["coding"].spec.tools
    res = box.run(box.parse(call("calculator", expression="1+1")))
    assert not res.ok and "not available to this agent" in res.text
    box.allow = None
    assert box.run(box.parse(call("calculator", expression="1+1"))).text == "2"


def test_coding_agent_reads_a_project_through_the_graph_and_the_read_tool(tmp_path):
    api = FakeAPI([call("read_source", path="shop/pay.py", start=1, end=5), "It charges the card."])
    svc, api, _ = build(tmp_path, api=api)
    att = svc.graph.ingest("s1", "shop.zip", make_zip())
    out = ask(svc, "how does checkout charge the card", "coding", [att])
    assert out.tools and out.tools[0]["name"] == "read_source" and out.tools[0]["ok"]
    assert "read_source(path, start, end)" in api.seen[0][0]["content"] and out.graph["nodes"] == 1
    assert out.reply == "It charges the card."


def test_documentation_agent_gets_documents_inline_then_only_relevant_parts(tmp_path):
    svc, api, _ = build(tmp_path)
    ask(svc, "write a readme from these notes", "docs", [doc("notes.md", "Papertrail searches notes by meaning.")])
    assert "Papertrail searches notes by meaning." in api.seen[-1][-1]["content"]
    ask(svc, "add a features section", "docs")           # nothing in the notes matches: no chunk is sent at all
    unrelated = " ".join(m["content"] for m in api.seen[-1] if m["role"] == "user")
    assert "Papertrail searches notes by meaning." not in unrelated and "[file: notes.md" in unrelated
    ask(svc, "add a section on how Papertrail searches notes", "docs")
    last = api.seen[-1][-1]["content"]                   # a matching question brings back just the relevant part
    assert "<documents>" in last and last.count("Papertrail searches notes by meaning.") == 1


def test_agent_calls_are_spaced_out_for_free_tiers(tmp_path):
    svc, api, _ = build(tmp_path, scheduler={"api_min_interval": {"gemini": 0.3}})
    stamps = []
    orig = api.chat
    api.chat = lambda syscall: stamps.append(time.monotonic()) or orig(syscall)
    ask(svc, "explain the cache", "coding")
    ask(svc, "and the eviction", "coding")
    assert stamps[1] - stamps[0] >= 0.28
    ask(svc, "write docs", "docs", sid="s2")             # another agent has its own key and its own schedule
    assert stamps[2] - stamps[1] < 0.28


def test_rate_limit_errors_are_retried_by_the_line_then_explained_and_never_leak_the_key(tmp_path):
    api = FakeAPI(boom=RuntimeError(f"429 RESOURCE_EXHAUSTED for key {CODE_KEY}"))
    svc, _, _ = build(tmp_path, api=api, scheduler={"api_min_interval": {"gemini": 0.0}, "api_backoff_seconds": 0.01})
    with pytest.raises(ProviderError) as e:
        ask(svc, "explain the cache", "coding")
    assert len(api.seen) == 3                                   # the first call plus two retries, then it gives up
    assert "rate limit or quota" in str(e.value) and CODE_KEY not in str(e.value)


# -- web ----------------------------------------------------------------------------------------------------
def test_config_lists_agents_and_whether_they_are_ready(tmp_path, monkeypatch):
    svc, _, _ = build(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY_DOCS")
    agents = {a["id"]: a for a in client(svc).get("/api/config").get_json()["agents"]}
    assert agents["coding"]["ready"] and not agents["docs"]["ready"] and agents["docs"]["key_env"] == "GEMINI_API_KEY_DOCS"
    assert CODE_KEY not in json.dumps(agents)


def test_chat_endpoint_runs_an_agent_without_a_browser_key(tmp_path):
    svc, api, built = build(tmp_path)
    r = client(svc).post("/api/chat", data={"session_id": "w1", "message": "explain the cache", "agent": "coding"})
    assert r.status_code == 200 and r.get_json()["handled_by"] == "api" and built[-1][2] == CODE_KEY


def test_chat_endpoint_agent_errors_are_400(tmp_path, monkeypatch):
    svc, _, _ = build(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY_CODING")
    r = client(svc).post("/api/chat", data={"session_id": "w1", "message": "hi there", "agent": "coding"})
    assert r.status_code == 400 and "GEMINI_API_KEY_CODING" in r.get_json()["error"]


def test_agent_upload_of_a_zip_becomes_a_code_graph(tmp_path):
    svc, api, _ = build(tmp_path)
    r = client(svc).post("/api/chat", data={"session_id": "w2", "message": "how does checkout work", "agent": "coding",
                                            "files": (io.BytesIO(make_zip()), "shop.zip")})
    assert r.status_code == 200 and r.get_json()["graph"]["nodes"] == 1


# -- CLI ------------------------------------------------------------------------------------------------------
def test_cli_agent_runs_a_turn_with_an_attached_file_and_reports_tokens(tmp_path, monkeypatch, capsys):
    svc, api, _ = build(tmp_path)
    monkeypatch.setattr("pal.chat.ChatService", lambda settings: svc)
    f = tmp_path / "notes.md"
    f.write_text("Papertrail searches notes by meaning.", encoding="utf-8")
    assert cli.main(["agent", "docs", "write a readme", "--file", str(f)]) == 0
    out = capsys.readouterr()
    assert "api reply" in out.out and "tokens" in out.err
    assert "Papertrail searches notes by meaning." in api.seen[-1][-1]["content"]


def test_cli_agent_reports_errors_without_a_traceback(tmp_path, monkeypatch, capsys):
    svc, _, _ = build(tmp_path)
    monkeypatch.setattr("pal.chat.ChatService", lambda settings: svc)
    monkeypatch.delenv("GEMINI_API_KEY_DOCS")
    assert cli.main(["agent", "docs", "write a readme"]) == 1
    assert "GEMINI_API_KEY_DOCS" in capsys.readouterr().err
    assert cli.main(["agent", "docs", "x", "--file", str(tmp_path / "missing.md")]) in (1, 2)


def test_doctor_reports_agent_keys_without_printing_them(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY_DOCS")
    from pal.config import load_settings
    checks = {c.name: c for c in cli.run_doctor(load_settings(env={}), env={"GEMINI_API_KEY_CODING": CODE_KEY},
                                                which=lambda n: None, has_module=lambda m: False,
                                                http_get=lambda u: {"models": []})}
    assert checks["agent: coding"].status == "ok" and checks["agent: docs"].status == "info"
    assert CODE_KEY not in "".join(c.detail for c in checks.values())
