import pytest

from pal import cli
from pal.config import load_settings
from tests.fakes import FakeAPI


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path / "home"))


def by_name(checks):
    return {c.name: c for c in checks}


def doctor(**kw):
    kw.setdefault("http_get", lambda url: {"models": [{"name": "llama3:latest"}]})
    return by_name(cli.run_doctor(load_settings(env={}), env=kw.pop("env", {}), **kw))


def test_healthy_machine():
    c = doctor(which=lambda n: "/usr/bin/" + n, has_module=lambda m: True, env={"ANTHROPIC_API_KEY": "x"})
    assert {k: v.status for k, v in c.items() if k in ("python", "data folder", "pandoc", "pypdf", "graphify",
                                                          "local LLM", "API keys")} == {
        "python": "ok", "data folder": "ok", "pandoc": "ok", "pypdf": "ok", "graphify": "ok", "local LLM": "ok",
        "API keys": "ok"}
    assert c["API keys"].detail == "anthropic"


def test_missing_pieces_are_warnings_with_the_fix_in_the_message():
    c = doctor(which=lambda n: None, has_module=lambda m: False)
    assert c["pandoc"].status == "warn" and "pandoc.org" in c["pandoc"].detail
    assert c["graphify"].status == "warn" and "pal-code[full]" in c["graphify"].detail
    assert c["chromadb"].status == "info" and c["API keys"].status == "info"
    assert not [x for x in c.values() if x.status == "fail"]


def test_ollama_down_or_model_missing():
    def down(url):
        raise ConnectionError("refused")
    assert "not reachable" in doctor(http_get=down)["local LLM"].detail
    assert "ollama pull llama3" in doctor(http_get=lambda u: {"models": []})["local LLM"].detail


def test_unwritable_data_folder_fails(tmp_path, monkeypatch):
    f = tmp_path / "afile"
    f.write_text("x", encoding="utf-8")
    monkeypatch.setenv("PAL_HOME", str(f / "sub"))
    assert doctor()["data folder"].status == "fail"


def test_cli_doctor_exit_codes_and_output(capsys):
    assert cli.main(["doctor"]) in (0, 1)
    assert "python" in capsys.readouterr().out


def test_cli_config_prints_yaml_and_bad_config_exits_2(capsys, tmp_path):
    assert cli.main(["config"]) == 0 and "server:" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text("server:\n  port: nope\n", encoding="utf-8")
    assert cli.main(["config", "--config", str(bad)]) == 2
    assert "config error" in capsys.readouterr().err
    assert cli.main(["doctor", "--config", str(bad)]) == 1


def test_serve_flags_override_settings_and_warn_on_public_hosts(monkeypatch, capsys):
    started = {}

    class FakeService:
        def __init__(self, settings):
            started["settings"] = settings

        def local_status(self):
            return {"ok": True, "detail": "llama3 ready"}

        def close(self):
            started["closed"] = True

    monkeypatch.setattr("pal.chat.ChatService", FakeService)
    monkeypatch.setattr("pal.web.create_app", lambda service: "app")
    import waitress
    monkeypatch.setattr(waitress, "serve", lambda app, **kw: started.update(kw, app=app))
    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "8123"]) == 0
    assert started["host"] == "0.0.0.0" and started["port"] == 8123 and started["closed"] and started["app"] == "app"
    assert "no login" in capsys.readouterr().err
    assert started["settings"].server.port == 8123


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and "pal-code" in capsys.readouterr().out


# -- .env loading ---------------------------------------------------------------------------------------
def test_dotenv_loads_keys_without_overriding_and_ignores_junk(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# comment\n\nGEMINI_API_KEY="abc123"\nexport OPENAI_API_KEY=sk-x # trailing\nEXISTING=new\nBAD LINE\nEMPTY=\n'
                 "QUOTED='single'\n", encoding="utf-8-sig")
    env = {"EXISTING": "keep"}
    loaded = cli.load_dotenv(f, env)
    assert env == {"EXISTING": "keep", "GEMINI_API_KEY": "abc123", "OPENAI_API_KEY": "sk-x", "QUOTED": "single"}
    assert loaded == ["GEMINI_API_KEY", "OPENAI_API_KEY", "QUOTED"]


def test_dotenv_missing_file_is_fine(tmp_path):
    assert cli.load_dotenv(tmp_path / "nope.env", {}) == []


def test_serve_reads_dotenv_prints_no_secret_and_warns_when_local_model_is_down(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=AQ.secret-value-123\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class FakeService:
        def __init__(self, settings):
            pass

        def local_status(self):
            return {"ok": False, "detail": "Ollama unreachable: ConnectionError"}

        def close(self):
            pass

    monkeypatch.setattr("pal.chat.ChatService", FakeService)
    monkeypatch.setattr("pal.web.create_app", lambda service: "app")
    import waitress
    monkeypatch.setattr(waitress, "serve", lambda app, **kw: None)
    opened = []
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    assert cli.main(["serve", "--open"]) == 0
    out = capsys.readouterr()
    assert "API key found for: gemini" in out.out and "secret-value" not in out.out + out.err
    assert "WARNING: local model" in out.err
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


# -- LiteLLM and fallback checks ---------------------------------------------------------------------------------------------------------
def test_doctor_checks_the_litellm_pin():
    from pal.llm_wrapper.components.litellm_wrapper import PINNED
    assert doctor(version_of=lambda n: PINNED)["litellm"].status == "ok"
    old = doctor(version_of=lambda n: "1.82.7")["litellm"]
    assert old.status == "warn" and PINNED in old.detail and "1.82.7" in old.detail and "litellm.lock" in old.detail
    missing = doctor(version_of=lambda n: None)["litellm"]
    assert missing.status == "warn" and "require-hashes" in missing.detail


def test_doctor_explains_the_fallback_state():
    def check(env=None, **fb):
        settings = load_settings(env={}, overrides={"fallback": fb} if fb else None)
        return by_name(cli.run_doctor(settings, env=env or {}, http_get=lambda u: {"models": []}))["fallback"]
    assert check().status == "info" and "off" in check().detail
    assert check(enabled=True).status == "warn" and "fallback.model is empty" in check(enabled=True).detail
    no_key = check(enabled=True, model="x:free")
    assert no_key.status == "warn" and "OPENROUTER_API_KEY" in no_key.detail
    ready = check(env={"OPENROUTER_API_KEY": "k"}, enabled=True, model="x:free")
    assert ready.status == "ok" and "openrouter/x:free" in ready.detail


# -- terminal chat -----------------------------------------------------------------------------------------------------------------------
def run_chat(tmp_path, monkeypatch, lines, argv=("chat", "--provider", "anthropic", "--model", "m"), api=None, **svc_kw):
    from tests.fakes import RouterLocal
    from tests.helpers import make_service
    svc_kw.setdefault("local", RouterLocal(route='{"route": "api", "reason": "needs the big model"}'))
    svc, fake_api = make_service(tmp_path, api=api, **svc_kw)
    monkeypatch.setattr("pal.chat.ChatService", lambda settings: svc)
    feed = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None
    monkeypatch.setattr("builtins.input", fake_input)
    return cli.main(list(argv)), svc, fake_api


def test_terminal_chat_answers_prints_a_savings_note_and_leaves_on_quit(tmp_path, monkeypatch, capsys):
    code, svc, api = run_chat(tmp_path, monkeypatch, ["please explain the caching layer", "/quit"])
    out = capsys.readouterr()
    assert code == 0 and "api reply" in out.out and "in /" in out.err and "api" in out.err
    assert len(api.seen) == 1 and svc.metrics.chat_summary()["turns"] == 1


def test_terminal_chat_commands_attach_files_switch_agents_and_reset(tmp_path, monkeypatch, capsys):
    notes = tmp_path / "notes.md"
    notes.write_text("Papertrail searches notes by meaning.", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY_CODING", "agent-key")
    code, svc, api = run_chat(tmp_path, monkeypatch, [
        "/help", f"/file {notes}", "what does Papertrail do", "/agent coding", "explain the cache", "/agent none", "/agent chef",
        "/stats", "/reset", "/nonsense", "/file /nope/missing.md", "", "/exit"], scheduler={"api_min_interval": {"gemini": 0.0}})
    out = capsys.readouterr().out
    assert code == 0 and "/file PATH" in out and "attached notes.md" in out and "now talking to: coding" in out
    assert "now talking to: plain chat" in out and "unknown agent 'chef'" in out and "turns 2" in out
    assert "new conversation" in out and "unknown command" in out and "error:" in out
    assert "Papertrail searches notes by meaning." in api.seen[0][-1]["content"]


def test_terminal_chat_shows_errors_without_dying_and_never_prints_the_key(tmp_path, monkeypatch, capsys):
    class Boom(FakeAPI):
        def chat(self, syscall):
            raise RuntimeError("401 bad key sk-secret-key-12345")
    from tests.fakes import FakeAPI as _F  # noqa: F401
    code, _, _ = run_chat(tmp_path, monkeypatch, ["explain the cache", "/quit"], api=Boom(), scheduler={"api_retry_429": 0})
    out = capsys.readouterr().out
    assert code == 0 and "error:" in out and "sk-secret-key-12345" not in out


def test_terminal_chat_needs_a_provider_and_a_model_and_says_so(tmp_path, monkeypatch, capsys):
    for var in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    code, _, _ = run_chat(tmp_path, monkeypatch, ["/quit"], argv=("chat",))
    assert code == 1 and "no API key found" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    code, _, _ = run_chat(tmp_path / "b", monkeypatch, ["/quit"], argv=("chat",))
    assert code == 1 and "choose a model" in capsys.readouterr().err
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    code, _, _ = run_chat(tmp_path / "c", monkeypatch, ["/quit"], argv=("chat", "--provider", "gemini"))       # gemini has a default model
    assert code == 0


# -- service tools -----------------------------------------------------------------------------------------------------------------------------
def test_doctor_reports_the_service_tools_off_unknown_and_missing_keys():
    def check(env=None, **tools):
        s = load_settings(env={}, overrides={"tools": tools} if tools else None)
        return by_name(cli.run_doctor(s, env=env or {}, http_get=lambda u: {"models": []}))["service tools"]
    off = check()
    assert off.status == "info" and "off" in off.detail and "tools.services" in off.detail
    bad = check(services=["gogle"])
    assert bad.status == "fail" and "unknown service" in bad.detail and "google_search" in bad.detail
    some = check(services=["moon_phase", "currency_convert", "google_search", "text_to_image"])
    assert some.status == "warn" and "2 of 4 ready (moon_phase, currency_convert)" in some.detail
    assert "SERPAPI_API_KEY for google_search" in some.detail and "HF_TOKEN for text_to_image" in some.detail
    full = check(env={"SERPAPI_API_KEY": "k", "RAPIDAPI_KEY": "k"}, services=["google_search", "words_api", "moon_phase"])
    assert full.status == "ok" and "3 of 3 ready" in full.detail and "never checked" not in full.detail
    assert check(env={"x": "y"}, services=["all"]).status == "warn"
    for gone in ("bing_search", "google_places", "imdb_rank", "shazam", "tripadvisor"):        # removed tools are unknown names now
        assert check(services=[gone]).status == "fail"


# -- the two modes and the model browser -----------------------------------------------------------------------------------------------------------
def models_payload():
    return {"data": [
        {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o", "context_length": 128000, "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
        {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B", "context_length": 131072, "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "mistralai/mistral-7b", "name": "Mistral 7B", "pricing": {}}]}


@pytest.fixture
def fresh_catalogue():
    from pal.llm_wrapper.components import openrouter_wrapper as orw
    orw._cache.update(at=0.0, models=[])
    yield
    orw._cache.update(at=0.0, models=[])


def models_args(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**{"search": None, "free": False, "limit": 30, "json": False, **kw})


def test_models_lists_what_openrouter_can_reach_with_context_price_and_the_name_to_pass(fresh_catalogue, capsys):
    assert cli._models(models_args(), fetch=lambda url: models_payload()) == 0
    out = capsys.readouterr().out
    assert "MODEL (pass as --model)" in out and "openai/gpt-4o" in out and "128,000" in out and "2.5 / 10" in out
    assert "meta-llama/llama-3.3-70b-instruct:free" in out and "free" in out and "3 model(s)" in out and "pal chat --mode explore --model <MODEL>" in out
    assert "mistralai/mistral-7b" in out and out.count("?") >= 2                                                           # unknown context and price are shown as ?


def test_models_can_be_searched_narrowed_to_free_limited_and_printed_as_json(fresh_catalogue, capsys):
    assert cli._models(models_args(search="llama"), fetch=lambda url: models_payload()) == 0
    assert "1 model(s)" in capsys.readouterr().out
    assert cli._models(models_args(free=True), fetch=lambda url: models_payload()) == 0
    assert "meta-llama" in capsys.readouterr().out
    assert cli._models(models_args(limit=1), fetch=lambda url: models_payload()) == 0
    assert "1 model(s)" in capsys.readouterr().out
    import json
    assert cli._models(models_args(json=True, free=True), fetch=lambda url: models_payload()) == 0
    data = json.loads(capsys.readouterr().out)
    assert [m["id"] for m in data] == ["meta-llama/llama-3.3-70b-instruct:free"] and data[0]["free"] is True
    assert cli._models(models_args(search="zzz"), fetch=lambda url: models_payload()) == 0
    assert "No models match." in capsys.readouterr().out


def test_models_says_so_when_the_list_cannot_be_read_and_exits_with_an_error(fresh_catalogue, capsys):
    def down(url):
        raise OSError("no network")
    assert cli._models(models_args(), fetch=down) == 1
    err = capsys.readouterr().err
    assert "could not read OpenRouter's model list" in err and "OSError" in err and "type a model name yourself" in err


def test_the_command_line_accepts_the_models_command_and_the_mode_option():
    p = cli.build_parser()
    a = p.parse_args(["models", "--free", "--search", "llama", "--limit", "5", "--json"])
    assert (a.cmd, a.free, a.search, a.limit, a.json) == ("models", True, "llama", 5, True)
    assert p.parse_args(["chat", "--mode", "explore", "--model", "x"]).mode == "explore" and p.parse_args(["chat"]).mode is None
    with pytest.raises(SystemExit):
        p.parse_args(["chat", "--mode", "cheap"])


@pytest.mark.parametrize("env,mode,provider,expect", [
    ({"OPENROUTER_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, "explore", None, "openrouter"),
    ({"OPENROUTER_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, "own_key", None, "anthropic"),
    ({"OPENROUTER_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, None, None, "anthropic"),          # no mode: the first provider with a key, as before
    ({"ANTHROPIC_API_KEY": "k"}, "own_key", "anthropic", "anthropic")])
def test_a_mode_limits_which_provider_the_terminal_chat_picks(env, mode, provider, expect):
    assert cli._pick_provider(load_settings(env={}), provider, "some-model", env, mode) == (expect, "some-model")


def test_a_mode_without_its_key_or_with_the_wrong_provider_explains_what_to_do():
    from pal.errors import ConfigError
    s = load_settings(env={})
    with pytest.raises(ConfigError, match="Explore mode needs an OpenRouter key: set OPENROUTER_API_KEY"):
        cli._pick_provider(s, None, "m", {"ANTHROPIC_API_KEY": "k"}, "explore")
    with pytest.raises(ConfigError, match="own-key mode needs a key for one provider"):
        cli._pick_provider(s, None, "m", {"OPENROUTER_API_KEY": "k"}, "own_key")
    with pytest.raises(ConfigError, match="openai is not an explore models provider"):
        cli._pick_provider(s, "openai", "m", {"OPENAI_API_KEY": "k"}, "explore")
    with pytest.raises(ConfigError, match="choose a model: --model <name> \\(for openrouter\\)"):
        cli._pick_provider(s, None, None, {"OPENROUTER_API_KEY": "k"}, "explore")


def test_doctor_reports_whether_each_mode_is_ready():
    c = doctor(which=lambda n: "/usr/bin/" + n, has_module=lambda m: True, env={"OPENROUTER_API_KEY": "k"})
    assert c["mode: explore models"].status == "ok" and "openrouter" in c["mode: explore models"].detail
    assert c["mode: use my own key"].status == "info" and "not set up: set ANTHROPIC_API_KEY or OPENAI_API_KEY ..." in c["mode: use my own key"].detail
    both = doctor(which=lambda n: "/usr/bin/" + n, has_module=lambda m: True, env={"GEMINI_API_KEY": "k"})
    assert both["mode: use my own key"].status == "ok" and "gemini" in both["mode: use my own key"].detail and both["mode: explore models"].status == "info"


# -- opening the Studio -------------------------------------------------------------------------------------------------------------------------------
def test_serve_can_open_the_studio_page_and_the_chat_stays_the_default():
    p = cli.build_parser()
    assert p.parse_args(["serve", "--open", "--page", "studio"]).page == "studio" and p.parse_args(["serve"]).page == ""
    with pytest.raises(SystemExit):
        p.parse_args(["serve", "--page", "admin"])


def test_the_launcher_has_a_studio_command_and_double_click_files_that_use_it(monkeypatch):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_launcher", root / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    calls = []
    monkeypatch.setattr(run, "ensure_venv", lambda: Path("py"))
    monkeypatch.setattr(run, "ensure_installed", lambda py: None)
    monkeypatch.setattr(run, "_run", lambda cmd: calls.append([str(c) for c in cmd]) or 0)
    monkeypatch.setattr(run.sys, "prefix", run.sys.base_prefix)
    assert run.main(["studio"]) == 0 and calls[-1][1:] == ["-m", "pal", "serve", "--open", "--page", "studio"]
    assert run.main([]) == 0 and calls[-1][1:] == ["-m", "pal", "serve", "--open"]
    assert "python run.py studio" in (root / "Studio.bat").read_text() and "python3 run.py studio" in (root / "Studio.command").read_text()
    assert (root / "Studio.bat").read_text().splitlines()[0] == "@echo off"
