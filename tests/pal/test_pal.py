"""The Pal terminal session: its logic (no screen), the safety rules for files it proposes, and the `pal ask` command."""
import re
from pathlib import Path

import pytest

from pal.llm_wrapper.components.base import LLMResponse
from pal.terminal import backend
from pal.terminal.cli import build_parser, main
from pal.terminal.clients import Reply
from pal.terminal.policy import PolicyError
from pal.terminal.session import PalSession, check_path
from pal.terminal.theme import wordmark
from tests.pal.fakes import ScriptedCore, files_reply


class FakeClient:
    """Answers with `text` and remembers every message and the references it was sent."""

    def __init__(self, text="ok"):
        self.text, self.calls = text, []

    def ask(self, message, references=None):
        self.calls.append((message, references))
        return Reply(self.text, prompt_tokens=len(message) // 4, completion_tokens=len(self.text) // 4, helper_tokens=5, seconds=0.1, calls=1)


def session(tmp_path, text="ok", mode="build"):
    return PalSession(FakeClient(text), "m", "local", tmp_path, mode), None


@pytest.mark.parametrize("path,ok", [("src/app.py", True), ("main.go", True), (".gitignore", True), ("a/b/c.json", True), ("../x.py", False), ("/etc/x.py", False), ("C:/x.py", False),
                                     ("run.sh", False), ("setup.bat", False), ("tool.exe", False), (".env", False), ("noext", False), ("con.py", False)])
def test_paths_pal_may_write(path, ok):
    if ok:
        assert check_path(path)
    else:
        with pytest.raises(PolicyError):
            check_path(path)


def test_a_turn_counts_tokens_and_a_plain_chat_estimate(tmp_path):
    s, _ = session(tmp_path, "hello there")
    t = s.ask("hi pal")
    assert s.client.calls[0][0].startswith("hi pal") and t.text == "hello there" and t.changes == []
    first = s.plain_tokens
    s.ask("and again")
    assert s.turns == 2 and s.pal_tokens > 0 and s.helper_tokens == 10 and s.plain_tokens > first * 1.5      # a plain chat carries the whole history along


def test_an_at_file_goes_to_the_model_once_and_again_only_when_it_changes(tmp_path):
    (tmp_path / "notes.md").write_text("# one", encoding="utf-8")
    s, _ = session(tmp_path, mode="plan")
    s.ask("summarise @notes.md")
    s.ask("shorter please @notes.md")
    (tmp_path / "notes.md").write_text("# two", encoding="utf-8")
    s.ask("now @notes.md")
    refs = [c[1] for c in s.client.calls]
    assert [r[0]["name"] for r in (refs[0], refs[2])] == ["notes.md", "notes.md"] and refs[1] is None


def test_at_files_outside_the_folder_or_missing_or_binary_are_left_out_with_a_warning(tmp_path):
    (tmp_path.parent / "secret.txt").write_text("nope", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
    s, _ = session(tmp_path)
    t = s.ask("read @../secret.txt and @missing.py and @blob.bin")
    assert s.client.calls[0][1] is None and len(t.warnings) == 3


def test_build_mode_proposes_changes_with_a_diff_and_writes_nothing_until_applied(tmp_path):
    (tmp_path / "app.py").write_text("print('old')\n", encoding="utf-8")
    reply = files_reply([("app.py", "print('new')\n"), ("util/helpers.py", "X = 1\n"), ("run.sh", "rm -rf /"), ("../evil.py", "x")], "done")
    s, _ = session(tmp_path, reply, mode="build")
    t = s.ask("make it new")
    assert "Files in the working folder" in s.client.calls[0][0] and "app.py" in s.client.calls[0][0]
    assert [(c.path, c.is_new) for c in t.changes] == [("app.py", False), ("util/helpers.py", True)] and len(t.refused) == 2
    assert ["add", "print('new')"] in t.changes[0].diff and (tmp_path / "app.py").read_text() == "print('old')\n" and not (tmp_path / "util").exists()
    assert "```" not in t.text and not t.text.startswith("NOTE")
    done = dict(s.apply(t.changes))
    assert done == {"app.py": "written", "util/helpers.py": "written"} and (tmp_path / "util" / "helpers.py").read_text() == "X = 1\n"
    assert not (tmp_path.parent / "evil.py").exists()


def test_an_unchanged_file_is_not_proposed_and_the_plan_agent_never_proposes(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert session(tmp_path, files_reply([("a.py", "x = 1\n")]), "build")[0].ask("same").changes == []
    assert session(tmp_path, files_reply([("a.py", "x = 2\n")]), "plan")[0].ask("change").changes == []


def test_the_listing_skips_hidden_and_dependency_folders(tmp_path):
    for p in ("a.py", "src/b.py", ".git/config", "node_modules/x/y.js", "__pycache__/z.pyc"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text("x", encoding="utf-8")
    assert [x.split(" ")[0] for x in session(tmp_path)[0].listing()] == ["a.py", "src/b.py"]


def test_tab_cycles_through_the_build_plan_and_ask_agents(tmp_path):
    s, _ = session(tmp_path)
    assert (s.mode, s.toggle_mode(), s.toggle_mode(), s.toggle_mode()) == ("build", "plan", "ask", "build")
    assert PalSession(FakeClient(), "m", "local", tmp_path, "chat").mode == "build"       # an old saved mode falls back to Build


def test_the_plan_agent_asks_for_a_plan_with_the_file_list_and_never_for_files(tmp_path):
    (tmp_path / "app.py").write_text("x", encoding="utf-8")
    s, _ = session(tmp_path, "1. do it", mode="plan")
    s.ask("add logging")
    sent = s.client.calls[0][0]
    assert "PLAN mode" in sent and "app.py" in sent and "### FILE" not in sent and "COMPLETE new content" not in sent


def test_the_last_model_is_remembered_only_if_it_still_exists_else_the_strongest_local_one(monkeypatch):
    info = {"ok": True, "recommended": "big:70b", "models": [{"name": "big:70b", "embedding": False}, {"name": "small:1b", "embedding": False}, {"name": "nomic-embed", "embedding": True}]}
    assert backend.pick_default(info, {"provider": "local", "model": "small:1b"}) == ("local", "small:1b")
    assert backend.pick_default(info, {"provider": "local", "model": "gone:7b"}) == ("local", "big:70b")
    assert backend.pick_default({"ok": False}, {}) == ("local", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert backend.pick_default({"ok": False}, {"provider": "openrouter", "model": "x/y"}) == ("openrouter", "x/y")
    assert backend.chat_models(info) == ["big:70b", "small:1b"]


def test_preferences_are_saved_without_any_key(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    backend.save_prefs(provider="local", model="m", mode="build", api_key="SECRET")
    saved = backend.prefs_path().read_text()
    assert backend.load_prefs()["mode"] == "build" and "SECRET" not in saved and "api_key" not in saved


def test_a_missing_api_key_says_what_to_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from pal.errors import UserInputError
    with pytest.raises(UserInputError, match="OPENROUTER_API_KEY"):
        backend.make_core("openrouter", "x/y")


def test_the_wordmark_is_five_lines_of_blocks_with_a_red_dot():
    lines = wordmark()
    assert len(lines) == 5 and lines[0][-1][1] == "#E50914" and all(set("".join(c for c, _ in line)) <= set("\u2588\u2580\u2584 ") for line in lines)


class _Core(ScriptedCore):
    def chat(self, syscall):
        system, user = syscall.messages[0]["content"], syscall.messages[-1]["content"]
        if system.startswith(("You are the planner", "You route", "You audit")) or re.search(r"<text>\n", user) or user == "Reply with the single word: ready":
            return super().chat(syscall)
        text = "Because it saves tokens."
        return LLMResponse(text=text, prompt_tokens=50, completion_tokens=6, raw=None, meta={})


def test_pal_ask_prints_the_answer_and_the_token_count(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path / "home"))
    code = main(["--dir", str(tmp_path), "--model", "m", "ask", "why", "pal?"], core_factory=lambda provider, model, host: _Core())
    out = capsys.readouterr()
    assert code == 0 and "Because it saves tokens." in out.out and "tokens" in out.err


def test_pal_ask_without_any_model_explains_what_to_do(capsys, monkeypatch, tmp_path):
    from pal.terminal.hardware import from_spec
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(backend, "_HW", from_spec("gpu:8"))
    monkeypatch.setattr(backend, "local_info", lambda host=None: {"ok": False, "models": [], "recommended": None})
    assert main(["ask", "hi"]) == 1
    assert "ollama pull" in capsys.readouterr().out


def test_the_command_line_has_the_documented_options():
    a = build_parser().parse_args(["--provider", "openrouter", "--model", "x/y", "--mode", "ask", "ask", "hi", "there", "--plan"])
    assert (a.provider, a.model, a.mode, a.cmd, a.question, a.plan, a.build) == ("openrouter", "x/y", "ask", "ask", ["hi", "there"], True, False)
    assert build_parser().parse_args([]).cmd is None


def test_the_installed_command_is_called_pal():
    text = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^pal = "pal\.terminal\.cli:main"', text, re.M) and 'name = "pal-code"' in text


class SavingClient(FakeClient):
    """Reports engine savings like the real client does."""

    def ask(self, message, references=None):
        r = super().ask(message, references)
        r.saved_by = {"history": 40, "input": 10}
        return r


def test_each_request_is_recorded_with_where_it_saved_and_unchanged_files_count_as_not_resent(tmp_path):
    (tmp_path / "big.md").write_text("word " * 400, encoding="utf-8")
    s = PalSession(SavingClient("ok"), "m", "local", tmp_path, "plan")
    s.ask("read @big.md")
    s.ask("again @big.md")
    first, second = s.records
    assert first.saved_by == {"history": 40, "input": 10} and "files" not in first.saved_by
    assert second.saved_by["files"] >= 400 and second.plain > second.used
    assert s.saved == s.plain_tokens - s.pal_tokens and s.saved_by_total()["history"] == 80


def test_the_token_commands_show_the_session_the_last_request_every_request_and_how_it_is_counted(tmp_path):
    from rich.console import Console

    from pal.terminal import ui
    s = PalSession(SavingClient("ok"), "m", "local", tmp_path, "plan")
    console = Console(record=True, width=120, color_system=None)
    ui.show_tokens(console, s)
    assert "No requests yet" in console.export_text()
    s.ask("first question here")
    s.ask("second one")
    for arg, expect in (("", "This session"), ("last", "Last request"), ("turns", "second one"), ("help", "How Pal counts tokens")):
        ui.show_tokens(console, s, arg)
        assert expect in console.export_text(clear=True), arg
    ui.show_tokens(console, s)
    text = console.export_text()
    assert "Left out old turns" in text and "Trimmed filler" in text and "plain chat (est.)" in text


def test_the_live_line_shows_the_running_saving(tmp_path):
    from pal.terminal.cli import _live
    s = PalSession(SavingClient("ok"), "m", "local", tmp_path, "build")
    assert _live(s, 3).plain.startswith("Pal is building… 3s")
    s.ask("hello " * 50)
    s.ask("more " * 50)
    assert ("saved so far" in _live(s, 1).plain) == (s.saved > 0)


def test_pal_has_a_cli_and_a_web_command_and_options_work_on_either_side():
    p = build_parser()
    a = p.parse_args(["cli", "--model", "x", "--mode", "plan"])
    assert (a.cmd, a.model, a.mode) == ("cli", "x", "plan")
    b = p.parse_args(["--model", "y", "cli"])
    assert (b.cmd, b.model, b.dir) == ("cli", "y", ".")
    w = p.parse_args(["web", "--port", "8080", "--no-open"])
    assert (w.cmd, w.port, w.no_open, w.page) == ("web", 8080, True, "studio")


def test_without_a_gpu_pal_ask_says_to_use_an_api_model(capsys, monkeypatch, tmp_path):
    from pal.terminal.hardware import from_spec
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(backend, "_HW", from_spec("cpu:16"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["ask", "hi"]) == 1
    assert "No GPU on this computer, so Pal uses an API model" in capsys.readouterr().out


def test_pal_web_starts_the_local_server_and_opens_the_studio_in_the_browser(monkeypatch, tmp_path):
    import pal.cli as pal_cli
    seen = {}
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(pal_cli, "_serve", lambda settings, open_browser=False, page="", **kw: seen.update(port=settings.server.port, host=settings.server.host, open=open_browser, page=page) or 0)
    assert main(["web", "--port", "8123"]) == 0
    assert seen == {"port": 8123, "host": "127.0.0.1", "open": True, "page": "studio"}
    assert main(["web", "--page", "chat", "--no-open"]) == 0 and seen["page"] == "" and seen["open"] is False


def test_the_logo_is_drawn_in_its_colours_and_shown_beside_the_wordmark():
    from pal.terminal.logo import GRID, logo_lines, ptk_style, rich_style
    from pal.terminal.ui import brand_block
    lines = logo_lines()
    assert len(lines) == len(GRID) // 2 and all(len(line) == len(GRID[0]) for line in lines)
    styles = {st for line in lines for _, st in line}
    assert any("#E50914" in s for s in styles) and any("#FFFFFF" in s for s in styles)
    assert ptk_style("#FFFFFF|#E50914") == "fg:#FFFFFF bg:#E50914" and rich_style("#FFFFFF|") == "#FFFFFF" and ptk_style("") == ""
    text = "".join(t for _, t in brand_block())
    assert text.count("\n") == len(lines) - 1


def test_the_web_pages_use_the_logo_and_it_is_packaged():
    from pathlib import Path
    web = Path(__file__).parents[2] / "src/pal/web"
    for size in (32, 64, 256):
        assert (web / "static" / f"pal-logo-{size}.png").stat().st_size > 1000
    assert "pal-logo" in (web / "templates/studio.html").read_text() and "pal-logo" in (web / "templates/index.html").read_text()
    assert "web/static/*.png" in (Path(__file__).parents[2] / "pyproject.toml").read_text()


def test_the_ask_agent_answers_with_the_folder_for_context_and_never_proposes_files(tmp_path):
    (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")
    s = PalSession(FakeClient(files_reply([("app.py", "x = 2\n")])), "m", "local", tmp_path, "ask")
    t = s.ask("what does app.py do?")
    sent = s.client.calls[0][0]
    assert "ASK mode" in sent and "app.py" in sent and "PLAN mode" not in sent and "COMPLETE new content" not in sent and t.changes == []


def test_pal_ask_uses_the_ask_agent_unless_told_otherwise(monkeypatch, tmp_path):
    import pal.terminal.cli as cli
    seen = []
    monkeypatch.setattr(cli, "start", lambda args, core_factory=None: seen.append(args.mode))
    for extra, mode in (([], "ask"), (["--plan"], "plan"), (["--build"], "build")):
        cli.main(["ask", "hi", *extra])
        assert seen[-1] == mode
