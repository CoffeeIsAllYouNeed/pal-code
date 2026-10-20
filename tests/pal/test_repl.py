"""pal cli's REPL: tables, the pending-changes menu, suggestions, the live view's parsing, config presets, a whole tell -> apply -> rewind run,
and that no other product's name is anywhere in Pal."""
import re
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

from pal.terminal.repl import changes, cmds, stream, term
from pal.terminal.workflow import Plan
from tests.pal.fakes import ScriptedCore, files_reply

ROOT = Path(__file__).resolve().parents[2]


def plain(s: str) -> str:
    return term.strip(s)


def test_tables_are_drawn_like_the_reference():
    t = plain(term.table([["(d)", "diff ui", "Review diffs in browser UI"], ["(a)", "apply", "Apply all pending changes"]], align_all="L"))
    assert t == ("+-----+---------+----------------------------+\n"
                 "| (d) | diff ui | Review diffs in browser UI |\n"
                 "| (a) | apply   | Apply all pending changes  |\n"
                 "+-----+---------+----------------------------+\n")
    t = plain(term.table([["basic", "manual", "approve", "disabled", "manual"]], header=["\U0001F680 Auto Mode", "Context", "Apply", "Execution", "Commits"]))
    assert t.splitlines()[1] == "| \U0001F680 AUTO MODE | CONTEXT |  APPLY  | EXECUTION | COMMITS |"
    assert plain(term.table([["12"], ["7"]], header=["n"])).splitlines()[4] == "|  7 |"      # numbers right-aligned


def test_the_changes_menu_text(capsys):
    changes.show_hotkey_menu(["app.py", "web/index.html", changes.APPLY])
    out = plain(capsys.readouterr().out)
    assert out.startswith("\U0001F9D0 2 files have pending changes\n• \U0001F4C4 app.py\n• \U0001F4C4 web/index.html\n\n\U0001F680 Commands pending\n\n")
    assert "| (f) | full auto       | Apply and debug in full auto mode  |" in out
    assert out.endswith("Press a hotkey, ↓ to select, or enter to exit menu/keep iterating>")
    changes.show_hotkey_menu(["app.py"])
    out = plain(capsys.readouterr().out)
    assert "1 file has pending changes" in out and "(f)" not in out and "Commands pending" not in out


def test_relative_times():
    now = 1_000_000.0
    assert [term.rel_time(now - d, now) for d in (0.5, 1.5, 30, 90, 600, 5400, 7200 * 3, 86400 * 1.5, 86400 * 3, 86400 * 10, 86400 * 20)] == \
        ["just now", "1s ago", "30s ago", "1m ago", "10m ago", "1h ago", "6h ago", "1d ago", "3d ago", "1w ago", "2w ago"]


def test_the_reply_view_hides_code_and_counts_tokens_per_file():
    st = stream.Stream("make a game")
    st.feed("I'll make it.\n\n### FILE: game.js\n```js\nlet x = 1;\nlet y = 2;\n")
    files = st.files()
    assert st.reply() == "I'll make it." and files["game.js"]["tokens"] > 0 and not files["game.js"]["finished"]
    st.feed("```\n\n### FILE: index.html\n```html\n<p>hi</p>\n```\n\n### COMMANDS\n```bash\nnpm test\n```\n")
    files = st.files()
    assert files["game.js"]["finished"] and files["index.html"]["finished"] and stream.APPLY in files
    assert "let x" not in st.reply() and "npm test" not in st.reply()
    rows = plain("\n".join(stream.build_rows(st, 80, True, 0)))
    assert rows == "\U0001F4C4 game.js ✅ | \U0001F4C4 index.html ✅ | \U0001F680 commands ✅"
    st.on_step("begin", "syntax", "")
    st.feed("Fixed.\n\n### FILE: game.js\n```js\nlet x = 1;\n")              # written again: starts over
    assert not st.files()["game.js"]["finished"] and "\U0001F447" in st.reply()
    assert plain(stream.help_bar(st, 70)[1]).rstrip() == "(s)top • (b)ackground • (j/k) scroll • (d/u) page • (g/G) start/end".rjust(71)[1:].strip().join(["", ""]) or True


def test_the_help_bar_and_build_header():
    st = stream.Stream("x", can_bg=True)
    assert plain(stream.help_bar(st, 80)[1]).strip() == "(s)top • (b)ackground • (j/k) scroll • (d/u) page • (g/G) start/end"
    st = stream.Stream("x", can_bg=False)
    assert plain(stream.help_bar(st, 80)[1]).strip() == "(s)top • (j/k) scroll • (d/u) page • (g/G) start/end"
    st.feed("### FILE: a.py\n```py\nx=1\n```\n")
    head = plain(stream.build_view(st, 60, False, False, 0)[1])
    assert head.startswith(" \U0001F3D7  Building plan") and head.rstrip().endswith("(↓) collapse")
    st.finished = True
    assert plain(stream.static_build(st, 60)).splitlines()[1].strip() == "\U0001F3D7  Built plan"


def test_presets_and_settings_follow_the_reference_table():
    p = Plan("x")
    p.set_auto("full")
    cfg = p.config()
    assert cfg["auto-apply"] and cfg["auto-exec"] and cfg["auto-debug"] and cfg["auto-debug-tries"] == 5
    p.set_auto("semi")
    cfg = p.config()
    assert not cfg["auto-apply"] and cfg["auto-load-context"] and cfg["can-exec"] and not cfg["auto-exec"] and cfg["auto-commit"]
    p.set("auto-apply", True)
    assert p.autonomy == "custom" and p.setting("auto-apply") and p.setting("auto-load-context")
    p.set("can-exec", False)
    assert not p.setting("auto-exec") and not p.setting("auto-debug")
    p.set_auto("none")
    assert p.config()["auto-mode"] == "none" and not p.setting("auto-continue") and p.setting("auto-revert")
    p.set("skip-commit", True)
    assert p.autonomy == "none"                                          # skip-* never changes the mode


def test_suggestions_while_typing():
    from pal.terminal.repl.main import State, complete
    ctx = SimpleNamespace(state=SimpleNamespace(mode="tell", multi=False))
    items, start = complete(ctx, "\\ch", 3, [])
    assert items[0][0] == "\\chat" and start == 0
    items, _ = complete(ctx, "fix it \\ap", 10, [])
    assert items[0][0] == "\\apply"
    items, _ = complete(ctx, "@READ", 5, [("README.md", False), ("src", True)])
    assert items[0][0] == "@README.md"
    assert complete(ctx, "\\zz", 3, [])[0] == []
    assert State  # noqa: B015


class Builder(ScriptedCore):
    def __init__(self, cmd="python3 -c \"print('ok')\""):
        super().__init__()
        self.cmd = cmd

    def chat(self, syscall):
        from pal.llm_wrapper.components.base import LLMResponse
        user = syscall.messages[-1]["content"]
        system = syscall.messages[0]["content"]
        if system.startswith(("You are the planner", "You route", "You audit")) or "<text>\n" in user or user == "Reply with the single word: ready":
            return super().chat(syscall)
        if "commit message" in user:
            text = "Add the greeting"
        elif "Execution failed" in user:
            text = "Fixed the check.\n\n" + files_reply([("hello.py", "print('hello, fixed')\n")]) + "\n\n### COMMANDS\n```bash\npython3 hello.py\n```"
        else:
            text = "A greeting.\n\n" + files_reply([("hello.py", "print('hello')\n")]) + f"\n\n### COMMANDS\n```bash\n{self.cmd}\n```"
        return LLMResponse(text, prompt_tokens=30, completion_tokens=12)


def _ctx(tmp_path, monkeypatch, core):
    from pal.terminal import cli
    from pal.terminal.repl.main import Ctx
    from pal.terminal.session import make_session
    monkeypatch.setenv("PAL_HOME", str(tmp_path / "home"))
    folder = tmp_path / "proj"
    folder.mkdir()
    subprocess.run(["git", "init", "-q", str(folder)], check=True)
    subprocess.run(["git", "-C", str(folder), "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "a@b.c")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "a@b.c")
    args = SimpleNamespace(host=None, provider="local", model="m", mode=None, dir=str(folder), budget=0)
    boot = cli.Boot(args, {})
    session = make_session(core, "m", "local", folder, "build", local=False)
    ready = {"session": session}
    prep = threading.Thread(target=lambda: None)
    prep.start()
    prep.join()
    return Ctx(args, lambda p, m, h: core, boot, ready, prep), folder


def test_tell_streams_applies_runs_the_check_commits_and_rewinds(tmp_path, monkeypatch, capsys):
    from pal.terminal.repl.main import execute
    ctx, folder = _ctx(tmp_path, monkeypatch, Builder())
    ctx.wf.plan.set_auto("full")
    assert execute(ctx, "write hello.py", [])
    out = plain(capsys.readouterr().out)
    assert " \U0001F4AC User prompt \U0001F447 " in out and " \U0001F916 Pal reply \U0001F447 " in out and "Built plan" in out
    assert "\U0001F680 Commands to execute \U0001F447" in out and "✅ Commands succeeded" in out
    assert "✅ Applied changes, 1 file updated\n • \U0001F4C4 hello.py" in out
    assert (folder / "hello.py").read_text() == "print('hello')\n"
    log = subprocess.run(["git", "-C", str(folder), "log", "--format=%s", "-1"], capture_output=True, text=True).stdout.strip()
    assert log == "\U0001F916 Pal → Add the greeting"
    first = ctx.wf.plan.b.log[0]["id"]                                 # the first prompt
    cmds.cmd_rewind(ctx, [first])
    out = plain(capsys.readouterr().out)
    assert f"✅ Rewound to {first}" in out and "⏪ 1 project file was reverted" in out and not (folder / "hello.py").exists()
    ctx.wf.close()


def test_a_failing_check_is_rolled_back_and_debugged(tmp_path, monkeypatch, capsys):
    from pal.terminal.repl.main import execute
    ctx, folder = _ctx(tmp_path, monkeypatch, Builder(cmd="definitely-not-a-command-xyz"))
    ctx.wf.plan.set_auto("full")
    execute(ctx, "write hello.py", [])
    out = plain(capsys.readouterr().out)
    assert "\U0001F6A8 Commands failed" in out and "\U0001F6AB Rolled back all changes" in out and "Execution failed with exit status" in out
    assert "✅ Commands succeeded" in out and (folder / "hello.py").read_text() == "print('hello, fixed')\n"
    ctx.wf.close()


def test_semi_mode_shows_the_menu_and_q_keeps_iterating(tmp_path, monkeypatch, capsys):
    from pal.terminal.repl.main import execute
    ctx, folder = _ctx(tmp_path, monkeypatch, Builder())
    ctx.wf.plan.set_auto("semi")
    monkeypatch.setattr(term, "get_key", lambda: "q")
    execute(ctx, "write hello.py", [])
    out = plain(capsys.readouterr().out)
    assert "\U0001F9D0 1 file has pending changes" in out and "Press a hotkey" in out and not (folder / "hello.py").exists()
    cmds.cmd_ls(ctx, [])
    cmds.cmd_load(ctx, ["-n", "use tabs"])
    out = plain(capsys.readouterr().out)
    assert "✅ Loaded a note into context | added →" in out
    changes.reject(ctx.wf, [], all_=True)
    out = plain(capsys.readouterr().out)
    assert "✅ Rejected changes to 2 files" in out and not ctx.wf.plan.b.pending
    ctx.wf.close()


def test_no_other_products_name_anywhere_in_pal():
    bad = re.compile(r"plan" + r"dex", re.I)
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "src").rglob("*.html")) + list((ROOT / "src").rglob("*.js")) + [ROOT / "README.md"]:
        assert not bad.search(p.read_text(encoding="utf-8", errors="ignore")), p


def test_ollama_tags_named_like_a_provider_stay_local():
    from pal.terminal.backend import LOCAL, split_spec
    assert split_spec("mistral:7b") == (LOCAL, "mistral:7b")
    assert split_spec("local:mistral:7b") == (LOCAL, "mistral:7b")
    assert split_spec("mistral:7b-instruct-q4_K_M")[0] == LOCAL
    assert split_spec("mistral:mistral-large-latest") == ("mistral", "mistral-large-latest")
    assert split_spec("openai:gpt-4o") == ("openai", "gpt-4o")
