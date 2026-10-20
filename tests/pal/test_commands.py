"""Claude-Code-style commands, prompt history, voice input and the keyboard."""
import io
import wave
from types import SimpleNamespace

import pytest

from pal.terminal import ui
from pal.terminal.cli import handle
from pal.terminal.history import Browser, History
from pal.terminal.session import make_session
from pal.terminal.workflow import Store, Workflow
from tests.pal.fakes import ScriptedCore, files_reply


class Core(ScriptedCore):
    def __init__(self, reply="an answer"):
        super().__init__()
        self.reply, self.seen = reply, []

    def chat(self, syscall):
        from pal.llm_wrapper.components.base import LLMResponse
        self.seen.append(syscall.messages[-1]["content"])
        text = self.reply(syscall.messages[-1]["content"]) if callable(self.reply) else self.reply
        return LLMResponse(text, prompt_tokens=20, completion_tokens=5)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PAL_HOME", str(tmp_path / "home"))
    folder = tmp_path / "proj"
    folder.mkdir()
    core = Core()
    factory = lambda p, m, h: core  # noqa: E731
    s = make_session(core, "m", "local", folder, "ask", local=False)
    box = {"wf": Workflow(s, Store(folder, tmp_path / "home"))}
    args = SimpleNamespace(host=None)
    yield SimpleNamespace(s=s, box=box, args=args, factory=factory, core=core, folder=folder, console=ui.make_console())
    box["wf"].close()


def run(env, line):
    env.s, go = handle(line, env.args, env.s, env.console, env.factory, env.box)
    return go


def test_every_new_command_is_in_the_menu():
    names = {n for n, _ in ui.all_commands()}
    for c in ("/init", "/clear", "/compact", "/btw", "/cost", "/context", "/memory", "/add-dir", "/permissions", "/config", "/plan", "/build", "/ask",
              "/copy", "/export", "/agents", "/simplify", "/status", "/doctor", "/resume", "/voice", "/shortcuts", "/debug", "/project", "/clear-context"):
        assert c in names, c
    assert len(names) == len(ui.all_commands())                        # no command listed twice


def test_plan_build_ask_switch_agent_and_can_ask_at_once(env):
    run(env, "/plan")
    assert env.s.mode == "plan"
    run(env, "/ask what is this project?")
    assert env.s.mode == "ask" and "what is this project?" in env.core.seen[-1]


def test_clear_starts_a_fresh_conversation_and_keeps_the_plan(env):
    env.box["wf"].tell("first question")
    old = env.s
    plan = env.box["wf"].plan.name
    run(env, "/clear")
    assert env.s is not old and env.box["wf"].plan.name == plan and env.box["wf"].plan.b.convo == []


def test_compact_turns_the_conversation_into_a_summary(env):
    for i in range(3):
        env.box["wf"].tell(f"question {i} " + "detail " * 200)
    env.core.reply = "- decided X\n- built Y"
    run(env, "/compact the API names")
    convo = env.box["wf"].plan.b.convo
    assert len(convo) == 1 and "decided X" in convo[0]["text"] and "the API names" in env.core.seen[-1]


def test_btw_is_answered_but_not_added_to_the_conversation(env):
    env.box["wf"].tell("main thread")
    n = len(env.box["wf"].plan.b.convo)
    env.core.reply = "409 means conflict."
    run(env, "/btw what does HTTP 409 mean?")
    assert len(env.box["wf"].plan.b.convo) == n and "HTTP 409" in env.core.seen[-1]


def test_cost_status_context_agents_config_and_shortcuts_print(env, capsys):
    env.box["wf"].tell("hi")
    for c in ("/cost", "/status", "/context", "/agents", "/config", "/shortcuts", "/permissions"):
        assert run(env, c), c


def test_copy_uses_the_clipboard_tool(env, monkeypatch):
    from pal.terminal import commands
    env.box["wf"].tell("hi")
    got = {}
    monkeypatch.setattr(commands.shutil, "which", lambda n: "/usr/bin/" + n if n in ("pbcopy", "wl-copy", "clip") else None)
    monkeypatch.setattr(commands.subprocess, "run", lambda cmd, input=None, check=True, timeout=5: got.update(cmd=cmd, input=input))
    run(env, "/copy")
    assert got["cmd"][0] in ("pbcopy", "wl-copy", "clip") and got["input"]


def test_export_writes_the_conversation_as_markdown(env):
    env.box["wf"].tell("how do I start?")
    (env.folder / "notes").mkdir()
    run(env, "/export notes/convo.md")
    text = (env.folder / "notes" / "convo.md").read_text()
    assert "how do I start?" in text and "**You**" in text


def test_init_writes_a_pal_md_and_memory_shows_it(env):
    env.core.reply = files_reply([("PAL.md", "# proj\n\n- run with python app.py\n")])
    run(env, "/init")
    assert (env.folder / "PAL.md").read_text().startswith("# proj") and not env.box["wf"].plan.b.pending and env.s.mode == "ask"   # written at once
    (env.folder / "PAL.md").write_text("# proj\n\n- tabs, not spaces\n")
    assert run(env, "/memory")


def test_pal_md_goes_with_the_first_request(tmp_path):
    from pal.terminal.project import Project
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "PAL.md").write_text("Always use tabs.")
    p = Project(tmp_path, home=tmp_path / "home")
    p.prepare(graph_in_background=False)
    assert "Always use tabs." in p.brief() and "Project instructions (PAL.md)" in p.brief()


def test_add_dir_loads_another_folder_into_the_context(env, tmp_path):
    lib = tmp_path / "shared"
    lib.mkdir()
    (lib / "util.py").write_text("def u(): pass\n")
    run(env, f"/add-dir {lib}")
    assert any("util.py" in c for c in env.box["wf"].plan.b.context)


def test_resume_lists_plans_and_switches(env):
    env.box["wf"].new("second")
    run(env, "/resume")
    run(env, "/resume 1")
    assert env.box["wf"].plan.name == sorted(env.box["wf"].store.names())[0]


def test_old_clear_of_the_context_is_now_clear_context(env):
    (env.folder / "a.py").write_text("x")
    env.box["wf"].load(["a.py"])
    run(env, "/clear-context")
    assert env.box["wf"].plan.b.context == []


# -- history ------------------------------------------------------------------------------------------------------------------------
def test_history_is_kept_per_folder_without_keys(tmp_path):
    h = History(tmp_path / "a", home=tmp_path)
    for line in ("first", "second", "/key openai sk-abcdefghijklmnopqrstuvwxyz123456", "my key is sk-abcdefghijklmnopqrstuvwxyz123456", "second"):
        h.add(line)
    again = History(tmp_path / "a", home=tmp_path)
    assert again.items[:2] == ["first", "second"] and len(again.items) == 4 and "sk-abc" not in again.path.read_text()
    assert History(tmp_path / "b", home=tmp_path).items == []


def test_up_and_down_walk_the_history_and_keep_the_draft(tmp_path):
    h = History(tmp_path, home=tmp_path)
    for x in ("one", "two", "three"):
        h.add(x)
    b = Browser(h)
    assert b.older("draft") == "three" and b.older("") == "two" and b.older("") == "one" and b.older("") is None
    assert b.newer() == "two" and b.newer() == "three" and b.newer() == "draft" and b.newer() is None
    b2 = Browser(h)
    assert b2.find("t") == "three" and b2.find("t") == "two" and b2.find("t") is None


# -- voice ----------------------------------------------------------------------------------------------------------------------------
def test_voice_tries_the_engines_in_order_and_says_what_is_missing(monkeypatch):
    from pal.terminal import voice
    calls = []

    def bad(a, s):
        calls.append("local")
        raise RuntimeError("model download failed")
    t = voice.transcribe(b"RIFF", ".wav", prefer=["local", "openai"], runners={"local": bad, "openai": lambda a, s: calls.append("openai") or "hello pal"})
    assert t.text == "hello pal" and t.engine == "openai" and calls == ["local", "openai"]
    monkeypatch.setattr(voice, "engines", lambda: [])
    with pytest.raises(voice.VoiceUnavailable, match="reinstall Pal"):
        voice.transcribe(b"x")


def test_audio_is_written_as_16_bit_mono_wav():
    from pal.terminal.voice import RATE, wav_bytes
    data = wav_bytes([0.0, 0.5, -0.5, 1.2])
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, RATE, 4)


def test_the_web_transcribes_what_the_page_recorded(monkeypatch, tmp_path):
    from pal.terminal import voice
    from pal.web.app import create_app
    from tests.helpers import make_service
    svc, _ = make_service(tmp_path)
    c = create_app(service=svc).test_client()
    monkeypatch.setattr(voice, "engines", lambda: ["local"])
    assert c.get("/api/voice").get_json() == {"engines": ["local"]}
    seen = {}
    monkeypatch.setattr(voice, "transcribe", lambda audio, suffix=".wav": seen.update(n=len(audio), suffix=suffix) or voice.Transcript("add a login page", "local whisper (base)"))
    r = c.post("/api/transcribe", data=b"\x1aE\xdf\xa3webm", headers={"Content-Type": "audio/webm;codecs=opus"})
    assert r.get_json()["text"] == "add a login page" and seen == {"n": 8, "suffix": ".webm"}
    assert c.post("/api/transcribe", data=b"").status_code == 400


def test_voice_without_a_microphone_library_explains(monkeypatch):
    from pal.terminal import cli, voice
    monkeypatch.setattr(voice, "can_record", lambda: False)
    assert cli.voice_flow(ui.make_console()) == ""


# -- keys -------------------------------------------------------------------------------------------------------------------------
def test_esc_watcher_does_nothing_without_a_terminal():
    from pal.terminal.keys import EscToStop
    with EscToStop(enabled=False) as w:
        pass
    assert not w.pressed


def test_shift_tab_cycles_autonomy_but_never_into_full(env):
    from pal.terminal.cli import cycle_autonomy
    wf = env.box["wf"]
    wf.plan.autonomy = "semi"
    seen = [wf.plan.autonomy]
    for _ in range(5):
        cycle_autonomy(wf)
        seen.append(wf.plan.autonomy)
    assert "full" not in seen and set(seen) == {"none", "basic", "plus", "semi"}
    wf.plan.autonomy = "full"
    assert "autonomy: none" in cycle_autonomy(wf)
