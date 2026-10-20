"""The syntax check before review, /debug, one terminal per plan, pal bench, and reasoning text kept out of answers."""
import os

import pytest

from pal.terminal.clients import Reply
from pal.terminal.session import PalSession, strip_thinking
from pal.terminal.syntax import check
from pal.terminal.workflow import Store, Workflow, pid_alive
from tests.pal.fakes import files_reply


class Script:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def ask(self, message, references=None, query=None):
        self.calls.append(message)
        text = self.replies.pop(0) if self.replies else "ok"
        return Reply(text, prompt_tokens=len(message) // 4, completion_tokens=len(text) // 4, seconds=0.01, calls=1)


@pytest.mark.parametrize("path,text,bad_line", [
    ("a.py", "def f(:\n    pass\n", 1), ("a.py", "x = 1\n", None), ("c.json", '{"a": 1,\n}', 1), ("d.yml", "a: [1, 2\nb: 3", 2),
    ("e.ts", "export const f = <T,>(a: T): T => a;\n", None), ("e.ts", "const x: number = ;\n", 1),
    ("g.go", 'package main\nfunc main() {\n fmt.Println("x"\n}\n', 3), ("h.rs", "fn main() { let x = 1; }\n", None),
    ("k.tsx", "export default function A(){ return <div className='x'>hi</div> }\n", None), ("j.css", "broken {", None), ("notes.md", "# (", None),
])
def test_syntax_is_checked_for_many_languages(path, text, bad_line):
    found = check(path, text)
    line = found.line if found else None
    if path.endswith(".json") and bad_line:                     # Python 3.12 points at the line after a trailing comma, 3.13 at the comma
        assert line in (bad_line, bad_line + 1)
    else:
        assert line == bad_line


def test_a_file_that_does_not_parse_goes_back_to_the_build_agent_before_you_see_it(tmp_path):
    s = PalSession(Script(files_reply([("app.py", "def f(:\n    return 1\n")]), files_reply([("app.py", "def f():\n    return 1\n")])), "m", "local", tmp_path, "build")
    t = s.ask("make f")
    assert [c.new for c in t.changes] == ["def f():\n    return 1\n"] and "app.py line 1" in s.client.calls[1]
    assert any("fixed a syntax error" in w for w in t.warnings) and s.pal_tokens >= t.prompt_tokens


def test_a_file_still_broken_after_two_tries_is_marked_so_it_is_not_applied_by_mistake(tmp_path):
    broken = files_reply([("app.py", "def f(:\n")])
    s = PalSession(Script(broken, broken, broken), "m", "local", tmp_path, "build")
    t = s.ask("make f")
    assert len(s.client.calls) == 3 and t.changes[0].findings[0].title == "Syntax error" and any("still has a syntax error" in w for w in t.warnings)


def test_reasoning_is_kept_out_of_the_answer(tmp_path):
    assert strip_thinking("<think>\nlet me see\n</think>\n\nThe answer") == "The answer"
    assert strip_thinking("no thinking here") == "no thinking here"
    s = PalSession(Script("<think>plan the files</think>\n" + files_reply([("a.py", "x = 1\n")])), "m", "local", tmp_path, "build")
    t = s.ask("go")
    assert "plan the files" not in t.text and t.changes[0].path == "a.py"


# -- /debug -------------------------------------------------------------------------------------------------------------------------
def _flow(tmp_path, *replies):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    s = PalSession(Script(*replies), "m", "local", proj, "ask")
    return Workflow(s, Store(proj, tmp_path / "home")), proj


def test_debug_runs_the_command_applies_the_fix_and_runs_it_again(tmp_path, monkeypatch):
    from pal.terminal import flow_cli, ui
    wf, proj = _flow(tmp_path, files_reply([("calc.py", "def add(a, b):\n    return a + b\n")]))
    (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (proj / "check.py").write_text("from calc import add\nassert add(2, 2) == 4, 'add is wrong'\n")
    monkeypatch.setattr(ui, "confirm", lambda console, q: True)
    ok = flow_cli.debug_flow(wf, "python check.py", ui.make_console())
    assert ok and (proj / "calc.py").read_text().endswith("a + b\n") and "add is wrong" in wf.session.client.calls[0]
    assert wf.session.mode == "ask"                                 # the agent the person was using is back


def test_debug_that_never_passes_offers_to_put_the_files_back(tmp_path, monkeypatch):
    from pal.terminal import flow_cli, ui
    wf, proj = _flow(tmp_path, *[files_reply([("calc.py", f"def add(a, b):\n    return a * b  # try {i}\n")]) for i in range(5)])
    (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (proj / "check.py").write_text("from calc import add\nassert add(2, 3) == 5\n")
    asked = []
    monkeypatch.setattr(ui, "confirm", lambda console, q: asked.append(q) or True)
    ok = flow_cli.debug_flow(wf, "python check.py", ui.make_console(), tries=2)
    assert not ok and any("back as they were" in q for q in asked) and (proj / "calc.py").read_text().endswith("a - b\n")


def test_debug_stops_when_the_fix_is_declined(tmp_path, monkeypatch):
    from pal.terminal import flow_cli, ui
    wf, proj = _flow(tmp_path, files_reply([("calc.py", "x = 2\n")]))
    (proj / "calc.py").write_text("x = 1\n")
    wf.plan.custom["auto-apply"] = False                                 # the asking path (auto-apply is on by default)
    monkeypatch.setattr(ui, "confirm", lambda console, q: False)
    assert flow_cli.debug_flow(wf, "python -c \"import sys; sys.exit(1)\"", ui.make_console()) is False
    assert (proj / "calc.py").read_text() == "x = 1\n" and wf.plan.b.pending == {"calc.py": "x = 2\n"}


def test_debug_is_a_command(tmp_path, monkeypatch):
    from pal.terminal import flow_cli, ui
    wf, _ = _flow(tmp_path)
    seen = []
    monkeypatch.setattr(flow_cli, "debug_flow", lambda wf, cmd, console: seen.append(cmd))
    assert flow_cli.handle_flow("/debug", "npm test", wf, ui.make_console()) and seen == ["npm test"]
    assert any(name == "/debug" for name, _ in flow_cli.FLOW_COMMANDS)


# -- one terminal per plan ------------------------------------------------------------------------------------------------------------
def test_a_plan_open_in_another_pal_is_not_opened_twice(tmp_path, monkeypatch):
    wf, proj = _flow(tmp_path)
    name = wf.plan.name
    other_pid = os.getpid() + 1
    (wf.store.root / f"{name}.lock").write_text(f"{other_pid} 0")
    import pal.terminal.workflow as w
    monkeypatch.setattr(w, "pid_alive", lambda pid: pid == other_pid)
    wf2 = Workflow(PalSession(Script(), "m", "local", proj, "ask"), Store(proj, tmp_path / "home"))
    assert wf2.plan.name != name and "open in another Pal" in wf2.notice
    with pytest.raises(ValueError, match="open in another Pal"):
        wf2.switch(name)
    monkeypatch.setattr(w, "pid_alive", lambda pid: False)              # the other Pal has exited: its lock is taken over
    wf2.switch(name)
    assert wf2.plan.name == name and (wf2.store.root / f"{name}.lock").read_text().split()[0] == str(os.getpid())


def test_closing_frees_the_plan(tmp_path):
    wf, _ = _flow(tmp_path)
    lock = wf.store.root / f"{wf.plan.name}.lock"
    assert lock.is_file()
    wf.close()
    assert not lock.exists()


def test_this_process_is_alive_and_a_made_up_one_is_not():
    assert pid_alive(os.getpid()) and not pid_alive(2 ** 22 + 12345)


# -- pal bench ------------------------------------------------------------------------------------------------------------------------
def test_bench_runs_the_same_task_both_ways_in_copies_of_the_folder(tmp_path):
    from pal.llm_wrapper.components.base import (
        DeploymentType,
        LLMResponse,
        LLMWrapper,
        WrapperCapabilities,
        WrapperConfig,
    )
    from pal.terminal.bench import run_bench
    from pal.tokens import estimate_tokens

    class Core(LLMWrapper):
        name = "bench-fake"

        def __init__(self):
            super().__init__(WrapperConfig(backend="ollama", model="fake"))

        def chat(self, s):
            u = s.messages[-1]["content"]
            t = files_reply([("tests/test_slug.py", "def test_slug():\n    assert True\n")]) if "test" in u.lower()[:120] else \
                "It joins words with dashes." if "three sentences" in u else files_reply([("app/slug.py", "def slug(s):\n    return '-'.join(s.split())\n")])
            return LLMResponse(t, prompt_tokens=sum(estimate_tokens(m["content"]) for m in s.messages), completion_tokens=estimate_tokens(t))

        def token_count(self, text):
            return estimate_tokens(text)

        def capabilities(self):
            return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER)
    folder = tmp_path / "proj"
    (folder / "app").mkdir(parents=True)
    for i in range(12):
        (folder / "app" / f"mod{i}.py").write_text("\n".join(f"def f{j}(x):\n    return x + {j}\n" for j in range(60)))
    r = run_bench(folder, "Add slug() in app/slug.py", Core, "fake", home=tmp_path / "home")
    assert len(r.plain.steps) == len(r.pal.steps) == 3 and not r.plain.error and not r.pal.error
    assert r.pal.total < r.plain.total and r.saved_pct > 50
    assert r.pal.steps[0].files == ["app/slug.py"] and r.plain.steps[1].files == ["tests/test_slug.py"]
    assert not (folder / "app" / "slug.py").exists()                     # the person's folder is not changed
    assert list((tmp_path / "home" / "bench").glob("bench-*.json")) and r.plain.estimated_calls == 0


def test_bench_command_needs_a_model(capsys, monkeypatch, tmp_path):
    from pal.terminal import backend
    from pal.terminal.cli import main
    monkeypatch.setenv("PAL_HOME", str(tmp_path))
    monkeypatch.setattr(backend, "local_info", lambda host=None: {"ok": False, "models": []})
    assert main(["bench", "make a thing"]) == 2 and "choose a model" in capsys.readouterr().err


# -- newer models -----------------------------------------------------------------------------------------------------------------
def test_newer_models_are_in_the_catalogue_and_thinking_ones_are_not_picked_by_themselves():
    from pal.terminal import local_models as lm
    for tag in ("qwen3-coder:30b", "devstral-small-2:24b", "gpt-oss:20b", "gemma4:12b", "qwen3.5:9b", "qwen3.6:27b"):
        assert tag in lm.BY_TAG
    assert lm.BY_TAG["gpt-oss:20b"].thinks and lm.BY_TAG["qwen3.6:27b"].thinks and not lm.BY_TAG["qwen3-coder:30b"].thinks


def test_word_files_convert_without_pandoc():
    from pal.input_converters.components.pandoc_converter import docx_without_pandoc
    from tests.pal.test_project import make_docx
    md = docx_without_pandoc(make_docx([("Title", "Plan"), ("Heading2", "Goals"), ("ListParagraph", "fast"), ("ListParagraph", "cheap"), ("", "End.")]))
    assert md == "# Plan\n\n## Goals\n\n- fast\n- cheap\n\nEnd."


def test_a_syntax_fix_that_matches_the_file_on_disk_drops_the_broken_change(tmp_path):
    (tmp_path / "d.py").write_text("def d(x):\n    return x\n")
    s = PalSession(Script(files_reply([("d.py", "def d(x:\n    return x\n")]), files_reply([("d.py", "def d(x):\n    return x\n")])), "m", "local", tmp_path, "build")
    t = s.ask("touch d")
    assert t.changes == [] and not any("still has" in w for w in t.warnings)


def test_a_pending_file_that_does_not_parse_is_not_applied_without_asking(tmp_path, monkeypatch):
    from pal.terminal import flow_cli, ui
    wf, proj = _flow(tmp_path)
    wf.plan.b.versions[-1:] = []
    from pal.terminal.workflow import Version
    wf.plan.b.versions.append(Version("v1", "hand", 0.0, {"bad.py": "def f(:\n"}, []))
    asked = []
    monkeypatch.setattr(ui, "confirm", lambda console, q: asked.append(q) or False)
    flow_cli.apply_flow(wf, ui.make_console(), yes=True)
    assert "does not parse" in asked[0] and not (proj / "bad.py").exists()


def test_the_ask_agent_can_ask_the_person_to_choose(tmp_path, monkeypatch):
    from pal.terminal import decide, flow_cli, ui
    block = ('Two readings.\n\n### QUESTION\n```json\n{"header": "Scope", "question": "Explain which part?", '
             '"options": [{"label": "The API", "description": "routes"}, {"label": "The UI"}], "recommended": 0}\n```')
    wf, _ = _flow(tmp_path, block, "The API has two routes.")
    t = wf.session.ask("explain it")
    assert t.text == "Two readings." and t.questions[0].options[0].label == "The API"
    monkeypatch.setattr(decide, "ask", lambda console, q, reader=None: "The API")
    (tmp_path / "b").mkdir()
    wf2, _ = _flow(tmp_path / "b", block, "The API has two routes.")
    flow_cli.run_turn(wf2, "explain it", ui.make_console())
    assert "My decisions:\n- Explain which part? → The API" in wf2.session.client.calls[1]


def test_build_and_plan_never_get_the_question_instructions(tmp_path):
    for mode in ("build", "plan"):
        s = PalSession(Script("ok"), "m", "local", tmp_path, mode)
        s.ask("make a thing")
        assert "### QUESTION" not in s.client.calls[0]
    s = PalSession(Script("ok"), "m", "local", tmp_path, "ask")
    s.ask("what is this?")
    assert "### QUESTION" in s.client.calls[0]


def test_questions_parse_and_bad_blocks_stay_text():
    from pal.terminal.decide import Question, ask, parse_questions
    qs, rest = parse_questions('Hi\n### QUESTIONS\n```json\n[{"question": "A?", "options": ["x", "y"]}, {"question": "B?", "options": ["only one"]}]\n```')
    assert rest == "Hi" and len(qs) == 1 and [o.label for o in qs[0].options] == ["x", "y"]
    assert parse_questions("### QUESTION\n```json\nnot json\n```")[0] == []
    from pal.terminal.decide import Option
    from pal.terminal.ui import make_console
    q = Question("Pick?", [Option("a"), Option("b")], recommended=1)
    assert ask(make_console(), q, reader=lambda p: "") == "b"
    assert ask(make_console(), q, reader=lambda p: "1") == "a"
    answers = iter(["3", "my own way"])
    assert ask(make_console(), q, reader=lambda p: next(answers)) == "my own way"
