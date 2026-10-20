"""The build workflow: plans and branches saved per folder, context, the change sandbox, apply with checks, rollback, auto-debug and commits."""
import subprocess

import pytest

from pal.terminal.clients import Reply
from pal.terminal.review import render
from pal.terminal.session import PalSession
from pal.terminal.workflow import Store, Workflow, git_commit, is_git_repo, mentioned_files, parse_steps
from tests.pal.fakes import files_reply


class Script:
    """A model that answers with the next reply in the list, and remembers every message and the references it got."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def ask(self, message, references=None):
        self.calls.append((message, references))
        text = self.replies.pop(0) if self.replies else "ok"
        return Reply(text, prompt_tokens=len(message) // 4, completion_tokens=len(text) // 4, seconds=0.01, calls=1)


def flow(tmp_path, *replies, mode="build", home=None):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    s = PalSession(Script(*replies), "m", "local", proj, mode)
    return Workflow(s, Store(proj, home or tmp_path / "home")), proj


def test_build_changes_wait_in_the_sandbox_and_your_files_do_not_change(tmp_path):
    wf, proj = flow(tmp_path, files_reply([("app.py", "x = 1\n")]), files_reply([("app.py", "x = 2\n"), ("util.py", "y = 1\n")]))
    wf.tell("make app")
    wf.tell("change it")
    assert not (proj / "app.py").exists() and wf.plan.b.pending == {"app.py": "x = 2\n", "util.py": "y = 1\n"}
    diffs = {p: lines for p, _, lines, _ in wf.diffs()}
    assert ["add", "x = 2"] in diffs["app.py"]
    second_message = wf.session.client.calls[1][0]
    assert "make app" not in second_message                       # the engine keeps the history; this request is only the new ask


def test_a_second_change_is_diffed_against_the_sandbox_not_the_disk(tmp_path):
    wf, _ = flow(tmp_path, files_reply([("a.py", "1\n")]), files_reply([("a.py", "2\n")]))
    wf.tell("one")
    t = wf.tell("two")
    assert t.changes[0].old == "1\n" and not t.changes[0].is_new


def test_rewind_and_reject_and_log(tmp_path):
    wf, _ = flow(tmp_path, files_reply([("a.py", "1\n")]), files_reply([("b.py", "2\n")]))
    wf.tell("a")
    wf.tell("b")
    assert [len(v.pending) for v in wf.log()] == [1, 2]
    assert wf.reject(["a.py"]) == ["a.py"] and wf.plan.b.pending == {"b.py": "2\n"}
    assert wf.rewind() == 1 and set(wf.plan.b.pending) == {"a.py", "b.py"}
    first = wf.log()[0].id
    wf.rewind(to=first)
    assert wf.plan.b.pending == {"a.py": "1\n"}
    with pytest.raises(ValueError):
        wf.rewind(to="nope")


def test_apply_writes_runs_the_check_and_records_a_version(tmp_path):
    wf, proj = flow(tmp_path, files_reply([("ok.py", "print('hi')\n")]) + "\n\n### COMMANDS\n```\npython ok.py\n```")
    wf.tell("go")
    assert wf.plan.b.commands == ["python ok.py"]
    ran = []
    res = wf.apply(lambda cmd: (ran.append(cmd), (True, "hi"))[1])
    assert res.written == ["ok.py"] and ran == ["python ok.py"] and not res.rolled_back and (proj / "ok.py").read_text() == "print('hi')\n"
    assert wf.plan.b.pending == {} and wf.log()[-1].label.startswith("applied")


def test_a_failing_check_puts_every_file_back(tmp_path):
    wf, proj = flow(tmp_path, files_reply([("a.py", "new\n"), ("b.py", "brand new\n")]) + "\n\n### COMMANDS\n```\npytest\n```")
    (proj / "a.py").write_text("old\n")
    wf.tell("change")
    res = wf.apply(lambda cmd: (False, "1 failed"))
    assert res.rolled_back and res.failed_command == "pytest" and (proj / "a.py").read_text() == "old\n" and not (proj / "b.py").exists()
    assert set(wf.plan.b.pending) == {"a.py", "b.py"}                           # still pending, to fix


def test_auto_debug_asks_the_build_agent_to_fix_and_tries_again(tmp_path):
    wf, proj = flow(tmp_path, files_reply([("m.py", "bug\n")]) + "\n\n### COMMANDS\n```\ncheck\n```", files_reply([("m.py", "fixed\n")]))
    wf.tell("make m")
    results = iter([(False, "Error: bug"), (True, "fine")])

    def debug(cmd, output):
        return bool(wf.tell(f"fix {cmd}: {output}").changes)
    res = wf.apply(lambda cmd: next(results), debug)
    assert not res.rolled_back and (proj / "m.py").read_text() == "fixed\n" and "Error: bug" in wf.session.client.calls[1][0]


def test_a_declined_check_leaves_the_changes_applied(tmp_path):
    wf, proj = flow(tmp_path, files_reply([("x.py", "1\n")]) + "\n\n### COMMANDS\n```\nrm -rf /\n```")
    wf.tell("go")
    res = wf.apply(lambda cmd: (None, ""))
    assert res.written == ["x.py"] and not res.rolled_back


def test_checks_never_run_when_can_exec_is_off(tmp_path):
    wf, _ = flow(tmp_path, files_reply([("x.py", "1\n")]) + "\n\n### COMMANDS\n```\nnpm test\n```")
    wf.plan.autonomy = "basic"
    wf.tell("go")
    res = wf.apply(lambda cmd: pytest.fail("must not run"))
    assert res.written == ["x.py"]


def test_plans_are_saved_per_folder_and_come_back_with_their_conversation(tmp_path):
    home = tmp_path / "home"
    wf, proj = flow(tmp_path, "1. Make a page\n2. Style it", mode="plan", home=home)
    wf.tell("a landing page")
    assert wf.plan.b.steps == ["Make a page", "Style it"]
    name = wf.plan.name
    wf2 = Workflow(PalSession(Script("ok"), "m", "local", proj, "ask"), Store(proj, home))
    assert wf2.plan.name == name and wf2.plan.b.steps == ["Make a page", "Style it"] and "a landing page" in wf2.session.preamble
    wf2.tell("what next?")
    assert "Earlier in this plan" in wf2.session.client.calls[0][0]
    other = Workflow(PalSession(Script(), "m", "local", tmp_path, "ask"), Store(tmp_path, home))
    assert other.plan.name != name or other.store.root != wf2.store.root      # another folder, other plans


def test_new_switch_rename_delete_plans(tmp_path):
    wf, _ = flow(tmp_path)
    first = wf.plan.name
    wf.new("billing")
    assert wf.store.names() == sorted([first, "billing"]) and wf.plan.name == "billing"
    wf.switch("1")
    assert wf.plan.name == sorted([first, "billing"])[0]
    wf.switch("billing")
    wf.rename("payments")
    assert "payments" in wf.store.names() and "billing" not in wf.store.names()
    with pytest.raises(ValueError):
        wf.new("bad name!")
    wf.delete("payments")
    assert wf.store.names() == [first] and wf.plan.name == first


def test_branches_start_as_copies_and_change_on_their_own(tmp_path):
    wf, _ = flow(tmp_path, files_reply([("a.py", "1\n")]), files_reply([("b.py", "2\n")]))
    wf.tell("a")
    assert wf.checkout("try-b") is True
    wf.tell("b")
    assert set(wf.plan.b.pending) == {"a.py", "b.py"}
    wf.checkout("main")
    assert set(wf.plan.b.pending) == {"a.py"}
    wf.delete_branch("try-b")
    assert list(wf.plan.branches) == ["main"]
    with pytest.raises(ValueError):
        wf.delete_branch("main")


def test_context_load_list_remove_and_it_goes_with_every_request(tmp_path):
    wf, proj = flow(tmp_path, "ok", "ok", mode="ask")
    (proj / "src").mkdir()
    (proj / "src" / "a.py").write_text("A = 1\n")
    (proj / "src" / "b.py").write_text("B = 2\n")
    (proj / ".env").write_text("KEY=sk-abcdefghijklmnopqrstuvwxyz")
    added, problems = wf.load(["src", "missing.py"], note="use tabs")
    assert added == ["note", "src/a.py", "src/b.py"] and problems == ["missing.py: nothing found"]
    assert [r[1] for r in wf.context_rows()][:2] == ["src/a.py", "src/b.py"]
    wf.tell("explain")
    refs = wf.session.client.calls[0][1]
    assert {r["name"] for r in refs} == {"src/a.py", "src/b.py", "notes.md"}
    wf.tell("again")
    assert wf.session.client.calls[1][1] is None                                   # unchanged: not sent again
    assert wf.remove(["1"]) == ["src/a.py"] and wf.remove(["src/*"]) == ["src/b.py"] and wf.clear() == 1


def test_the_plan_agent_loads_the_files_it_names_at_semi(tmp_path):
    wf, proj = flow(tmp_path, "1. Edit app.py and config/settings.py", mode="plan")
    (proj / "app.py").write_text("x")
    (proj / "config").mkdir()
    (proj / "config" / "settings.py").write_text("y")
    t = wf.tell("plan it")
    assert wf.plan.b.context == ["app.py", "config/settings.py"] and any("loaded into context" in w for w in t.warnings)


def test_continue_builds_one_step_at_a_time(tmp_path):
    wf, _ = flow(tmp_path, "1. first\n2. second", files_reply([("1.py", "1\n")]), files_reply([("2.py", "2\n")]), mode="plan")
    wf.tell("plan")
    wf.build_step()
    assert wf.plan.b.done == 1 and "step 1 of 2" in wf.session.client.calls[1][0] and wf.session.mode == "plan"
    wf.build_step()
    assert wf.plan.b.done == 2 and wf.build_step() is None and set(wf.plan.b.pending) == {"1.py", "2.py"}


def test_auto_commit_in_a_git_repository(tmp_path):
    if not is_git_repo(tmp_path) and subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git is not installed")
    wf, proj = flow(tmp_path, files_reply([("c.py", "c = 1\n")]))
    subprocess.run(["git", "init", "-q"], cwd=proj, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init"], cwd=proj, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=proj, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=proj, check=True)
    wf.tell("go")
    res = wf.apply(lambda cmd: (True, ""), commit_message=lambda files: "Add c")
    log = subprocess.run(["git", "log", "--oneline"], cwd=proj, capture_output=True, text=True).stdout
    assert res.commit and "Add c" in log


def test_git_commit_without_files_does_nothing(tmp_path):
    assert git_commit(tmp_path, [], "x") == ""


def test_the_review_page_escapes_everything_and_allows_no_scripts(tmp_path):
    wf, _ = flow(tmp_path, files_reply([("page.html", "<script>alert(1)</script>\n")]))
    wf.tell("go")
    page = render(wf)
    assert "<script>alert" not in page and "&lt;script&gt;alert(1)" in page and "default-src 'none'" in page and "page.html" in page


def test_steps_and_file_names_are_read_from_a_plan(tmp_path):
    assert parse_steps("Goal\n1. Add **login**\n2) Write `test_login.py`\nStep 3: Ship") == ["Add login", "Write test_login.py", "Ship"]
    (tmp_path / "a.py").write_text("x")
    assert mentioned_files("Change a.py and b.py, and ../../etc/passwd.txt", tmp_path) == ["a.py"]
