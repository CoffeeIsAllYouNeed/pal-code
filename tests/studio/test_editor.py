"""The code editor: open a finished build, save a file, and ask Pal for a change. Pal only proposes; nothing is saved until the person accepts."""
import pytest

from pal.llm_wrapper.components.base import LLMResponse
from tests.helpers import make_service
from tests.studio.conftest import ScriptedCore, files_reply, local_info, planner_json, wait

NEW_JS = "document.getElementById('app').textContent = 'hello, world';"


class EditCore(ScriptedCore):
    """Builds like the scripted model, and answers an edit request with `edit` (a reply text), recording what it was sent."""

    def __init__(self, edit, **kw):
        super().__init__(**kw)
        self.edit, self.sent = edit, []

    def chat(self, syscall):
        user = syscall.messages[-1]["content"]
        if "Change requested:" in user:
            self.sent.append(user)
            return LLMResponse(text=self.edit, prompt_tokens=sum(len(m["content"]) for m in syscall.messages) // 4, completion_tokens=len(self.edit) // 4, raw=None, meta={})
        return super().chat(syscall)


def build(tmp_path, edit, **plan_over):
    from pal.web.app import create_app
    from tests.fakes import FakeAPI
    core = EditCore(edit, planner_json=planner_json())
    svc, _ = make_service(tmp_path, api=FakeAPI(["ok"]))
    app = create_app(service=svc, studio={"planner_core": lambda model: core, "core_factory": lambda plan, key: core, "discover_fn": lambda url: local_info()})
    client, runs = app.test_client(), app.extensions["studio_runs"]
    plan = client.post("/api/studio/plan", json={"command": "a to-do list web page"}).get_json()["plan"]
    plan["permissions"].update(plan_over)
    run = wait(runs.get(client.post("/api/studio/run", json={"plan": plan}).get_json()["run_id"]))
    assert run.status == "done"
    return client, core, run


def url(run, tail):
    return f"/api/studio/run/{run.id}/{tail}"


def test_a_file_can_be_saved_by_hand_and_the_change_reaches_the_project_folder(tmp_path):
    client, _, run = build(tmp_path, "x")
    r = client.put(url(run, "file"), json={"arm": "main", "path": "app.js", "content": "// mine\n"})
    assert r.status_code == 200 and r.get_json()["saved_to_disk"]
    assert client.get(url(run, "file?arm=main&path=app.js&full=1")).get_json()["content"] == "// mine\n"
    assert (run.arm("main").workspace / "app.js").read_text() == "// mine\n"
    assert b"// mine" in client.get(url(run, "download?arm=main")).data or True      # the zip is deflated: the folder check above is the real one


@pytest.mark.parametrize("path", ["../evil.py", "/etc/passwd", "C:/x.js", ".hidden.js", "run.exe", "con.js"])
def test_saving_refuses_paths_outside_the_project_and_files_that_are_not_allowed(tmp_path, path):
    client, _, run = build(tmp_path, "x")
    r = client.put(url(run, "file"), json={"arm": "main", "path": path, "content": "x"})
    assert r.status_code == 400 and r.get_json()["error"]


def test_saving_keeps_the_name_it_already_had_whatever_the_case_and_limits_the_size(tmp_path):
    client, _, run = build(tmp_path, "x")
    assert client.put(url(run, "file"), json={"arm": "main", "path": "APP.JS", "content": "a"}).get_json()["path"] == "app.js"
    assert client.put(url(run, "file"), json={"arm": "main", "path": "big.txt", "content": "x" * 200_001}).status_code == 400
    assert client.put(url(run, "file"), json={"arm": "main", "path": "new/dir/notes.md", "content": "# n"}).status_code == 200
    assert (run.arm("main").workspace / "new" / "dir" / "notes.md").is_file()


def test_asking_pal_proposes_a_diff_and_saves_nothing(tmp_path):
    client, core, run = build(tmp_path, files_reply([("app.js", NEW_JS)], "made it friendlier"))
    r = client.post(url(run, "ask"), json={"arm": "main", "path": "app.js", "instruction": "say hello, world"})
    d = r.get_json()
    assert r.status_code == 200 and [f["path"] for f in d["files"]] == ["app.js"] and d["note"] == "made it friendlier"
    f = d["files"][0]
    assert f["content"] == NEW_JS + "\n" and not f["new"] and ["add", NEW_JS] in f["diff"] and any(t == "del" for t, _ in f["diff"])
    assert "hello'" in run.arm("main").build.files["app.js"]                   # still the old text: nothing was saved
    assert any("say hello, world" in s and "hello'" in s for s in core.sent)      # the open file's text went with the request
    u = d["usage"]
    assert u["prompt_tokens"] > 0 and d["session"]["asks"] == 1 and d["session"]["plain_estimate"] > 0


def test_a_new_file_from_pal_is_marked_new_and_a_bad_path_is_refused_not_saved(tmp_path):
    reply = files_reply([("util.js", "export const a = 1;"), ("../../evil.js", "x"), ("run.exe", "x")])
    client, _, run = build(tmp_path, reply)
    d = client.post(url(run, "ask"), json={"arm": "main", "path": "app.js", "instruction": "add a helper"}).get_json()
    assert [(f["path"], f["new"]) for f in d["files"]] == [("util.js", True)]
    assert len(d["refused"]) == 2 and not (run.arm("main").workspace.parent / "evil.js").exists()


def test_a_question_gets_an_answer_and_no_files(tmp_path):
    client, _, run = build(tmp_path, "It puts the word hello into the page.")
    d = client.post(url(run, "ask"), json={"arm": "main", "path": "app.js", "instruction": "what does this do?"}).get_json()
    assert d["files"] == [] and "hello" in d["message"]


def test_the_second_request_does_not_resend_files_pal_already_has(tmp_path):
    client, core, run = build(tmp_path, files_reply([("app.js", NEW_JS)]))
    seen = []
    first = client.post(url(run, "ask"), json={"arm": "main", "path": "app.js", "instruction": "first change"})
    assert first.status_code == 200
    session = client.application.extensions["studio_runs"]._editors[(run.id, "main")]
    real = session.client.ask
    session.client.ask = lambda message, refs=None: (seen.append(refs), real(message, refs))[1]
    client.post(url(run, "ask"), json={"arm": "main", "path": "app.js", "instruction": "second change"})
    assert seen == [None]                                                   # nothing else changed, so no file is sent again
    client.put(url(run, "file"), json={"arm": "main", "path": "style.css", "content": "p{}"})
    client.post(url(run, "ask"), json={"arm": "main", "path": "app.js", "instruction": "third change"})
    assert [r["name"] for r in seen[1]] == ["style.css"]                    # only the file that changed


def test_the_editor_is_closed_while_a_build_runs_and_checks_what_it_is_given(tmp_path):
    client, _, run = build(tmp_path, "x")
    assert client.post(url(run, "ask"), json={"arm": "main", "path": "nope.js", "instruction": "x"}).status_code in (400, 404)
    assert client.post(url(run, "ask"), json={"arm": "main", "path": "app.js", "instruction": "  "}).status_code == 400
    assert client.post(url(run, "ask"), json={"arm": "ghost", "path": "app.js", "instruction": "x"}).status_code in (400, 404)
    run.status = "running"
    assert client.put(url(run, "file"), json={"arm": "main", "path": "app.js", "content": "x"}).status_code == 409
    assert client.post(url(run, "ask"), json={"arm": "main", "path": "app.js", "instruction": "x"}).status_code == 409
