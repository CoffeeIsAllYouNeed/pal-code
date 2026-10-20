"""The build loop, the runs (background, cancel, build twice) and the web endpoints, with a scripted model: no network, no Ollama."""
import io
import json
import zipfile

import pytest

from pal.studio import plan as planlib
from pal.studio.builder import Build, default_name, first_message, step_message
from pal.studio.clients import CountingWrapper, PalClient, PlainClient, Reply
from pal.studio.planner import Planner, extract_json, heuristic_plan, heuristic_revise
from pal.studio.policy import Policy
from pal.studio.runs import RunBusy, RunStore, compare
from tests.studio.conftest import GOOD, INDEX, ScriptedCore, files_reply, local_info, planner_json, wait

INFO = {"installed": ["llama3.1:8b"], "recommended": "llama3.1:8b", "ready_tools": ["calculator"]}


def make_plan(**raw):
    plan, _ = planlib.normalize({"product": {"name": "Todo app", "summary": "A to-do list page.", "kind": "web_app"}, "roles": ["designer", "coder", "docs", "reviewer"], **raw},
                                installed=["llama3.1:8b"], recommended="llama3.1:8b")
    return plan


class Scripted:
    """A client that answers each step from a list of replies."""
    label = "scripted"

    def __init__(self, replies, tokens=(100, 40, 0)):
        self.replies, self.tokens, self.asked = list(replies), tokens, []

    def ask(self, message, references=None):
        self.asked.append((message, references))
        text = self.replies.pop(0) if self.replies else ""
        return Reply(text, self.tokens[0], self.tokens[1], self.tokens[2], 0.5, 1)

    def close(self):
        pass


def run_build(plan, replies, perms=None, tmp=None, refs=None, cancel=lambda: False):
    events = []
    plan = {**plan, "permissions": {**plan["permissions"], **(perms or {})}}
    client = Scripted(replies)
    b = Build(plan, client, tmp, Policy(dict(plan["permissions"])), lambda kind, text="", **d: events.append((kind, text, d)), cancel, refs or [])
    return b, b.run(), events, client


# -- the planner ----------------------------------------------------------------------------------------------------------------------------------------
def test_json_is_found_in_fences_prose_and_with_trailing_commas_and_strings_with_braces():
    assert extract_json('Sure:\n```json\n{"a": 1, "b": [1, 2,],}\n```\nHope it helps') == {"a": 1, "b": [1, 2]}
    assert extract_json('text {"note": "a } brace", "x": {"y": 2}} more') == {"note": "a } brace", "x": {"y": 2}}
    assert extract_json('{"broken": } {"ok": 1}') == {"ok": 1}
    for bad in ("", "no json", "[1, 2]", "{unclosed"):
        with pytest.raises(ValueError):
            extract_json(bad)


def test_the_bot_turns_what_you_want_into_a_plan_and_explains_it():
    core = ScriptedCore(planner_json=planner_json())
    r = Planner(core, {**INFO, "installed": ["llama3.1:8b-instruct-q4_K_M"], "recommended": "llama3.1:8b-instruct-q4_K_M"}).plan("a to-do list web page")
    assert r.source == "model" and r.plan["roles"] == ["designer", "coder", "docs", "reviewer"] and r.plan["tools"] == ["calculator"]
    assert "**Todo app**" in r.message and "does not go online" in r.message and "does not run the code" in r.message and "press Build" in r.message


def test_the_bot_cannot_switch_on_going_online_even_if_it_tries():
    core = ScriptedCore(planner_json=planner_json(tools=["wikipedia"], permissions={"network_access": True}))
    r = Planner(core, {**INFO, "ready_tools": ["calculator", "wikipedia"]}).plan("look things up online and build a quiz")
    assert not r.plan["permissions"]["network_access"] and r.plan["tools"] == [] and any("only you can switch it on" in n for n in r.notes)


def test_an_unreadable_answer_is_retried_once_then_simple_rules_make_the_plan_and_say_so():
    core = ScriptedCore(planner_json="I think you should build a nice app!")
    r = Planner(core, INFO).plan("a python script that converts csv files to json")
    assert r.source == "fallback" and r.plan["product"]["kind"] == "cli_tool" and r.plan["roles"] == ["coder", "tester", "docs", "reviewer"] and "simple rules" in r.message
    assert sum(1 for system, _ in core.calls if system.startswith("You are the planner")) == 2             # one retry, with the error fed back


def test_a_model_that_cannot_be_reached_still_gives_a_plan():
    class Down:
        def chat(self, syscall):
            raise ConnectionError("refused")
    r = Planner(Down(), INFO).plan("a landing page for my bakery")
    assert r.source == "fallback" and r.plan["product"]["kind"] == "web_app" and "designer" in r.plan["roles"]


@pytest.mark.parametrize("command,kind,has", [("a portfolio website with a contact form", "web_app", "designer"), ("a command line tool to rename files", "cli_tool", "tester"),
                                              ("write unit tests for a calculator page", "web_app", "tester"), ("something", "web_app", "coder")])
def test_the_fallback_rules_pick_sensible_agents(command, kind, has):
    raw = heuristic_plan(command, INFO)
    plan, _ = planlib.normalize(raw, installed=["llama3.1:8b"], recommended="llama3.1:8b")
    assert plan["product"]["kind"] == kind and has in plan["roles"] and plan["model"]["model"] == "llama3.1:8b"


def test_the_bot_changes_the_plan_when_you_ask_and_lists_exactly_what_changed():
    plan = make_plan()
    revised = json.loads(planner_json(roles=["designer", "coder", "tester", "docs", "reviewer"], tools=["calculator"], metrics=True))
    core = ScriptedCore(planner_json=json.dumps(revised))
    r = Planner(core, INFO).revise(plan, "add a tester and the calculator and build twice to compare")
    assert "added agent: Tester" in r.changes and "added tool: calculator" in r.changes and "build twice to compare: on" in r.changes and r.message.startswith("Done:")
    assert Planner(ScriptedCore(planner_json=json.dumps(plan)), INFO).revise(plan, "keep it as it is").message.startswith("I made no change")


def test_when_the_model_cannot_be_read_a_few_plain_edits_still_work_and_anything_else_is_not_guessed():
    plan = make_plan()
    bad = ScriptedCore(planner_json="nope")
    assert "removed agent: Docs writer" in Planner(bad, INFO).revise(plan, "remove the docs writer").changes
    assert "added agent: Tester" in Planner(bad, INFO).revise(plan, "also add a tester").changes
    r = Planner(bad, INFO).revise(plan, "build twice to compare")
    assert "build twice to compare: on" in r.changes and r.plan["metrics"] is True
    r = Planner(bad, INFO).revise(plan, "make it purple")
    assert r.changes == [] and "did not understand" in r.message
    assert heuristic_revise(plan, "remove the reviewer", INFO)["roles"] == ["designer", "coder", "docs"]


# -- a build ---------------------------------------------------------------------------------------------------------------------------------------------
def test_a_build_writes_each_steps_files_checks_them_and_counts_every_token(tmp_path):
    replies = [files_reply(GOOD["Designer"]), files_reply(GOOD["Coder"]), files_reply(GOOD["Docs writer"]), files_reply(GOOD["Reviewer"])]
    b, totals, events, client = run_build(make_plan(), replies, tmp=tmp_path)
    assert sorted(b.files) == ["README.md", "app.js", "index.html", "style.css"] and (tmp_path / "style.css").read_text() == "body { font-family: sans-serif; }\n"
    assert totals["steps_done"] == 4 and totals["files"] == 4 and totals["problems_left"] == 0 and totals["blocked"] == 0 and totals["calls"] == 4
    assert totals["main_tokens"] == 4 * 140 and totals["prompt_tokens"] == 400 and totals["seconds"] == 2.0
    kinds = [k for k, _, _ in events]
    assert kinds.count("step_start") == kinds.count("step_done") == 4 and kinds.count("model_call") == 4 and "no problems found" in " ".join(t for k, t, _ in events if k == "check")
    assert "Step 1 (Designer)" in client.asked[0][0] and "Step 2 of 4 (Coder)" in client.asked[1][0] and "Open index.html" not in client.asked[0][0]
    assert "no external URLs or CDNs" in client.asked[0][0] and "COMPLETE content" in client.asked[1][0]


def test_a_mistake_is_sent_back_once_and_the_fixed_files_replace_the_broken_ones(tmp_path):
    broken = [("index.html", INDEX), ("style.css", "body { color: red;")]
    replies = [files_reply(broken), files_reply([("style.css", "body { color: black; }")]), files_reply(GOOD["Coder"]), files_reply(GOOD["Docs writer"]), files_reply(GOOD["Reviewer"])]
    plan = make_plan()
    b, totals, events, client = run_build(plan, replies, tmp=tmp_path)
    assert "problems were found" in client.asked[1][0] and "style.css" in client.asked[1][0] and b.files["style.css"] == "body { color: black; }\n"
    assert totals["problems_left"] == 0 and totals["calls"] == 5 and any("found problems" in t for k, t, _ in events if k == "check")


def test_a_problem_that_one_fix_does_not_solve_is_reported_not_hidden():
    still_broken = files_reply([("a.py", "def f(:")])
    b, totals, events, _ = run_build(make_plan(roles=["coder"]), [still_broken, still_broken], perms={"write_project": False})
    assert totals["problems_left"] == 1 and any("still there after one fix" in t for k, t, _ in events)


def test_files_the_policy_refuses_are_reported_and_never_written(tmp_path):
    reply = files_reply([("../evil.html", "x"), ("run.bat", "del *"), ("ok.html", "<p>fine</p>"), ("/etc/passwd.txt", "x")])
    b, totals, events, _ = run_build(make_plan(roles=["coder"]), [reply], tmp=tmp_path)
    assert list(b.files) == ["ok.html"] and totals["blocked"] == 3 and [p.name for p in tmp_path.iterdir()] == ["ok.html"]
    blocked = [d for k, _, d in events if k == "blocked"]
    assert len(blocked) == 3 and all(d["reason"] for d in blocked)
    assert not (tmp_path.parent / "evil.html").exists()


def test_with_saving_switched_off_the_files_are_shown_but_never_saved(tmp_path):
    b, totals, events, _ = run_build(make_plan(roles=["coder"]), [files_reply(GOOD["Coder"])], perms={"write_project": False}, tmp=tmp_path)
    assert "app.js" in b.files and list(tmp_path.iterdir()) == [] and any(d.get("action") == "dry" for k, _, d in events if k == "file")


def test_changing_a_file_it_already_wrote_can_be_switched_off(tmp_path):
    replies = [files_reply([("a.js", "let x = 1;")]), files_reply([("a.js", "let x = 2;")])]
    plan = make_plan(roles=["coder", "reviewer"])
    b, totals, events, _ = run_build(plan, replies, perms={"modify_own_files": False}, tmp=tmp_path)
    assert b.files["a.js"] == "let x = 1;\n" and totals["blocked"] == 1
    b2, _, _, _ = run_build(plan, [files_reply([("a.js", "let x = 1;")]), files_reply([("a.js", "let x = 2;")])], tmp=tmp_path / "x")
    assert b2.files["a.js"] == "let x = 2;\n"


def test_an_unnamed_code_block_uses_the_default_file_name_for_the_role_and_kind():
    plan = make_plan(roles=["coder", "docs"])
    assert default_name(plan, {"role": "coder"}) == "index.html" and default_name(plan, {"role": "docs"}) == "README.md" and default_name(plan, {"role": "tester"}) == "test_main.py"
    cli = planlib.normalize({"product": {"kind": "cli_tool"}, "roles": ["coder"]}, installed=["x"], recommended="x")[0]
    assert default_name(cli, {"role": "coder"}) == "main.py"
    b, totals, events, _ = run_build(make_plan(roles=["coder"]), ["```html\n<p>only block</p>\n```"], perms={"write_project": False})
    assert "index.html" in b.files


def test_the_reference_files_go_to_the_model_only_when_reading_them_is_allowed():
    refs = [{"name": "brief.md", "text": "# Brief"}]
    _, _, _, client = run_build(make_plan(roles=["coder"]), [files_reply(GOOD["Coder"])], refs=refs)
    assert client.asked[0][1] == refs and "reference files the person attached are available" in client.asked[0][0]
    _, _, _, off = run_build(make_plan(roles=["coder"]), [files_reply(GOOD["Coder"])], perms={"read_references": False}, refs=refs)
    assert off.asked[0][1] == [] and "reference files" not in off.asked[0][0]


def test_a_model_that_fails_stops_the_build_cleanly_with_what_was_done_so_far(tmp_path):
    class Failing(Scripted):
        def ask(self, message, references=None):
            if len(self.asked) == 1:
                self.asked.append(message)
                raise ConnectionError("the model went away")
            return super().ask(message, references)
    plan = make_plan()
    events = []
    b = Build(plan, Failing([files_reply(GOOD["Designer"])]), tmp_path, Policy(dict(plan["permissions"])), lambda k, t="", **d: events.append((k, t)), lambda: False, [])
    totals = b.run()
    assert totals["stopped"] == "error" and totals["steps_done"] == 1 and sorted(b.files) == ["index.html", "style.css"]
    assert any(k == "error" and "went away" in t for k, t in events)


def test_cancelling_stops_before_the_next_step():
    seen = []
    plan = make_plan()
    b = Build(plan, Scripted([files_reply(GOOD["Designer"])] * 4), None, Policy(dict(plan["permissions"])), lambda k, t="", **d: seen.append(k), lambda: len(seen) > 5, [])
    totals = b.run()
    assert totals["stopped"] == "cancelled" and totals["steps_done"] < 4 and "cancelled" in seen


def test_the_messages_to_the_model_name_the_product_the_team_and_the_rules():
    plan = make_plan()
    m = first_message(plan, plan["steps"][0], False)
    assert "Todo app" in m and "Designer, Coder, Docs writer, Reviewer" in m and "### FILE:" in m and "Step 1 (Designer)" in m
    assert step_message(plan["steps"][2], 3, 4).startswith("Step 3 of 4 (Docs writer)") and "consistent" in step_message(plan["steps"][2], 3, 4)


# -- the clients ---------------------------------------------------------------------------------------------------------------------------------------
def test_the_counting_wrapper_uses_the_models_numbers_and_estimates_when_there_are_none():
    core = ScriptedCore()
    w = CountingWrapper(core)
    w.chat(type("S", (), {"messages": [{"role": "user", "content": "x" * 400}], "temperature": 0, "max_tokens": 5})())
    snap = w.snapshot()
    assert snap.prompt == 100 and snap.calls == 1 and snap.completion >= 1 and snap.seconds >= 0

    class Silent(ScriptedCore):
        def chat(self, syscall):
            r = super().chat(syscall)
            r.prompt_tokens = r.completion_tokens = 0
            return r
    w2 = CountingWrapper(Silent())
    w2.chat(type("S", (), {"messages": [{"role": "user", "content": "x" * 400}]})())
    assert w2.snapshot().prompt == 100 and w2.snapshot().completion >= 1


def test_the_plain_client_resends_the_whole_conversation_and_pastes_the_references_every_time_they_are_given():
    core = ScriptedCore()
    c = PlainClient(core, "system")
    first = c.ask("one", [{"name": "brief.md", "text": "B" * 400}])
    second = c.ask("two")
    assert second.prompt_tokens > first.prompt_tokens and "Reference files from the person" in c.messages[1]["content"] and len(c.messages) == 5
    assert first.calls == 1 and first.helper_tokens == 0


def test_the_pal_client_runs_the_real_pipeline_and_counts_the_helpers_jobs_separately(tmp_path):
    core = ScriptedCore()
    c = PalClient(core, None, frozenset(), home=tmp_path / "h", model_name="scripted")
    r = c.ask("Hello! Could you please write the page now, thanks so much!")
    c.close()
    assert r.text.startswith("### FILE:") and r.prompt_tokens > 0 and r.helper_tokens > 0 and r.calls >= 1
    assert any(s.startswith("You route") or s.startswith("Edit") or s.startswith("You audit") for s, _ in core.calls) or r.helper_tokens > 0


# -- runs ------------------------------------------------------------------------------------------------------------------------------------------------
def make_store(tmp_path, core=None, **kw):
    core = core or ScriptedCore()
    return RunStore(tmp_path / "studio", lambda plan, key: core, **kw), core


def test_a_run_builds_in_the_background_reports_events_and_keeps_the_files(tmp_path):
    store, _ = make_store(tmp_path)
    run = wait(store.start(make_plan(), [], None))
    assert run.status == "done" and run.summary()["arms"]["main"]["label"] == "Pal" and run.summary()["arms"]["main"]["totals"]["files"] == 4
    kinds = [e["kind"] for e in run.events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_done" and "arm_done" in kinds and [e["id"] for e in run.events] == list(range(len(run.events)))
    assert [e["id"] for e in run.events_since(5)] == list(range(5, len(run.events))) and run.events_since(999) == []
    assert (tmp_path / "studio" / run.id / "main" / "todo-app" / "index.html").is_file()


def test_the_zip_holds_the_project_under_its_folder_name(tmp_path):
    store, _ = make_store(tmp_path)
    run = wait(store.start(make_plan(), [], None))
    z = zipfile.ZipFile(io.BytesIO(store.zip_bytes(run, "main")))
    assert sorted(z.namelist()) == ["todo-app/README.md", "todo-app/app.js", "todo-app/index.html", "todo-app/style.css"]
    assert z.read("todo-app/app.js").decode().startswith("document.getElementById")
    dry = wait(store.start(make_plan(permissions={"write_project": False}), [], None))
    from pal.errors import UserInputError
    with pytest.raises(UserInputError, match="saving was switched off"):
        store.zip_bytes(dry, "main")


def test_only_one_run_at_a_time_and_a_run_can_be_stopped(tmp_path):
    core = ScriptedCore(delay=0.15)
    store, _ = make_store(tmp_path, core)
    run = store.start(make_plan(), [], None)
    with pytest.raises(RunBusy, match="already running"):
        store.start(make_plan(), [], None)
    run.cancel.set()
    wait(run)
    assert run.status == "cancelled" and run.arms["main"].totals["stopped"] == "cancelled" and run.events[-1]["text"] == "Stopped"
    assert wait(store.start(make_plan(roles=["coder"]), [], None)).status == "done"                   # free again


def test_building_twice_runs_the_plain_client_then_pal_and_compares_them(tmp_path):
    store, _ = make_store(tmp_path)
    run = wait(store.start(make_plan(metrics=True), [], None))
    assert run.status == "done" and list(run.arms) == ["baseline", "pal"]
    c = run.comparison
    assert c and c["baseline"]["steps_done"] == c["pal"]["steps_done"] == 4 and c["pal"]["helper_tokens"] > 0 and c["baseline"]["helper_tokens"] == 0
    assert c["main_tokens_saved"] == c["baseline"]["main_tokens"] - c["pal"]["main_tokens"] and any("one run of each" in n for n in c["notes"])
    assert [e["kind"] for e in run.events if e["kind"] == "comparison"] and {e["arm"] for e in run.events if e["kind"] == "file"} == {"baseline", "pal"}
    for arm in ("baseline", "pal"):
        assert (tmp_path / "studio" / run.id / arm / "todo-app" / "index.html").is_file()


def test_both_builds_use_no_tools_when_comparing_so_the_comparison_is_fair(tmp_path):
    seen = []

    class Spy(ScriptedCore):
        def chat(self, syscall):
            seen.append(" ".join(m["content"] for m in syscall.messages))
            return super().chat(syscall)
    store, _ = make_store(tmp_path, Spy())
    wait(store.start(make_plan(metrics=True, tools=["calculator"]), [], None))
    assert not any("<tool>" in s and "reply with only" in s for s in seen)


def test_the_comparison_says_plainly_when_pal_did_not_save_and_what_that_usually_means():
    base = {"main_tokens": 1000, "prompt_tokens": 800, "helper_tokens": 0, "seconds": 10.0, "files": 4, "problems_left": 0}
    worse = compare(base, {**base, "main_tokens": 1100, "prompt_tokens": 900, "helper_tokens": 200, "seconds": 20.0, "files": 3, "problems_left": 1})
    text = " ".join(worse["notes"])
    assert worse["main_tokens_saved"] == -100 and worse["main_tokens_saved_pct"] == -10.0 and worse["seconds_saved_pct"] == -100.0 and worse["all_tokens_saved_pct"] == -30.0
    assert "as many tokens as the plain client or more" in text and "Pal was slower" in text and "The results differ" in text and "one run of each" in text
    better = compare(base, {**base, "main_tokens": 400, "prompt_tokens": 300, "helper_tokens": 50, "seconds": 8.0})
    assert better["main_tokens_saved_pct"] == 60.0 and better["all_tokens_saved_pct"] == 55.0 and len(better["notes"]) == 1


def test_a_failure_is_reported_in_the_run_and_the_key_never_appears_in_it(tmp_path):
    def factory(plan, key):
        raise PermissionError(f"401 invalid key {key}")
    store = RunStore(tmp_path / "studio", factory)
    run = wait(store.start(make_plan(), [], "sk-secret-key-123456789012345"))
    text = json.dumps(run.events)
    assert run.status == "error" and "PermissionError" in text and "sk-secret" not in text and "***" in text


def test_a_model_that_dies_mid_build_ends_the_run_as_an_error_but_keeps_the_files_so_far(tmp_path):
    store, _ = make_store(tmp_path, ScriptedCore(fail_on_step=3))
    run = wait(store.start(make_plan(), [], None))
    assert run.status == "error" and run.arms["main"].totals["steps_done"] == 2 and sorted(run.arms["main"].build.files) == ["app.js", "index.html", "style.css"]


def test_old_runs_are_deleted_beyond_the_limit_and_run_ids_are_validated(tmp_path):
    store, _ = make_store(tmp_path, max_runs=2)
    ids = [wait(store.start(make_plan(roles=["coder"]), [], None)).id for _ in range(4)]
    assert not (tmp_path / "studio" / ids[0]).exists() and (tmp_path / "studio" / ids[3]).exists()
    from pal.errors import NotFoundError
    for bad in ("../etc", "x", ids[0], "", None):
        with pytest.raises(NotFoundError):
            store.get(bad)


# -- the web endpoints -------------------------------------------------------------------------------------------------------------------------------
def test_the_catalogue_endpoint_gives_the_page_everything_it_needs_to_draw_the_controls(studio):
    c, _, _ = studio
    d = c.get("/api/studio/catalog").get_json()
    assert [r["id"] for r in d["roles"]][:2] == ["designer", "coder"] and d["local"]["recommended"] == "llama3.1:8b-instruct-q4_K_M" and d["local"]["ok"]
    assert {m["key"] for m in d["llm_modes"]} == {"explore", "own_key"} and "anthropic" in d["providers"] and d["limits"]["max_files"] == 60
    assert any(t["name"] == "calculator" and t["ready"] for t in d["tools"]) and any(p["id"] == "run_code" and p["locked"] for p in d["permissions"])


def test_asking_for_a_plan_returns_it_with_the_bots_explanation_and_the_model_used(studio):
    c, _, _ = studio
    r = c.post("/api/studio/plan", json={"command": "a to-do list web page"}).get_json()
    assert r["source"] == "model" and r["planner_model"] == "llama3.1:8b-instruct-q4_K_M" and r["plan"]["roles"][0] == "designer" and "**Todo app**" in r["message"] and r["local"]["ok"]
    assert r["plan"]["model"]["model"] == "llama3.1:8b-instruct-q4_K_M"


@pytest.mark.parametrize("payload,msg", [({}, "tell me what you want"), ({"command": "   "}, "tell me what you want"), ({"command": 5}, "tell me what you want"), ({"command": "x" * 5000}, "too long")])
def test_a_missing_or_silly_command_is_refused_with_a_reason(studio, payload, msg):
    c, _, _ = studio
    r = c.post("/api/studio/plan", json=payload)
    assert r.status_code == 400 and msg in r.get_json()["error"]
    assert c.post("/api/studio/plan", data="not json", content_type="text/plain").status_code == 400


def test_without_ollama_the_bot_still_makes_a_plan_by_rules_and_says_why(tmp_path):
    from pal.web.app import create_app
    from tests.fakes import FakeAPI
    from tests.helpers import make_service
    svc, _ = make_service(tmp_path, api=FakeAPI(["ok"]))
    c = create_app(service=svc, studio={"discover_fn": lambda url: local_info(ok=False)}).test_client()
    r = c.post("/api/studio/plan", json={"command": "a landing page"}).get_json()
    assert r["source"] == "fallback" and r["local"]["ok"] is False and "not reachable" in r["local"]["error"] and r["plan"]["model"]["model"] == ""


def test_the_person_edits_the_plan_by_hand_and_the_server_enforces_the_rules_on_it(studio):
    c, _, _ = studio
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    plan["permissions"]["network_access"] = True
    plan["permissions"]["run_code"] = True
    plan["roles"].append("wizard")
    off = c.post("/api/studio/validate", json={"plan": plan}).get_json()
    assert not off["plan"]["permissions"]["network_access"] and not off["plan"]["permissions"]["run_code"] and "wizard" not in off["plan"]["roles"] and off["notes"]
    on = c.post("/api/studio/validate", json={"plan": plan, "user_set": ["network_access"]}).get_json()
    assert on["plan"]["permissions"]["network_access"] and not on["plan"]["permissions"]["run_code"]            # only the risky one, and only because the person switched it on
    assert c.post("/api/studio/validate", json={"nope": 1}).status_code == 400


def test_telling_the_bot_to_change_the_plan_returns_the_new_plan_and_the_changes(studio):
    c, core, _ = studio
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    core.planner_json = planner_json(roles=["designer", "coder", "tester", "docs", "reviewer"])
    r = c.post("/api/studio/revise", json={"plan": plan, "instruction": "add a tester"}).get_json()
    assert "added agent: Tester" in r["changes"] and r["plan"]["roles"][2] == "tester" and r["message"].startswith("Done:")
    assert c.post("/api/studio/revise", json={"plan": plan, "instruction": ""}).status_code == 400


def full_run(c, plan_over=None, **extra):
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    plan.update(plan_over or {})
    return c.post("/api/studio/run", json={"plan": plan, **extra})


def test_starting_a_run_then_watching_it_through_polling_downloading_and_previewing_files(studio):
    c, _, runs = studio
    rid = full_run(c).get_json()["run_id"]
    wait(runs.get(rid))
    s = c.get(f"/api/studio/run/{rid}").get_json()
    assert s["status"] == "done" and s["arms"]["main"]["totals"]["files"] == 4 and s["events"][0]["kind"] == "run_start"
    later = c.get(f"/api/studio/run/{rid}?since={len(s['events']) - 2}").get_json()["events"]
    assert [e["id"] for e in later] == [len(s["events"]) - 2, len(s["events"]) - 1]
    files = c.get(f"/api/studio/run/{rid}/files?arm=main").get_json()["files"]
    assert [f["path"] for f in files] == ["README.md", "app.js", "index.html", "style.css"]
    f = c.get(f"/api/studio/run/{rid}/file?arm=main&path=APP.js").get_json()
    assert f["path"] == "app.js" and f["content"].startswith("document.getElementById") and not f["truncated"]
    z = c.get(f"/api/studio/run/{rid}/download?arm=main")
    assert z.mimetype == "application/zip" and "todo-app-main.zip" in z.headers["Content-Disposition"] and "todo-app/index.html" in zipfile.ZipFile(io.BytesIO(z.data)).namelist()


def test_the_file_and_run_endpoints_refuse_bad_ids_arms_and_paths(studio):
    c, _, runs = studio
    rid = full_run(c).get_json()["run_id"]
    wait(runs.get(rid))
    assert c.get("/api/studio/run/nothere").status_code == 404 and c.get("/api/studio/run/..%2f..%2fetc").status_code == 404
    assert c.get(f"/api/studio/run/{rid}/files?arm=evil").status_code == 404 and c.get(f"/api/studio/run/{rid}/download?arm=baseline").status_code == 404
    for bad in ("../x.html", "/etc/passwd", "C:\\a.html", "nothere.html", ""):
        assert c.get(f"/api/studio/run/{rid}/file?arm=main&path={bad}").status_code == 400
    assert c.get(f"/api/studio/run/{rid}?since=abc").status_code == 400


def test_a_second_run_while_one_is_running_is_a_conflict_and_cancel_works(studio):
    c, core, runs = studio
    core.delay = 0.15
    rid = full_run(c).get_json()["run_id"]
    r = full_run(c)
    assert r.status_code == 409 and "already running" in r.get_json()["error"]
    assert c.post(f"/api/studio/run/{rid}/cancel").get_json() == {"ok": True}
    wait(runs.get(rid))
    assert runs.get(rid).status == "cancelled"


def test_an_api_run_needs_a_key_and_the_key_is_used_but_never_kept(studio, monkeypatch):
    c, core, runs = studio
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    api = {"model": {"mode": "api", "provider": "anthropic", "model": "claude-x"}}
    r = full_run(c, api)
    assert r.status_code == 400 and "no API key for anthropic" in r.get_json()["error"] and "local model" in r.get_json()["error"]
    seen = {}
    runs.core_factory = lambda plan, key: seen.update(key=key) or core
    rid = full_run(c, api, api_key="sk-test-key-0000000000000000").get_json()["run_id"]
    wait(runs.get(rid))
    assert seen["key"] == "sk-test-key-0000000000000000"
    dump = json.dumps(c.get(f"/api/studio/run/{rid}").get_json()) + json.dumps(runs.get(rid).plan)
    assert "sk-test-key" not in dump


def test_reference_files_are_cleaned_limited_and_only_used_when_allowed(studio):
    c, core, runs = studio
    refs = [{"name": "..\\..\\evil name?.md", "text": "# Brief"}]
    rid = full_run(c, references=refs).get_json()["run_id"]
    run = wait(runs.get(rid))
    assert run.status == "done"
    for bad, msg in (([{"name": "a.md", "text": "x" * 200_000}], "too big"), ([{"name": "a"}], "name and text"), ("nope", "at most"), ([{"name": f"{i}.md", "text": "x"} for i in range(9)], "at most")):
        r = full_run(c, references=bad)
        assert r.status_code == 400 and msg in r.get_json()["error"], bad
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    plan["permissions"]["read_references"] = False
    assert c.post("/api/studio/run", json={"plan": plan, "references": [{"name": "a", "text": "b" * 200_000}]}).status_code == 200      # not even read when switched off


def test_the_server_rebuilds_the_plan_before_running_it_so_a_tampered_plan_cannot_unlock_anything(studio):
    c, _, runs = studio
    plan = c.post("/api/studio/plan", json={"command": "x"}).get_json()["plan"]
    plan["permissions"].update(network_access=True, run_code=True, delete_files=True, outside_project=True)
    plan["tools"] = ["wikipedia", "calculator"]
    rid = c.post("/api/studio/run", json={"plan": plan}).get_json()["run_id"]
    run = wait(runs.get(rid))
    assert not any(run.plan["permissions"][p] for p in ("network_access", "run_code", "delete_files", "outside_project")) and run.plan["tools"] == ["calculator"]


# -- the project check at the end ------------------------------------------------------------------------------------------------------------------
def test_a_page_may_link_to_a_file_a_later_step_writes_so_nothing_is_flagged_in_the_middle(tmp_path):
    replies = [files_reply(GOOD["Designer"]), files_reply(GOOD["Coder"]), files_reply(GOOD["Docs writer"]), files_reply(GOOD["Reviewer"])]
    b, totals, events, client = run_build(make_plan(), replies, tmp=tmp_path)
    assert "refers to app.js" not in " ".join(t for k, t, _ in events) and totals["calls"] == 4 and totals["problems_left"] == 0
    assert [t for k, t, _ in events if k == "check"][-1] == "project check: the files fit together"


def test_a_file_that_was_never_written_gets_one_fix_at_the_end_and_is_reported_if_it_is_still_missing(tmp_path):
    plan = make_plan(roles=["coder"])
    page_only = files_reply([("index.html", INDEX)])                       # links to style.css and app.js, which nobody wrote
    fixed, totals, events, client = run_build(plan, [page_only, files_reply([("style.css", "a { }"), ("app.js", "let a = 1;")])], tmp=tmp_path / "a")
    assert "finished project" in client.asked[1][0] and "style.css" in client.asked[1][0] and sorted(fixed.files) == ["app.js", "index.html", "style.css"]
    assert totals["problems_left"] == 0 and totals["calls"] == 2
    still, totals2, events2, _ = run_build(plan, [page_only, files_reply([("other.txt", "x")])], tmp=tmp_path / "b")
    last = [t for k, t, _ in events2 if k == "check"][-1]
    assert totals2["problems_left"] == 2 and "2 problem(s) left" in last and "app.js" in last and totals2["calls"] == 2


def test_no_final_fix_is_asked_for_after_a_cancel_or_an_error(tmp_path):
    plan = make_plan(roles=["designer"])
    b, totals, events, client = run_build(plan, [files_reply(GOOD["Designer"])], tmp=tmp_path, cancel=lambda: True)
    assert totals["stopped"] == "cancelled" and client.asked == []


def test_a_path_that_tries_to_leave_the_project_is_recorded_as_blocked_with_the_reason_not_silently_skipped():
    reply = "### FILE: a/../../x.html\n```html\n1\n```\n### FILE: C:\\Windows\\a.html\n```html\n2\n```\n### FILE: ..\\y.css\n```css\n3\n```"
    b, totals, events, _ = run_build(make_plan(roles=["coder"]), [reply], perms={"write_project": True})
    assert totals["blocked"] == 3 and b.files == {} and sorted(d["reason"][:12] for k, _, d in events if k == "blocked") == ["'..' is not ", "'..' is not ", "absolute pat"]
