"""The Studio page, driven like a person would use it (type, click, toggle, build, switch tabs) in jsdom against the real server and the real script.

Needs Node and jsdom: `cd tests/studio/ui && npm install`. Skipped when they are not there. A real browser was not available while writing this;
jsdom is a faithful DOM and script engine, but it does not lay anything out or enforce the content-security policy (that is checked separately below).
"""
import json
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path

import pytest
from werkzeug.serving import make_server

from pal.web.app import create_app
from tests.fakes import FakeAPI
from tests.helpers import make_service
from tests.studio.conftest import ScriptedCore, local_info, planner_json

UI = Path(__file__).parent / "ui"
needs_node = pytest.mark.skipif(not shutil.which("node") or not (UI / "node_modules" / "jsdom").exists(), reason="needs node and jsdom (cd tests/studio/ui && npm install)")


@contextmanager
def serve(core, tmp_path, local=None, planned=None):
    svc, _ = make_service(tmp_path, api=FakeAPI(["ok"]))
    app = create_app(service=svc, studio={"planner_core": lambda model: (planned.append(model) if planned is not None else None, core)[1], "core_factory": lambda plan, key: core, "discover_fn": lambda url: local or local_info()})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", app
    finally:
        server.shutdown()


def drive(base, scenario, timeout=90):
    done = subprocess.run(["node", "drive.js", base, scenario], cwd=UI, capture_output=True, text=True, timeout=timeout)
    line = (done.stdout.strip().splitlines() or ["{}"])[-1]
    data = json.loads(line)
    assert done.returncode == 0 and "error" not in data, f"{scenario}: {data} {done.stderr[-500:]}"
    return data


# -- the page itself --------------------------------------------------------------------------------------------------------------------------------
class _Audit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.inline, self.handlers, self.styles = [], 0, [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script":
            self.scripts.append(a.get("src"))
        self.handlers += [k for k in a if k.startswith("on")]
        self.styles += "style" in a


def test_the_page_carries_a_strict_content_security_policy_and_has_no_inline_script_style_or_handlers(tmp_path):
    svc, _ = make_service(tmp_path, api=FakeAPI(["ok"]))
    c = create_app(service=svc, studio={"discover_fn": lambda url: local_info()}).test_client()
    r = c.get("/studio")
    csp = r.headers["Content-Security-Policy"]
    assert "font-src 'self'" in csp
    assert r.status_code == 200 and "script-src 'self'" in csp and "unsafe-inline" not in csp and "default-src 'none'" in csp and "connect-src 'self'" in csp and "frame-ancestors 'none'" in csp
    audit = _Audit()
    audit.feed(r.get_data(as_text=True))
    assert audit.scripts == ["/static/voice.js", "/static/studio.js", "/static/editor.js"] and audit.handlers == [] and audit.styles == 0
    js, css = c.get("/static/studio.js"), c.get("/static/studio.css")
    assert js.status_code == css.status_code == 200 and b"textContent" in js.data and b"innerHTML" not in js.data and b"insertAdjacentHTML" not in js.data and b"eval(" not in js.data
    assert r.headers["X-Frame-Options"] == "DENY" and r.headers["Cache-Control"] == "no-store"


def test_the_script_never_builds_html_from_text_a_model_wrote():
    for name in ("studio.js", "editor.js"):
        js = (Path(__file__).parents[2] / "src/pal/web/static" / name).read_text(encoding="utf-8")
        assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|new Function|\beval\(", js), name


# -- the page in use ----------------------------------------------------------------------------------------------------------------------------------
@needs_node
def test_describing_a_product_shows_the_bots_plan_then_asks_one_question_at_a_time_with_the_choices_and_reasons(tmp_path, monkeypatch):
    monkeypatch.setattr("pal.terminal.voice.engines", lambda: [])       # no speech-to-text here: the mic button is off
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, _):
        f = drive(base, "plan")
    assert f["empty"] == {"visible": True, "suggestions": 4, "micDisabled": True}
    assert f["chipClass"] == "chip ok" and "llama3.1:8b-instruct-q4_K_M" in f["chip"] and "best fit" in f["chip"]
    assert f["head"] == "Question 1 of 8" and f["name"] == "Todo app" and f["kind"] == "web_app" and f["openCards"] == 1
    assert f["roles"] == ["designer", "coder", "docs", "reviewer"] and f["coderDisabled"] and f["tools"] == ["calculator"]
    assert f["perms"]["write_project"] == {"on": True, "disabled": False, "risky": False} and f["perms"]["network_access"] == {"on": False, "disabled": False, "risky": True}
    assert f["perms"]["run_code"] == {"on": False, "disabled": True, "risky": False} and f["perms"]["delete_files"]["disabled"] and f["riskyBadges"] == ["only you can switch this on"]
    assert f["modelMode"] == "local" and f["localValue"] == "llama3.1:8b-instruct-q4_K_M" and not any("nomic" in o for o in f["localOptions"]) and any("recommended: the strongest" in o for o in f["localOptions"])
    assert [s[0] for s in f["steps"]] == ["designer", "coder", "docs", "reviewer"] and "wikipedia" in f["toolsDisabled"] and f["userSet"] == []
    assert f["answeredLines"] == 8 and [r[0] for r in f["confirmRows"]] == ["Product", "Agents", "Access", "Tools", "Model", "Steps", "Files", "Compare"]
    bots = [m for m in f["chat"] if m[0] == "bot"]
    assert f["chat"][0] == ["user", "a to-do list web page", []] and "**Todo app**" in bots[0][1] and "does not go online" in bots[0][1] and "8 short questions" in bots[1][1]


@needs_node
def test_only_one_question_is_open_at_a_time_and_back_asks_the_earlier_one_again(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, _):
        r = drive(base, "onebyone")
    assert r["counts"] == [1, 1] and r["heads"] == ["Question 2 of 8"] and r["afterBack"] == 1 and r["answered"] == 0


@needs_node
def test_setting_everything_up_by_hand_works_without_the_bot(tmp_path):
    with serve(ScriptedCore(), tmp_path) as (base, _):
        f = drive(base, "hand")
    assert f["name"] == "My project" and f["roles"] == ["coder"] and (f["plan"]["model"]["mode"], f["plan"]["model"]["model"]) == ("local", "llama3.1:8b-instruct-q4_K_M") and f["plan"]["permissions"]["network_access"] is False
    assert "setting this up by hand" in f["chat"][0][1] and f["source"] == ""


@needs_node
def test_editing_by_hand_the_server_keeps_the_rules_and_only_the_person_can_switch_a_risky_permission_on(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, _):
        r = drive(base, "edit")
    assert "docs" not in r["afterDocs"]["roles"] and r["afterDocs"]["steps"] == ["designer", "coder", "reviewer"]          # the steps followed the agents
    assert r["afterDocs"]["perm"] is True and r["afterDocs"]["userSet"] == ["network_access"] and r["netOff"] == {"userSet": [], "perm": False}
    assert "tester" in r["withTester"] and r["added"] == r["before"] + 1 and r["afterDelete"] == r["before"] and r["stepsNow"] == r["before"] and r["lastTask"]


@needs_node
def test_typing_a_change_updates_the_plan_lists_what_changed_and_asks_the_same_question_again(tmp_path):
    revised = planner_json(roles=["designer", "coder", "tester", "docs", "reviewer"])
    with serve(ScriptedCore(planner_json=[planner_json(), revised]), tmp_path) as (base, _):
        f = drive(base, "revise")
    assert "tester" in f["roles"] and f["last"][1].startswith("Done:") and "added agent: Tester" in f["last"][2] and f["cards"] == 1


@needs_node
def test_pressing_build_shows_every_step_as_it_happens_then_the_files_a_preview_and_a_download(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, app):
        r = drive(base, "build")
        run = next(iter(app.extensions["studio_runs"]._runs.values()))
    assert r["status"].startswith("Todo app: Finished") and r["stopHidden"] and not r["buildDisabled"] and r["tabs"] == [] and r["sidebar"] == ["Todo app"] and r["dot"] == "dot done"
    text = " | ".join(e[1] for e in r["events"])
    assert "Step 1 of 4: Designer" in text and "Step 4 of 4: Reviewer" in text and "wrote style.css" in text and "the files fit together" in text
    assert [e[0] for e in r["events"]][0] == "ev ev-run_start" and "ev ev-file" in [e[0] for e in r["events"]] and "Step 4 of 4" in r["counters"] and "Files 4" in r["counters"]
    assert r["files"] == ["README.md", "app.js", "index.html", "style.css"] and r["preview"].startswith("document.getElementById")
    assert r["download"] == f"/api/studio/run/{run.id}/download?arm=main" and not r["downloadHidden"]


@needs_node
def test_building_twice_shows_both_builds_and_the_difference_in_a_table(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, app):
        r = drive(base, "compare")
    assert r["tabs"] == ["Plain chat client", "Pal", "Compare"] and r["activeTab"] == "Compare"
    assert r["headline"].startswith("Without Pal: ") and ("Saved " in r["headline"] or "more tokens" in r["headline"])
    rows = {row[0]: row for row in r["rows"]}
    helper = "Helper tokens (its own small jobs)"
    assert set(rows) >= {"Main-model tokens", "Tokens read (prompts)", helper, "All tokens, helper included", "Time (seconds)", "Files written", "Problems left in the files"}
    assert rows["Files written"][1:3] == ["4", "4"] and rows[helper][1] == "0" and int(rows[helper][2].replace(",", "")) > 0
    assert rows["Main-model tokens"][3].endswith("saved") or rows["Main-model tokens"][3].endswith("more")
    assert any("one run of each" in n for n in r["notes"]) and r["plainEvents"] > 5 and r["plainFiles"] == ["README.md", "app.js", "index.html", "style.css"] and r["plainDownload"].endswith("arm=baseline")


@needs_node
def test_choosing_an_api_model_without_a_key_asks_for_one_and_offers_the_local_model_instead(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, _):
        r = drive(base, "key")
    assert r["modes"] == ["explore", "own_key"] and "anthropic" in r["provs"][0] and "There is no API key for anthropic" in r["text"]
    assert r["useLocalVisible"] and r["stillOnModel"] and not r["runStarted"] and r["afterLocal"] and r["warnHidden"] and r["apiBoxHidden"]


@needs_node
def test_a_build_can_be_stopped_from_the_page_and_the_plan_is_offered_again(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json(), delay=0.4), tmp_path) as (base, _):
        r = drive(base, "stop")
    assert r["status"].startswith("Todo app: Stopped") and not r["buildDisabled"]


@needs_node
def test_reference_files_are_read_listed_limited_and_removable(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, _):
        r = drive(base, "references")
    assert r["names"] == ["brief.md"] and "huge.txt is too big" in r["msg"] and r["filesAnswer"] == 1 and r["afterRemove"] == 0 and r["chips"]


@needs_node
def test_text_a_model_wrote_is_shown_as_text_never_run_or_turned_into_page_elements(tmp_path):
    evil = '<img src="data:," onerror="window.__pwned=1">'
    core = ScriptedCore(planner_json=planner_json(roles=["coder"], product={"name": evil, "summary": "<script>window.__pwned=1</script>", "kind": "web_app"}),
                        files={"Coder": [("page.html", f"<html><body>{evil}<script>window.__pwned = 1</script></body></html>")]})
    with serve(core, tmp_path) as (base, _):
        r = drive(base, "hostile")
    assert r["name"] == evil and r["pwned"] is None                                                  # shown in the field as text, and nothing ran
    ours = '<img class="av" src="/static/pal-logo-64.png" alt="">'                                   # the logo avatar is the only image allowed
    assert "<img" not in r["threadHtml"].replace(ours, "").replace("&lt;img", "") and "&lt;img" in r["threadHtml"] and r["preview"].startswith('<html><body><img src="data:," onerror=')
    assert r["injectedNodes"] == 0 and r["scripts"] == 3


@needs_node
def test_a_slow_answer_for_one_build_cannot_overwrite_the_list_of_the_build_the_person_clicked_next(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, _):
        r = drive(base, "race")
    assert r["download"].endswith("arm=baseline") and r["files"] == ["README.md", "app.js", "index.html", "style.css"]


@needs_node
def test_while_the_bot_is_working_the_page_says_so_counts_the_seconds_and_the_answer_replaces_the_message(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json(), delay=1.4), tmp_path) as (base, _):
        r = drive(base, "thinking")
    assert r["during"]["text"].startswith("Thinking... ") and r["during"]["send"] and r["during"]["botmodel"] and r["during"]["stopVisible"]
    assert r["after"] == {"thinking": 0, "send": True, "stop": False}


@needs_node
def test_stopping_the_wait_frees_the_page_and_a_late_answer_is_ignored(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json(), delay=2.0), tmp_path) as (base, _):
        r = drive(base, "stopwait")
    assert r["cards"] == 0 and not r["planSet"] and r["thinking"] == 0 and any(m.startswith("Stopped waiting") and "smaller bot model" in m for m in r["chat"])


@needs_node
def test_the_bot_model_can_be_chosen_and_the_strongest_is_the_default_and_embedding_models_are_not_offered(tmp_path):
    planned = []
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path, planned=planned) as (base, _):
        r = drive(base, "botmodel")
    assert r["first"] == "llama3.1:8b-instruct-q4_K_M" and r["chosen"] == "llama3.2:1b" and planned == ["llama3.2:1b"]
    assert [o[0] for o in r["opts"]] == ["llama3.2:1b", "llama3.1:8b-instruct-q4_K_M"] and "(strongest, slowest)" in dict(r["opts"])["llama3.1:8b-instruct-q4_K_M"]


@needs_node
def test_the_microphone_types_what_it_hears_and_is_switched_off_where_the_browser_cannot_listen(tmp_path, monkeypatch):
    monkeypatch.setattr("pal.terminal.voice.engines", lambda: [])       # the browser's own speech recognition (simulated) does the listening
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, _):
        on, off = drive(base, "mic"), drive(base, "micoff")
    assert on["before"] == {"disabled": False, "pressed": "false", "sendDisabled": True}
    assert on["during"] == {"pressed": "true", "live": True, "text": "build me a hotel page", "sendDisabled": False} and on["after"] == {"pressed": "false", "live": False}
    assert off["disabled"] and "not available" in off["title"]


@needs_node
def test_new_build_clears_the_conversation_and_starts_over(tmp_path):
    with serve(ScriptedCore(planner_json=planner_json()), tmp_path) as (base, _):
        r = drive(base, "newbuild")
    assert r == {"emptyVisible": True, "threadHidden": True, "threadChildren": 0, "plan": None}
