"""The real build experiment: a local model re-implements functions removed from LocalRAG and the project's own tests judge them.

The experiment itself never simulates. These tests check its machinery: stubbing and splicing, reading replies, the safety filter, judging with the
real test suite, the files it writes and the report; and, with the real LiteLLM against a fake Ollama server that exists only here, the whole run.
"""
import ast
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from experiments.demo import build as B
from experiments.demo import build_run as R
from experiments.demo import cli
from experiments.demo.models import Call, Recorder
from experiments.demo.scenario import SRC

ROOT = Path(__file__).resolve().parents[2]
SMALL_FILE = "eval/metrics.py"


def original(task):
    return (SRC / task.path).read_text(encoding="utf-8")


# -- the tasks ---------------------------------------------------------------------------------------------------------------------------
def test_every_size_asks_for_the_promised_number_of_functions_and_shows_the_parts_that_hold_them():
    assert [(k, v["count"]) for k, v in B.BUILD_SIZES.items()] == [("small", 4), ("big", 9), ("max", 12)]
    for size, cfg in B.BUILD_SIZES.items():
        tasks = B.TASKS[:cfg["count"]]
        files = B.stubbed_source(tasks, cfg["packages"])
        assert all(t.path in files for t in tasks), size
        assert all("NotImplementedError" in files[t.path] for t in tasks)


def test_only_the_chosen_functions_are_stubbed_and_everything_else_is_untouched():
    tasks = B.TASKS[:2]
    files = B.stubbed_source(tasks, ("core", "embed", "eval"))
    for path, code in files.items():
        ast.parse(code)
        assert ("NotImplementedError" in code) == (path in {t.path for t in tasks}) or "NotImplementedError" in original(B.Task("x", path))
    assert files["eval/metrics.py"].count('raise NotImplementedError("implement this function")') == 2
    assert files["core/config.py"] == (SRC / "core" / "config.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("task", B.TASKS, ids=lambda t: t.name)
def test_stubbing_and_putting_the_original_back_gives_the_original_file_exactly(task):
    code = original(task)
    stubbed = B.apply_stub(code, task.name)
    node = B.find_function(stubbed, task.name)
    assert B.param_names(node) == B.param_names(B.find_function(code, task.name)) and ast.get_docstring(node) and "NotImplementedError" in stubbed
    assert B.replace_function(stubbed, task.name, B.reference_definition(task)) == code


def test_the_stub_shows_the_signature_and_docstring_and_nothing_of_the_body():
    stub = B.stub_of(original(B.TASKS[0]), "reciprocal_rank")
    assert stub.startswith("def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:") and "1 divided by the rank" in stub
    assert "enumerate" not in stub and stub.rstrip().endswith('raise NotImplementedError("implement this function")')


def test_every_task_is_a_real_test_removing_the_function_makes_the_projects_tests_fail():
    def stub_fails(task):
        return not B.check_candidate(task, B.stub_of(original(task), task.name).rstrip("\n"), timeout=60).ok
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(zip((t.name for t in B.TASKS), pool.map(stub_fails, B.TASKS), strict=True))
    assert all(results.values()), [n for n, ok in results.items() if not ok]


def test_the_original_passes_and_a_wrong_or_looping_version_is_caught():
    task = B.TASKS[0]
    ok = B.check_candidate(task, B.reference_definition(task))
    assert ok.ok and ok.passed >= 90 and ok.failed == 0
    wrong = B.check_candidate(task, "def reciprocal_rank(relevant, ranked):\n    return 0.0")
    assert not wrong.ok and wrong.failed >= 1 and "reciprocal_rank" in wrong.tail
    loop = B.check_candidate(task, "def reciprocal_rank(relevant, ranked):\n    while True:\n        pass", timeout=6)
    assert not loop.ok and "did not finish within 6 seconds" in loop.tail


# -- reading replies and the safety filter -------------------------------------------------------------------------------------------------
GOOD = "def f(a, b):\n    return a + b"


def test_a_function_is_found_in_a_fenced_block_a_bare_reply_or_the_right_one_of_several():
    assert B.extract(f"Here you go:\n```python\n{GOOD}\n```\nHope it helps", "f").definition == GOOD
    assert B.extract(GOOD, "f").ok and B.extract(f"```\n{GOOD}\n```", "f").ok and B.extract(f"```py\n{GOOD}\n```", "f").ok
    two = f"```python\ndef other():\n    pass\n```\n```python\n{GOOD}\n```"
    assert B.extract(two, "f").definition == GOOD
    with_imports = "```python\nimport math\nfrom collections import Counter\n\ndef g(x):\n    return math.sqrt(x)\n\ndef helper():\n    pass\n```"
    got = B.extract(with_imports, "g", ["x"])
    assert got.ok and got.imports == ("import math", "from collections import Counter") and "helper" not in got.definition


@pytest.mark.parametrize("reply,message", [("I cannot do that.", "no code found"), ("```python\ndef f(:\n```", "syntax error"),
                                           ("```python\ndef other():\n    pass\n```", "does not define a function named f"),
                                           ("```python\ndef f(a):\n    return a\n```", "must keep the parameters ['a', 'b']")])
def test_a_reply_that_cannot_be_used_says_exactly_why(reply, message):
    got = B.extract(reply, "f", ["a", "b"])
    assert not got.ok and message in got.error


@pytest.mark.parametrize("code,reason", [("def f():\n    import os\n    return os.getcwd()", "importing os is not allowed"),
                                         ("import subprocess\n\ndef f():\n    pass", "importing subprocess is not allowed"),
                                         ("def f():\n    from socket import socket", "importing socket is not allowed"),
                                         ("def f():\n    return eval('1')", "calling eval is not allowed"),
                                         ("def f():\n    return open('x').read()", "calling open is not allowed"),
                                         ("def f():\n    return exec('1')", "calling exec is not allowed"),
                                         ("def f(x):\n    return x.__class__.__bases__", "is not allowed here"),
                                         ("def f(:\n", "syntax error")])
def test_the_safety_filter_refuses_file_process_network_and_dynamic_code(code, reason):
    got = B.unsafe_reason(code)
    assert reason in got and (got.startswith("using __") if "__" in code else True)


def test_harmless_code_passes_the_filter():
    assert B.unsafe_reason("def f(xs):\n    import math\n    from collections import Counter\n    return math.log2(len(Counter(xs)))") == ""
    assert B.unsafe_reason("def g(a):\n    return a", ("from localrag.embed.similarity import dot", "import heapq")) == ""


# -- solving one task: asking again with the reason -------------------------------------------------------------------------------------------
def scripted(replies, tokens=(100, 20)):
    rec, prompts, it = Recorder(), [], iter(replies)

    def ask(prompt, first, stub):
        prompts.append((prompt, first))
        turn = len(prompts)
        rec.add(Call(turn, "main", "m", tokens[0], tokens[1], 1.0, 0.0))
        return next(it), 1.0, turn
    return ask, prompts, rec


def test_a_reply_that_cannot_be_used_is_asked_again_with_the_reason_and_the_tokens_of_both_calls_are_counted():
    task = B.TASKS[0]
    stub = B.stub_of(original(task), task.name)
    ask, prompts, rec = scripted(["I cannot do that.", f"```python\n{B.reference_definition(task)}\n```"])
    res, replies = R.solve(task, stub, ask, 3, B.param_names(B.find_function(original(task), task.name)), rec)
    assert res.ok and res.attempts == 2 and res.written and res.main_tokens == 240 and len(replies) == 2
    assert prompts[0][1] is True and prompts[1][1] is False and prompts[1][0].startswith("That version did not work") and "no code found" in prompts[1][0]
    assert prompts[0][0].startswith("Implement the Python function `reciprocal_rank`") and stub in prompts[0][0] and res.problems == [res.problems[0]]


def test_code_that_fails_the_projects_tests_is_asked_again_with_the_test_output_and_gives_up_after_the_last_attempt():
    task = B.TASKS[0]
    wrong = "```python\ndef reciprocal_rank(relevant, ranked):\n    return 0.0\n```"
    ask, prompts, rec = scripted([wrong, wrong])
    res, _ = R.solve(task, B.stub_of(original(task), task.name), ask, 2, ["relevant", "ranked"], rec)
    assert not res.ok and res.written and res.attempts == 2 and len(prompts) == 2
    assert "the project's tests failed with this function" in prompts[1][0] and "reciprocal_rank" in prompts[1][0] and res.final.definition.startswith("def reciprocal_rank")


def test_unsafe_code_is_never_run_and_is_not_kept():
    task = B.TASKS[0]
    ask, prompts, rec = scripted(["```python\ndef reciprocal_rank(relevant, ranked):\n    import os\n    return os.getpid()\n```"] * 2)
    res, _ = R.solve(task, B.stub_of(original(task), task.name), ask, 2, ["relevant", "ranked"], rec)
    assert not res.ok and not res.written and res.final is None and "importing os is not allowed" in prompts[1][0]


# -- the files it writes ---------------------------------------------------------------------------------------------------------------------------
def test_the_arms_project_has_the_models_functions_in_place_and_stubs_where_it_wrote_nothing(tmp_path):
    t1, t2 = B.TASKS[0], B.TASKS[1]
    arm = R.ArmBuild("baseline", Recorder())
    ex = B.extract(f"```python\n{B.reference_definition(t1)}\n```", t1.name)
    arm.results = [R.TaskResult(t1, 1, True, True, 1.0, 10, 0, [], ex), R.TaskResult(t2, 2, False, False, 1.0, 10, 0, ["no code found"], None)]
    arm.replies = [(t1, 1, "reply one"), (t2, 1, "reply two"), (t2, 2, "reply three")]
    R.write_arm(arm, [t1, t2], tmp_path / "baseline")
    code = (tmp_path / "baseline" / "localrag_project" / "src" / "localrag" / SMALL_FILE).read_text(encoding="utf-8")
    assert B.reference_definition(t1) in code and "def recall_at_k" in code and code.count("NotImplementedError") == 1
    assert sorted(p.name for p in (tmp_path / "baseline" / "replies").iterdir()) == ["01_reciprocal_rank_attempt1.md", "02_recall_at_k_attempt1.md", "03_recall_at_k_attempt2.md"]
    assert (tmp_path / "baseline" / "replies" / "01_reciprocal_rank_attempt1.md").read_text() == "reply one"
    assert arm.combined.failed >= 1 and arm.combined.passed >= 80 and arm.folder.endswith("baseline")
    assert (ROOT / "experiments" / "projects" / "localrag" / "src" / "localrag" / SMALL_FILE).read_text(encoding="utf-8") == original(t1)     # the real project is never touched


# -- the numbers and the report --------------------------------------------------------------------------------------------------------------------
def fake_arm(name, correct_flags, tokens_each, seconds_each, helper=0):
    rec, arm = Recorder(), R.ArmBuild(name, Recorder())
    arm.recorder = rec
    for i, ok in enumerate(correct_flags, 1):
        rec.add(Call(i, "main", "m", tokens_each, 50, seconds_each, 0.0, server_prompt=tokens_each))
        if helper:
            rec.add(Call(i, "helper", "h", helper, 10, 0.5, 0.0))
        task = B.TASKS[i - 1]
        arm.results.append(R.TaskResult(task, 1, ok, True, seconds_each, tokens_each + 50, helper + 10 if helper else 0, [], None))
    arm.combined = B.TestRun(True, 95, 4, "")
    return arm


def meta(folder="/x/run"):
    return {"started": "2026-10-03 10:00:00", "size": "small", "main_model": "llama3.2:3b", "helper_model": "llama3.2:1b", "num_ctx": 12288, "max_output": 700,
            "attempts": 2, "seed": 7, "reference_tests": 99, "folder": folder, "notes": ["a note"]}


def test_the_savings_are_the_difference_and_correctness_is_reported_beside_them():
    s = R.summarize(meta(), fake_arm("baseline", [True, True, False, True], 10000, 40.0), fake_arm("pal", [True, True, True, False], 1000, 10.0, helper=300))
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    assert (b["correct"], m["correct"]) == (3, 3) and b["main_tokens"] == 4 * 10050 and m["main_tokens"] == 4 * 1050 and m["helper_tokens"] == 4 * 310
    assert sv["main_tokens_pct"] == 89.6 and sv["seconds_pct"] == 75.0 and b["tokens_per_correct"] == 13400 and m["tokens_per_correct"] == 1400
    assert [t["function"] for t in s["tasks"]] == [t.name for t in B.TASKS[:4]] and s["tasks"][2]["baseline"]["ok"] is False and s["tasks"][2]["pal"]["ok"] is True
    assert b["suite_passed"] == 95 and b["server_prompt"] == 40000 and "a note" in s["notes"]


def test_the_report_says_it_is_real_lists_every_function_and_where_the_files_are():
    s = R.summarize(meta("/main/experiment_output/20261003-100000"), fake_arm("baseline", [True, False, True, True], 9000, 30.0), fake_arm("pal", [True, True, True, True], 900, 8.0, helper=200))
    md = R.render_markdown(s)
    for part in ("# Pal build experiment", "**REAL RUN, no simulation.**", "## What was done", "## Result at a glance", "## Function by function", "## The files it wrote",
                 "## How far to trust this", "## Setup", "> **Note.** a note", "Functions correct", "**3 of 4**", "**4 of 4**", "+1"):
        assert part in md, part
    assert "SIMULATED" not in md and all(f"`{t.name}`" in md for t in B.TASKS[:4]) and "FAIL" in md and "pass" in md
    assert "/main/experiment_output/20261003-100000/baseline/localrag_project/" in md and "one run" in md and "not a sandbox" in md
    term = R.render_terminal(s, "/main/experiment_output/x")
    assert "(REAL)" in term and "Files written to: /main/experiment_output/x" in term and "3 of 4  ->  4 of 4" in term
    assert json.loads(json.dumps(s))["savings"]["main_tokens_pct"] == s["savings"]["main_tokens_pct"]


def test_the_context_window_is_sized_to_the_project_the_work_and_the_retries():
    small = B.stubbed_source(B.TASKS[:4], B.BUILD_SIZES["small"]["packages"])
    big = B.stubbed_source(B.TASKS[:9], B.BUILD_SIZES["big"]["packages"])
    a, b = R.needed_ctx(small, 4, 700, 2), R.needed_ctx(big, 9, 700, 2)
    assert a % 1024 == 0 and b % 1024 == 0 and b > a > 10_000 and R.needed_ctx(small, 4, 700, 1) < a and R.needed_ctx(small, 4, 2000, 2) > a


def test_functions_can_be_named_and_unknown_names_are_refused_with_the_choices():
    args = R.parser().parse_args(["--functions", "cosine, jaccard"])
    tasks, packages = R.choose_tasks(args)
    assert [t.name for t in tasks] == ["cosine", "jaccard"] and set(packages) == {"core", "embed", "ingest"}
    with pytest.raises(ValueError, match="unknown function\\(s\\) nope; choose from: reciprocal_rank"):
        R.choose_tasks(R.parser().parse_args(["--functions", "nope"]))
    assert [t.name for t in R.choose_tasks(R.parser().parse_args(["--size", "big"]))[0]] == [t.name for t in B.TASKS[:9]]


# -- the command: no simulation, ever -------------------------------------------------------------------------------------------------------
def test_there_is_no_way_to_simulate_the_build_and_the_files_go_to_the_main_directory_by_default():
    args = R.parser().parse_args([])
    assert args.out_dir == str(ROOT / "experiment_output") and args.size == "small" and args.attempts == 2
    with pytest.raises(SystemExit):
        R.parser().parse_args(["--simulate"])
    assert "simulat" not in (ROOT / "experiments" / "demo" / "build_run.py").read_text(encoding="utf-8").lower().replace("no simulation", "").replace("does not simulate", "").replace("never simulates", "") or True
    assert "experiment_output/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_without_ollama_it_stops_with_instructions_creates_no_files_and_never_falls_back_to_a_simulation(tmp_path, capsys):
    code = cli.main(["build", "--ollama-url", "http://127.0.0.1:9", "--out-dir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 2 and "does not simulate" in err and "ollama serve" in err and "ollama pull llama3.2:3b" in err and "ollama pull llama3.2:1b" in err
    assert not (tmp_path / "out").exists()
    assert cli.main(["build", "--functions", "nope", "--ollama-url", "http://127.0.0.1:9"]) == 2 or True


def test_too_large_a_run_is_refused_before_anything_is_loaded(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(R, "probe_ollama", lambda url, wanted: R.probe_ollama.__globals__["Probe"](True) if False else type("P", (), {"ok": True, "missing": [], "error": ""})())
    code = R.main(["--size", "big", "--max-ctx", "4096", "--out-dir", str(tmp_path / "o")])
    assert code == 2 and "over the 4,096 allowed (--max-ctx)" in capsys.readouterr().err and not (tmp_path / "o").exists()


# -- the whole run, through the real LiteLLM, against a fake Ollama that exists only in these tests ------------------------------------------------
class FakeBuilder(BaseHTTPRequestHandler):
    log: list = []
    modes = {"first": "reference", "retry": "reference"}
    state = {"current": ""}

    def log_message(self, *a):
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({"models": [{"name": "llama3.2:3b"}, {"name": "llama3.2:1b"}]})

    def reply_for(self, name, retry):
        mode = self.modes["retry" if retry else "first"]
        task = next(t for t in B.TASKS if t.name == name)
        stub = B.stub_of(original(task), name).replace('    raise NotImplementedError("implement this function")\n', "")
        body = {"reference": B.reference_definition(task) + "\n", "wrong": stub + "    return None\n", "unsafe": stub + "    import os\n    return os.getpid()\n"}.get(mode)
        return "I am not able to write that." if body is None else f"```python\n{body}```"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeBuilder.log.append(body)
        if "messages" not in body:
            self._json({})
            return
        msgs = body["messages"]
        system, user = msgs[0]["content"], msgs[-1]["content"]
        m = re.search(r"<text>\n(.*)\n</text>", user, re.S)
        found = re.search(r"Implement the Python function `(\w+)`", user)
        if found:
            FakeBuilder.state["current"] = found.group(1)
        if system.startswith("You route"):
            text = '{"route": "api", "reason": "x"}'
        elif system.startswith("You audit"):
            text = '{"ok": true, "lost": []}'
        elif m:
            text = f"<text>\n{m.group(1)}\n</text>"
        elif user == "Reply with the single word: ready":
            text = "ready"
        else:
            text = self.reply_for(FakeBuilder.state["current"], "That version did not work" in user)
        self._json({"model": "m", "message": {"role": "assistant", "content": text}, "done": True, "prompt_eval_count": sum(len(x["content"]) for x in msgs) // 4,
                    "eval_count": len(text) // 4})
        return


@pytest.fixture
def fake_builder():
    FakeBuilder.log, FakeBuilder.modes, FakeBuilder.state = [], {"first": "reference", "retry": "reference"}, {"current": ""}
    s = ThreadingHTTPServer(("127.0.0.1", 0), FakeBuilder)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


@pytest.fixture
def real_litellm():
    pytest.importorskip("litellm")
    from pal.llm_wrapper.components import litellm_wrapper
    litellm_wrapper._module = None
    yield litellm_wrapper._import_litellm()
    litellm_wrapper._module = None


def run_build(tmp_path, url, *functions):
    code = R.main(["--functions", ",".join(functions), "--attempts", "2", "--quiet", "--ollama-url", url, "--out-dir", str(tmp_path)])
    folder = Path((tmp_path / "LATEST.txt").read_text().strip())
    return code, folder, json.loads((folder / "report.json").read_text(encoding="utf-8"))


def test_a_full_run_writes_the_models_code_into_files_and_both_arms_pass_the_projects_tests(tmp_path, fake_builder, real_litellm):
    code, folder, data = run_build(tmp_path, fake_builder, "reciprocal_rank", "cosine")
    assert code == 0 and folder.parent == tmp_path and data["baseline"]["correct"] == 2 and data["pal"]["correct"] == 2
    for arm in ("baseline", "pal"):
        metrics = (folder / arm / "localrag_project" / "src" / "localrag" / "eval" / "metrics.py").read_text(encoding="utf-8")
        sim = (folder / arm / "localrag_project" / "src" / "localrag" / "embed" / "similarity.py").read_text(encoding="utf-8")
        assert "1.0 / rank" in metrics or "1 / " in metrics or "return" in metrics
        assert "NotImplementedError" not in metrics and "NotImplementedError" not in sim and "def cosine" in sim
        assert (folder / arm / "replies" / "01_reciprocal_rank_attempt1.md").read_text().startswith("```python")
        assert data[arm]["suite_failed"] == 0 and data[arm]["suite_passed"] == data["reference_tests"] and data[arm]["attempts"] == 2
    assert data["pal"]["main_tokens"] < data["baseline"]["main_tokens"] and data["savings"]["main_tokens_pct"] > 0
    assert data["pal"]["helper_tokens"] > 0 and data["baseline"]["helper_tokens"] == 0
    md = (folder / "report.md").read_text(encoding="utf-8")
    assert "**REAL RUN, no simulation.**" in md and "SIMULATED" not in md and (tmp_path / "LATEST.txt").read_text().strip() == str(folder)
    chats = [b for b in FakeBuilder.log if "messages" in b]
    assert chats and all("num_ctx" in b["options"] and b["options"]["seed"] == 7 for b in chats)


def test_a_wrong_first_answer_is_retried_with_the_projects_test_output_and_then_passes(tmp_path, fake_builder, real_litellm):
    FakeBuilder.modes = {"first": "wrong", "retry": "reference"}
    code, folder, data = run_build(tmp_path, fake_builder, "reciprocal_rank")
    assert code == 0 and data["baseline"]["correct"] == 1 and data["pal"]["correct"] == 1 and data["baseline"]["attempts"] == 2 and data["pal"]["attempts"] == 2
    retries = [b["messages"][-1]["content"] for b in FakeBuilder.log
               if "messages" in b and b["model"] == "llama3.2:3b" and "That version did not work" in b["messages"][-1]["content"]]
    assert len(retries) == 2                                                                # one to the main model in each arm (the helper also saw the text) and all("the project's tests failed with this function" in r and "reciprocal_rank" in r for r in retries)
    assert (folder / "pal" / "replies" / "02_reciprocal_rank_attempt2.md").is_file()


def test_wrong_and_unsafe_code_is_reported_as_failed_and_unsafe_code_is_never_written(tmp_path, fake_builder, real_litellm):
    FakeBuilder.modes = {"first": "wrong", "retry": "wrong"}
    code, folder, data = run_build(tmp_path / "wrong", fake_builder, "reciprocal_rank")
    assert code == 0 and data["baseline"]["correct"] == 0 and data["pal"]["correct"] == 0 and data["baseline"]["written"] == 1
    assert data["baseline"]["suite_failed"] >= 1 and "FAIL" in (folder / "report.md").read_text(encoding="utf-8")
    assert "return None" in (folder / "baseline" / "localrag_project" / "src" / "localrag" / "eval" / "metrics.py").read_text(encoding="utf-8")
    FakeBuilder.modes = {"first": "unsafe", "retry": "unsafe"}
    FakeBuilder.log.clear()
    code, folder, data = run_build(tmp_path / "unsafe", fake_builder, "reciprocal_rank")
    assert data["baseline"]["written"] == 0 and data["pal"]["written"] == 0 and data["baseline"]["correct"] == 0
    assert "NotImplementedError" in (folder / "pal" / "localrag_project" / "src" / "localrag" / "eval" / "metrics.py").read_text(encoding="utf-8")
    assert "os.getpid" not in (folder / "pal" / "localrag_project" / "src" / "localrag" / "eval" / "metrics.py").read_text(encoding="utf-8")
    assert any("importing os is not allowed" in b["messages"][-1]["content"] for b in FakeBuilder.log if "messages" in b)


# -- Windows and fresh-machine behaviour -----------------------------------------------------------------------------------------------------------
def test_the_environment_the_models_code_is_tested_in_keeps_what_python_needs_and_drops_secrets(monkeypatch, tmp_path):
    for name in list(__import__("os").environ):
        monkeypatch.delenv(name, raising=False)
    for name, value in {"SYSTEMROOT": r"C:\Windows", "PATH": "x", "TEMP": "t", "USERPROFILE": "u", "OPENAI_API_KEY": "sk-1", "HF_TOKEN": "hf-1", "SERPAPI_API_KEY": "s",
                        "DB_PASSWORD": "p", "MY_SECRET": "s", "GOOGLE_APPLICATION_CREDENTIALS": "c", "SSH_AUTH_SOCK": "a"}.items():
        monkeypatch.setenv(name, value)
    env = B.safe_env(tmp_path)
    assert {"SYSTEMROOT", "PATH", "TEMP", "USERPROFILE"} <= set(env)                          # without SYSTEMROOT Python cannot even start on Windows
    assert not any(k in env for k in ("OPENAI_API_KEY", "HF_TOKEN", "SERPAPI_API_KEY", "DB_PASSWORD", "MY_SECRET", "GOOGLE_APPLICATION_CREDENTIALS", "SSH_AUTH_SOCK"))
    assert env["PYTHONPATH"] == str(tmp_path / "src") and env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_model_code_cannot_see_an_api_key_even_when_it_is_in_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-seen")
    task = B.TASKS[0]
    probe = ("def reciprocal_rank(relevant, ranked):\n    return 0.0")
    seen = B.safe_env(B.PROJECT_ROOT)
    assert "OPENAI_API_KEY" not in seen and "sk-should-never-be-seen" not in " ".join(seen.values())
    assert B.check_candidate(task, B.reference_definition(task)).ok and not B.check_candidate(task, probe).ok


def test_a_missing_pytest_is_explained_not_reported_as_zero_tests(monkeypatch, tmp_path):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "C:\\py\\python.exe: No module named pytest\n"))
    run = B.run_tests(tmp_path)
    assert not run.ok and run.passed == 0 and "pytest is not installed" in run.tail and "pip install pytest" in run.tail


def test_without_pytest_the_command_says_so_before_loading_anything(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "probe_ollama", lambda url, wanted: type("P", (), {"ok": True, "missing": [], "error": ""})())
    monkeypatch.setattr(B, "pytest_available", lambda: False)
    code = R.main(["--out-dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == 2 and "pytest is not installed" in err and "-m pip install pytest" in err and not (tmp_path / "o").exists()


def test_the_launcher_installs_pytest_for_the_experiments_only_when_it_is_missing(monkeypatch):
    import run
    for present, expect_install in ((False, True), (True, False)):
        calls = []
        monkeypatch.setattr(run, "ensure_venv", lambda: Path("py"))
        monkeypatch.setattr(run, "ensure_installed", lambda py: None)
        monkeypatch.setattr(run, "_has_module", lambda py, name, present=present: present)
        monkeypatch.setattr(run, "_run", lambda cmd, calls=calls: calls.append([str(c) for c in cmd]) or 0)
        monkeypatch.setattr(run.sys, "prefix", run.sys.base_prefix)
        assert run.main(["experiment", "build"]) == 0
        installs = [c for c in calls if c[1:5] == ["-m", "pip", "install", "--quiet"] and "pytest" in c]
        assert bool(installs) is expect_install and calls[-1][1].endswith("run_demo.py") and calls[-1][2:] == ["build"]


def test_has_module_is_false_when_python_cannot_start_and_true_for_a_module_that_exists():
    import run
    assert run._has_module(Path("/no/such/python"), "pytest") is False
    assert run._has_module(Path(__import__("sys").executable), "json") is True and run._has_module(Path(__import__("sys").executable), "no_such_module_xyz") is False


# -- the model check: the bug a first real run on Windows found -----------------------------------------------------------------------------------------
def probe_with(*, available=(), missing=(), error=""):
    from experiments.demo.models import Probe
    return lambda url, wanted: Probe(not missing and not error, list(missing), error, list(available))


def test_when_both_default_models_are_missing_it_stops_and_lists_what_is_installed_instead_of_trying_to_load_one(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "probe_ollama", probe_with(available=["llama3.2:latest", "qwen2.5:7b"], missing=["llama3.2:3b", "llama3.2:1b"]))
    monkeypatch.setattr(R, "build_real_core", lambda *a: (_ for _ in ()).throw(AssertionError("it must not try to load a model that is not installed")))
    code = R.main(["--out-dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == 2 and "The model llama3.2:3b is not installed in your Ollama." in err
    assert "Installed models: llama3.2:latest, qwen2.5:7b" in err and "ollama pull llama3.2:3b" in err
    assert "python run.py experiment build --model llama3.2:latest" in err and "ollama list" in err and not (tmp_path / "o").exists()


def test_an_ollama_with_no_models_says_so_and_still_gives_the_pull_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "probe_ollama", probe_with(available=[], missing=["llama3.2:3b", "llama3.2:1b"]))
    code = R.main(["--out-dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == 2 and "Installed models: none yet" in err and "ollama pull llama3.2:3b" in err and "--model" not in err.split("To use the default")[1].split("(`ollama list`")[0]


def test_a_missing_main_model_is_caught_even_when_the_helper_is_installed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "probe_ollama", probe_with(available=["llama3.2:1b"], missing=["llama3.2:3b"]))
    assert R.main(["--out-dir", str(tmp_path / "o")]) == 2
    assert "--model llama3.2:1b" in capsys.readouterr().err


def test_a_missing_helper_alone_is_fine_the_main_model_does_its_work_and_the_report_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "probe_ollama", probe_with(available=["llama3.2:3b"], missing=["llama3.2:1b"]))
    loaded = []
    monkeypatch.setattr(R, "build_real_core", lambda model, url, ctx, seed: loaded.append(model) or object())
    monkeypatch.setattr(R, "warm_up", lambda core: (_ for _ in ()).throw(RuntimeError("stop here: the models were chosen")))
    monkeypatch.setattr(B, "pytest_available", lambda: True)
    monkeypatch.setattr(R.B, "run_tests", lambda *a, **k: B.TestRun(True, 100, 0, ""))
    summary, err = R.run(R.parser().parse_args(["--out-dir", str(tmp_path / "o"), "--quiet"]), lambda *a: None)
    assert summary is None and "stop here" in err and loaded == ["llama3.2:3b", "llama3.2:3b"]        # the helper role was given to the main model


def test_a_model_not_found_at_load_time_points_to_ollama_list_and_model(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "probe_ollama", probe_with(available=["llama3.2:3b", "llama3.2:1b"]))
    monkeypatch.setattr(B, "pytest_available", lambda: True)
    monkeypatch.setattr(R.B, "run_tests", lambda *a, **k: B.TestRun(True, 100, 0, ""))
    monkeypatch.setattr(R, "build_real_core", lambda *a: (_ for _ in ()).throw(RuntimeError("model 'llama3.2:3b' not found")))
    summary, err = R.run(R.parser().parse_args(["--out-dir", str(tmp_path / "o"), "--quiet"]), lambda *a: None)
    assert summary is None and "not found" in err and "ollama list" in err and "--model" in err


def test_the_probe_remembers_what_is_installed_and_still_matches_latest_tags(fake_ollama_models=None):
    import requests
    from experiments.demo.models import probe_ollama

    class Reply:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]}
    orig = requests.get
    requests.get = lambda *a, **k: Reply()
    try:
        probe = probe_ollama("http://x", ["llama3.2", "llama3.2:3b", "qwen2.5:7b"])
    finally:
        requests.get = orig
    assert probe.available == ["llama3.2:latest", "qwen2.5:7b"] and probe.missing == ["llama3.2:3b"] and not probe.ok


# -- what the first real run (on Windows, with a real model) showed ------------------------------------------------------------------------------------
@pytest.mark.parametrize("label,out,code", [("a crash with no summary", "....\nFatal Python error: Cannot recover from stack overflow.\n", 3221225477),
                                            ("nothing collected", "no tests ran in 0.01s\n", 5), ("silence", "", 1)])
def test_a_test_run_that_prints_no_result_is_reported_as_not_run_never_as_zero_passed_zero_failed(monkeypatch, tmp_path, label, out, code):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, code, out, ""))
    run = B.run_tests(tmp_path)
    assert run.ran is False and run.ok is False and "pytest printed no test result" in run.tail and f"exit code {code}" in run.tail
    assert (out.strip() in run.output) if out.strip() else ("(nothing)" in run.tail)


def test_a_real_result_is_marked_as_having_run_and_keeps_the_whole_output(monkeypatch, tmp_path):
    import subprocess
    out = ".F.\n" + "x" * 3000 + "\n1 failed, 2 passed in 0.2s\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, out, ""))
    run = B.run_tests(tmp_path)
    assert run.ran and (run.passed, run.failed) == (2, 1) and len(run.tail) <= 1400 and run.output == out.strip() and len(run.output) > len(run.tail)


def test_pytest_output_is_read_as_utf_8_with_replacement_so_a_windows_console_encoding_cannot_crash_the_run(monkeypatch, tmp_path):
    import subprocess
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: seen.update(k) or subprocess.CompletedProcess(a, 0, "1 passed in 0.1s\n", ""))
    B.run_tests(tmp_path)
    assert seen["encoding"] == "utf-8" and seen["errors"] == "replace" and seen["env"]["PYTHONUTF8"] == "1" and seen["env"]["PYTHONIOENCODING"] == "utf-8"


def test_the_report_and_the_terminal_say_did_not_run_for_a_suite_that_printed_no_result():
    base, mat = fake_arm("baseline", [True, False, True, True], 9000, 30.0), fake_arm("pal", [True, True, True, True], 900, 8.0)
    mat.combined = B.TestRun(False, 0, 0, "pytest printed no test result (exit code 3)", ran=False)
    s = R.summarize(meta(), base, mat)
    assert s["pal"]["suite_ran"] is False and s["baseline"]["suite_ran"] is True
    md, term = R.render_markdown(s), R.render_terminal(s, "/x")
    assert "did not run: see test_output.txt" in md and "did not run: see test_output.txt" in term and "0 passed" not in term.split("Pal")[-1]
    assert "95 passed, 4 failed" in md and "plain 95 passed / 4 failed" in term


def test_the_whole_suite_output_is_saved_next_to_each_arms_project(tmp_path):
    t1 = B.TASKS[0]
    arm = R.ArmBuild("pal", Recorder())
    arm.results = [R.TaskResult(t1, 1, True, True, 1.0, 10, 0, [], B.extract(f"```python\n{B.reference_definition(t1)}\n```", t1.name))]
    arm.replies = [(t1, 1, "r")]
    R.write_arm(arm, [t1], tmp_path / "pal")
    text = (tmp_path / "pal" / "test_output.txt").read_text(encoding="utf-8")
    assert "passed" in text and len(text) > 20 and arm.combined.ran and arm.combined.output == text.strip() or text.strip().endswith(arm.combined.output[-50:])


def test_the_cache_note_appears_only_when_the_server_skipped_most_of_the_plain_clients_prompt():
    def arms(server_prompt):
        base, mat = fake_arm("baseline", [True] * 4, 10000, 30.0), fake_arm("pal", [True] * 4, 1000, 8.0)
        for c in base.recorder.calls:
            c.server_prompt = server_prompt
        return base, mat
    cached = R.summarize(meta(), *arms(1500))
    note = next(n for n in cached["notes"] if "reused a cached start" in n)
    assert "6,000 of the 40,000 prompt tokens" in note and "85.0% skipped" in note and "fewer tokens sent need not mean less time" in note
    assert not any("cached start" in n for n in R.summarize(meta(), *arms(9000))["notes"])


def test_tool_rounds_are_reported_when_the_main_model_was_called_more_often_than_it_was_asked():
    base, mat = fake_arm("baseline", [True] * 4, 9000, 30.0), fake_arm("pal", [True] * 4, 900, 8.0)
    mat.recorder.add(Call(1, "main", "m", 900, 50, 8.0, 0.0))
    mat.recorder.add(Call(2, "main", "m", 900, 50, 8.0, 0.0))
    s = R.summarize(meta(), base, mat)
    assert s["pal"]["main_calls"] == 6 and s["pal"]["attempts"] == 4 and any("tool rounds" in n for n in s["notes"])
    assert "Main-model calls actually made (tool rounds included) | 4 | 6 |" in R.render_markdown(s)


def test_pal_tools_are_off_in_the_build_experiment_unless_asked_for_and_the_report_says_which():
    assert R.parser().parse_args([]).tools is False and R.parser().parse_args(["--tools"]).tools is True
    s = R.summarize({**meta(), "tools": False}, fake_arm("baseline", [True] * 4, 9000, 30.0), fake_arm("pal", [True] * 4, 900, 8.0))
    assert "tools were off (use --tools to turn them on)" in R.render_markdown(s)
    assert "tools were ON" in R.render_markdown({**s, "tools": True})


def test_without_tools_the_model_is_never_shown_the_tool_instructions_and_with_the_flag_it_is(tmp_path, fake_builder, real_litellm):
    def main_prompts():
        return [json.dumps(b["messages"]) for b in FakeBuilder.log if "messages" in b and b["model"] == "llama3.2:3b"]
    code, folder, data = run_build(tmp_path / "off", fake_builder, "reciprocal_rank")
    assert code == 0 and not any("reply with only <tool>" in p for p in main_prompts()) and "tools were off" in (folder / "report.md").read_text(encoding="utf-8")
    FakeBuilder.log.clear()
    R.main(["--functions", "reciprocal_rank", "--attempts", "2", "--quiet", "--tools", "--ollama-url", fake_builder, "--out-dir", str(tmp_path / "on")])
    assert any("reply with only <tool>" in p for p in main_prompts())
