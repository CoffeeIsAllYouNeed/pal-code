"""The demo experiment (experiments/demo): not part of the package, but the report it writes must be right about what it says."""
import csv
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from experiments.demo import cli, report, runner
from experiments.demo import scenario as S
from experiments.demo.models import (
    SIM_CALL_OVERHEAD,
    SIM_SPEED,
    Probe,
    Recorder,
    RecordingCore,
    SimModel,
    SimScript,
    probe_ollama,
)
from experiments.demo.scenario import GREETINGS, SIGNOFFS, SIZES, build

from pal.chat.tools import direct_answer
from pal.llm_wrapper.components.base import LLMSyscall
from pal.tokens import estimate_tokens

ROOT = Path(__file__).resolve().parents[2]
QUICK = ["--size", "small", "--quiet", "--no-code"]


def run_exp(tmp_path, *extra, say=lambda *a: None):
    args = cli.parser().parse_args([*QUICK, "--simulate", "--out", str(tmp_path), *extra])
    return cli.run(args, say)


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    exp, err = run_exp(tmp_path_factory.mktemp("sim"))
    assert err is None
    return exp, report.summarize(exp)


@pytest.fixture(scope="module")
def sim_big(tmp_path_factory):
    args = cli.parser().parse_args(["--size", "big", "--quiet", "--no-code", "--simulate", "--out", str(tmp_path_factory.mktemp("big"))])
    exp, err = cli.run(args, lambda *a: None)
    assert err is None
    return exp, report.summarize(exp)


# -- the workload ---------------------------------------------------------------------------------------------------------------------------
def test_the_same_seed_builds_the_same_session_and_another_seed_draws_different_questions():
    a, b, c = build("big", 7), build("big", 7), build("big", 8)
    assert a == b and a.turns == b.turns and a.document == b.document
    assert c.turns != a.turns and c.document == a.document and c.project_files == a.project_files      # the project is fixed; the seed picks what is asked


@pytest.mark.parametrize("size", list(SIZES))
def test_each_size_has_the_promised_turns_and_attaches_the_promised_packages(size):
    sc, cfg = build(size), SIZES[size]
    assert len(sc.turns) == cfg["turns"] and [t.i for t in sc.turns] == list(range(1, cfg["turns"] + 1))
    assert {p.split("/")[0] for p in sc.project_files if "/" in p} == set(cfg["packages"])
    assert any("/" not in p for p in sc.project_files) == cfg["top"]


def test_the_session_is_the_real_localrag_project_not_generated_filler():
    sc = build("max")
    assert sc.project_name == "localrag" and len(sc.project_files) == len(list(S.SRC.rglob("*.py"))) - sum(1 for p in S.SRC.rglob("*.py") if p.stat().st_size == 0)
    assert sc.project_files["core/config.py"] == (S.SRC / "core" / "config.py").read_text(encoding="utf-8")
    for name in S.DOC_FILES:
        assert (S.PROJECT_ROOT / name).read_text(encoding="utf-8").strip() in sc.document and f"<!-- file: {name} -->" in sc.document
    import ast
    for path, code in sc.project_files.items():
        ast.parse(code, path)
    import zipfile
    names = zipfile.ZipFile(io.BytesIO(sc.project_zip())).namelist()
    assert len(names) == len(sc.project_files) and "localrag/core/config.py" in names and "localrag/mcp/server.py" in names


def test_the_big_session_is_big_enough_to_make_the_point_and_the_sizes_are_ordered():
    small, big, mx = build("small"), build("big"), build("max")
    tokens = [estimate_tokens(x.project_text()) for x in (small, big, mx)]
    assert tokens[0] < tokens[1] < tokens[2] and tokens[1] > 20_000 and tokens[2] > 35_000 and estimate_tokens(big.document) > 4_000 and mx.project_lines() > 3_000
    assert set(big.kinds()) == {"doc", "math", "time", "chat", "followup", "code"} and len(big.turns) >= 30
    assert big.turns[0].attach == "document" and [t.attach for t in big.turns].count("project") == 1


def test_every_expected_answer_is_checkable_against_the_real_files():
    sc = build("max")
    funcs = {f.name: f for f in S.functions(sc.project_files)}
    by_doc = {f.doc: f for f in funcs.values()}
    facts, defaults = S.doc_facts(), dict(S.setting_defaults(sc.project_files))
    seen = set()
    for t in sc.turns:
        name = (re.search(r"`(\w+)`", t.text) or [None, None])[1]
        if t.kind == "code" and "documented as" in t.text:
            assert by_doc[re.search(r'documented as "(.+?)"\?', t.text).group(1)].name == t.expect[0]
            seen.add("where")
        elif t.kind == "code" and "default value" in t.text:
            assert defaults[name] == t.expect[0]
            seen.add("setting")
        elif t.kind == "code" and "which files call" in t.text:
            assert S.callers(sc.project_files, funcs[name]) == sorted(t.expect)
            seen.add("callers")
        elif t.kind == "code":
            assert funcs[name].params == t.expect
            seen.add("params")
        elif t.kind in ("doc", "followup") and "error code" in t.text:
            assert (t.expect[0], re.search(r'means "(.+?)"', t.text).group(1)) in facts["errors"]
            seen.add("error")
        elif t.kind in ("doc", "followup") and "arguments does" in t.text:
            assert (name, list(t.expect)) in facts["tools"]
            seen.add("tool")
        elif t.kind in ("doc", "followup"):
            assert (name, t.expect[0]) in facts["defaults"]
            seen.add("documented default")
        elif t.kind == "math":
            a, b = (int(x) for x in re.findall(r"\d+", t.text))
            assert t.expect == (str(a * b),)
        elif t.kind == "time":
            assert t.expect_regex and not t.expect
    assert seen == {"where", "setting", "callers", "params", "error", "tool", "documented default"}      # every kind of question was asked and checked


def test_the_documentation_tables_the_questions_come_from_agree_with_the_code():
    code = dict(S.setting_defaults(S.load_source(SIZES["max"]["packages"], True)))
    documented = S.doc_facts()["defaults"]
    assert len(documented) >= 25 and all(code[name] == value for name, value in documented if name in code)
    assert ("min_score", "0.15") in documented and ("rrf_k", "60") in documented


def test_wordy_questions_are_wordy_and_the_easy_ones_are_exactly_what_a_direct_answer_accepts():
    for t in build("big").turns:
        if t.kind in ("doc", "code", "chat", "followup"):
            assert any(t.text.startswith(g) for g in GREETINGS) and any(t.text.endswith(s) for s in SIGNOFFS), t.i
        else:
            assert direct_answer(t.text) is not None, t.text                  # math and time need no model at all


def test_no_code_leaves_out_the_source_and_its_questions_but_keeps_the_documentation_and_a_bad_size_is_refused():
    sc = build("big", include_code=False)
    assert sc.project_files == {} and "code" not in sc.kinds() and all(t.attach != "project" for t in sc.turns) and sc.include_code is False
    assert sc.turns[0].attach == "document" and sc.document
    with pytest.raises(ValueError, match="size must be one of"):
        build("huge")


# -- the models and the recorder -----------------------------------------------------------------------------------------------------------
def test_the_simulated_model_answers_by_prompt_kind_and_its_speed_is_the_stated_one():
    script = SimScript(answer="MAIN ANSWER", short_answer="SHORT", route="local")
    helper, main = SimModel("helper", script), SimModel("main", script)
    ask = lambda m, system, user: m.chat(LLMSyscall(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]))
    assert json.loads(ask(helper, "You route chat tasks.", "<task>x</task>").text)["route"] == "local"
    assert json.loads(ask(helper, "You audit a rewrite.", "x").text) == {"ok": True, "lost": []}
    assert ask(helper, "You answer short tasks.", "x").text == "SHORT"
    greeted = GREETINGS[0] + "how many days is D3 retained?" + SIGNOFFS[0]
    assert ask(helper, "Rewrite.", f"<text>\n{greeted}\n</text>").text == "<text>\nHow many days is D3 retained?\n</text>"
    r = ask(main, "You are a helpful assistant.", "q " * 900)
    assert r.text == "MAIN ANSWER"
    p, c = r.prompt_tokens, r.completion_tokens
    assert r.meta["sim_seconds"] == pytest.approx(SIM_CALL_OVERHEAD + p / SIM_SPEED["main"][0] + c / SIM_SPEED["main"][1])


def test_the_recorder_counts_tokens_the_same_way_for_every_call_and_never_claims_server_counts_for_a_simulation():
    rec, script = Recorder(), SimScript(answer="four words right here")
    main = RecordingCore(SimModel("main", script), "main", "m", rec)
    rec.turn = 3
    main.chat(LLMSyscall(messages=[{"role": "system", "content": "a" * 40}, {"role": "user", "content": "b" * 80}]))
    c = rec.calls[0]
    assert (c.turn, c.role, c.model) == (3, "main", "m")
    assert c.prompt_tokens == estimate_tokens("a" * 40) + estimate_tokens("b" * 80) and c.completion_tokens == estimate_tokens("four words right here")
    assert c.server_prompt is None and c.server_completion is None and c.truncated is False
    assert c.seconds > c.wall                                                          # computed model time, not the instant fake call


def test_streamed_answers_are_recorded_after_the_last_piece_and_a_stopped_stream_still_counts():
    rec = Recorder()
    helper = RecordingCore(SimModel("helper", SimScript(answer="one two three four")), "helper", "h", rec)
    assert "".join(helper.stream_chat(LLMSyscall(messages=[{"role": "user", "content": "hi"}]))) == "one two three four"
    assert len(rec.calls) == 1 and rec.calls[0].completion_tokens == estimate_tokens("one two three four")
    gen = helper.stream_chat(LLMSyscall(messages=[{"role": "user", "content": "hi"}]))
    next(gen)
    gen.close()
    assert len(rec.calls) == 2


def test_both_arms_are_held_to_the_same_answer_cap_but_the_helper_is_not_capped():
    seen = []

    class Spy(SimModel):
        def chat(self, syscall):
            seen.append(syscall.max_tokens)
            return super().chat(syscall)
    rec = Recorder()
    main = RecordingCore(Spy("main", SimScript()), "main", "m", rec, max_output=300)
    helper = RecordingCore(Spy("helper", SimScript()), "helper", "h", rec, max_output=300)
    main.chat(LLMSyscall(messages=[{"role": "user", "content": "x"}], max_tokens=2048))
    main.chat(LLMSyscall(messages=[{"role": "user", "content": "x"}], max_tokens=100))
    helper.chat(LLMSyscall(messages=[{"role": "user", "content": "x"}], max_tokens=2048))
    assert seen == [300, 100, 2048]


# -- the two arms ---------------------------------------------------------------------------------------------------------------------------
def test_the_baseline_resends_everything_every_turn_with_the_handbook_pasted_into_the_first_message():
    sc = build("small", include_code=False)
    replies = [f"reply {i}" for i in range(1, len(sc.turns) + 1)]
    first = runner.baseline_messages(sc, 1, [])
    assert first[0] == {"role": "system", "content": runner.BASELINE_SYSTEM} and sc.document in first[1]["content"] and sc.turns[0].text in first[1]["content"]
    last = runner.baseline_messages(sc, len(sc.turns), replies[:-1])
    assert len(last) == 1 + 2 * len(sc.turns) - 1 and sc.document in last[1]["content"]
    assert [m["content"] for m in last if m["role"] == "assistant"] == replies[:-1]
    assert runner.baseline_peak_tokens(sc) > estimate_tokens(sc.document)


def test_a_simulated_session_answers_every_turn_in_both_arms(sim):
    exp, s = sim
    assert len(exp.baseline.outcomes) == len(exp.pal.outcomes) == len(exp.scenario.turns)
    assert all(o.reply for o in exp.baseline.outcomes + exp.pal.outcomes)
    assert [o.turn for o in exp.pal.outcomes] == [t.i for t in exp.scenario.turns]


def test_easy_turns_are_answered_without_the_main_model_and_correctly(sim):
    exp, _ = sim
    by_turn = {t.i: t for t in exp.scenario.turns}
    for o in exp.pal.outcomes:
        t = by_turn[o.turn]
        if t.kind in ("math", "time"):
            assert o.handled_by == "tool" and exp.pal.recorder.of_turn(o.turn, "main") == []
            assert o.ok is True or o.ok is None or runner.check(t, o.reply)
    math = next(o for o in exp.pal.outcomes if by_turn[o.turn].kind == "math")
    assert by_turn[math.turn].expect[0] in math.reply
    assert any(re.search(r"\d{1,2}:\d{2}", o.reply) for o in exp.pal.outcomes if by_turn[o.turn].kind == "time")


def test_every_document_question_sends_less_than_the_baseline_even_in_the_small_session(sim):
    exp, _ = sim
    for i in [t.i for t in exp.scenario.turns if t.kind == "doc"][1:]:
        base = sum(c.prompt_tokens for c in exp.baseline.recorder.of_turn(i, "main"))
        mine = sum(c.prompt_tokens for c in exp.pal.recorder.of_turn(i, "main"))
        assert base > estimate_tokens(exp.scenario.document) and mine < 0.6 * base, (i, base, mine)


def test_in_the_big_session_a_document_question_sends_under_forty_percent_of_what_the_baseline_sends(sim_big):
    exp, s = sim_big
    for i in [t.i for t in exp.scenario.turns if t.kind == "doc"][1:]:
        base = sum(c.prompt_tokens for c in exp.baseline.recorder.of_turn(i, "main"))
        mine = sum(c.prompt_tokens for c in exp.pal.recorder.of_turn(i, "main"))
        assert base > estimate_tokens(exp.scenario.document) and mine < 0.4 * base, (i, base, mine)           # measured: 22% to 35%
    assert s["savings"]["main_tokens_pct"] > 75 and s["savings"]["seconds_pct"] > 60            # the best case it was built to be


def test_the_answer_check_looks_for_every_expected_fact_or_a_pattern():
    from experiments.demo.scenario import Turn
    t = Turn(1, "doc", "q", expect=("42 days", "Security"))
    assert runner.check(t, "It is 42 days, owned by Security.") is True and runner.check(t, "42 days only") is False
    assert runner.check(Turn(2, "math", "q", expect=("7006652",)), "The product is 7,006,652") is True        # commas are ignored
    assert runner.check(Turn(3, "time", "q", expect_regex=r"\d{1,2}:\d{2}"), "It is 14:32") is True and runner.check(Turn(3, "time", "q", expect_regex=r"\d{1,2}:\d{2}"), "no idea") is False
    assert runner.check(Turn(4, "x", "q"), "anything") is None


def test_a_turns_latency_is_the_model_time_plus_only_the_time_outside_the_models():
    from experiments.demo.models import Call
    rec = Recorder()
    rec.add(Call(1, "main", "m", 10, 5, seconds=7.0, wall=0.002))
    rec.add(Call(1, "helper", "h", 5, 2, seconds=1.5, wall=0.001))
    secs, over = runner._turn_seconds(rec, 1, wall=0.05)
    assert over == pytest.approx(0.047) and secs == pytest.approx(7.0 + 1.5 + 0.047)
    assert runner._turn_seconds(rec, 2, wall=0.2) == (pytest.approx(0.2), pytest.approx(0.2))            # no calls: all of it is overhead


# -- the numbers -----------------------------------------------------------------------------------------------------------------------------
def test_the_savings_are_the_difference_of_the_two_arms_and_the_percentages_follow(sim):
    _, s = sim
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    assert sv["main_tokens"] == b["main_total"] - m["main_total"] > 0 and sv["main_tokens_pct"] == report.pct(sv["main_tokens"], b["main_total"])
    assert sv["all_tokens"] == (b["main_total"] + b["helper_total"]) - (m["main_total"] + m["helper_total"])
    assert sv["main_calls_avoided"] == b["main_calls"] - m["main_calls"] > 0 and b["helper_total"] == 0 and m["helper_total"] > 0
    assert sv["seconds"] == pytest.approx(b["seconds"] - m["seconds"], abs=0.02) and sv["seconds"] > 0 and 0 < sv["seconds_pct"] < 100
    assert b["main_total"] == b["main_prompt"] + b["main_completion"] and m["main_total"] == m["main_prompt"] + m["main_completion"]


def test_the_kinds_add_up_to_the_totals_and_every_turn_is_counted_once(sim):
    _, s = sim
    assert sum(r["turns"] for r in s["by_kind"]) == s["scenario"]["turns"] == s["baseline"]["turns"] == s["pal"]["turns"]
    assert sum(r["baseline_tokens"] for r in s["by_kind"]) == s["baseline"]["main_total"]
    assert sum(r["pal_tokens"] for r in s["by_kind"]) == s["pal"]["main_total"]
    assert sum(r["baseline_seconds"] for r in s["by_kind"]) == pytest.approx(s["baseline"]["seconds"], abs=0.05)
    assert sum(s["handled_by"].values()) == s["scenario"]["turns"] and s["handled_by"].get("tool", 0) == s["scenario"]["kinds"]["math"] + s["scenario"]["kinds"]["time"]
    easy = {r["kind"]: r for r in s["by_kind"]}
    assert easy["math"]["pal_tokens"] == 0 and easy["math"]["tokens_saved_pct"] == 100.0 and easy["math"]["seconds_saved_pct"] == 100.0


def test_pal_reports_which_of_its_parts_saved_tokens_as_its_own_estimates(sim):
    _, s = sim
    est = s["pal_estimates"]
    assert est["documents"] > est["input_savers"] > 0 and est["total"] >= est["documents"]


def test_percentages_and_the_95th_percentile_handle_empty_and_tiny_inputs():
    assert report.pct(5, 0) == 0.0 and report.pct(1, 3) == 33.3 and report.pct(-5, 10) == -50.0
    assert report.p95([]) == 0.0 and report.p95([4.0]) == 4.0 and report.p95(list(map(float, range(1, 101)))) == 95.0


# -- the report ---------------------------------------------------------------------------------------------------------------------------
def test_the_markdown_report_has_every_section_and_says_it_is_simulated_and_a_best_case(sim):
    _, s = sim
    md = report.render_markdown(s)
    for h in ("# Pal demo experiment", "## Result at a glance", "## What was run", "## How each turn was answered", "## By kind of turn",
              "## Where the savings came from", "## How far to trust this", "## Setup"):
        assert h in md
    assert md.count("SIMULATED") >= 2 and "answers are scripted" in md and "best case on purpose" in md and "deliberately naive" in md
    assert "In this SIMULATED run, Pal used" in md and "Answers containing the expected fact" not in md and "REAL RUN" not in md
    assert f"{s['savings']['main_tokens_pct']}%" in md and f"{s['savings']['seconds_pct']}%" in md
    assert re.search(r"\| \*\*Latency, whole session\*\* \(s\) \| [\d,.]+ \| [\d,.]+ \|", md)


def test_a_real_report_shows_the_answer_check_the_cache_caveat_and_no_simulation_text(sim):
    _, s = sim
    real = json.loads(json.dumps(s))
    real["mode"], real["sim_speed"] = "real", None
    md = report.render_markdown(real)
    assert "**REAL RUN.**" in md and "Answers containing the expected fact" in md and "reuse a repeated prefix" in md and "SIMULATED" not in md
    assert "In this run, Pal used" in md and "truncated calls" in md


def test_the_notes_the_reader_must_know_are_printed_in_the_report(sim):
    _, s = sim
    md = report.render_markdown({**s, "notes": ["graphify is not installed, so the code part was left out."]})
    assert "> **Note.** graphify is not installed" in md


def test_files_are_written_with_the_json_the_csv_and_the_markdown_that_agree(tmp_path, sim):
    exp, s = sim
    paths = report.write_all(exp, tmp_path / "out")
    assert set(paths) == {"md", "json", "csv"} and all(Path(p).is_file() for p in paths.values())
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["savings"]["main_tokens"] == s["savings"]["main_tokens"] and data["mode"] == "simulated"
    rows = list(csv.DictReader(io.StringIO(Path(paths["csv"]).read_text(encoding="utf-8"))))
    assert len(rows) == len(exp.scenario.turns) and rows[0]["turn"] == "1"
    assert sum(int(r["baseline_main_tokens"]) for r in rows) == s["baseline"]["main_total"]
    assert sum(int(r["pal_main_tokens"]) for r in rows) == s["pal"]["main_total"]
    assert sum(int(r["pal_helper_tokens"]) for r in rows) == s["pal"]["helper_total"]
    assert Path(paths["md"]).read_text(encoding="utf-8").startswith("# Pal demo experiment")


def test_the_terminal_summary_is_short_labelled_and_points_at_the_files(sim):
    _, s = sim
    out = report.render_terminal(s, {"md": "a.md", "json": "a.json", "csv": "a.csv"})
    assert len(out.splitlines()) <= 16 and "SIMULATED run" in out and "Main-model tokens" in out and "Latency (seconds)" in out
    assert "a.md" in out and "best-case workload by design" in out


# -- the command ----------------------------------------------------------------------------------------------------------------------------
def test_one_command_runs_the_whole_experiment_and_writes_the_report(tmp_path, capsys):
    code = cli.main([*QUICK, "--simulate", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "SIMULATED run" in out and "saved" in out
    assert len(list(tmp_path.glob("demo-*.md"))) == 1 and len(list(tmp_path.glob("demo-*.json"))) == 1 and len(list(tmp_path.glob("demo-*-turns.csv"))) == 1


def test_the_same_seed_gives_the_same_tokens_and_the_same_simulated_model_time(tmp_path):
    a, _ = run_exp(tmp_path / "a")
    b, _ = run_exp(tmp_path / "b")
    sa, sb = report.summarize(a), report.summarize(b)
    for arm in ("baseline", "pal"):
        for key in ("main_calls", "main_prompt", "main_completion", "helper_total", "peak_prompt"):
            assert sa[arm][key] == sb[arm][key], (arm, key)
    assert sa["handled_by"] == sb["handled_by"] and sa["by_kind"][0]["baseline_tokens"] == sb["by_kind"][0]["baseline_tokens"]
    assert sa["baseline"]["seconds"] == pytest.approx(sb["baseline"]["seconds"], abs=0.05)


def test_the_two_arms_use_separate_recorders_so_nothing_is_counted_twice(sim):
    exp, _ = sim
    assert exp.baseline.recorder is not exp.pal.recorder
    assert all(c.role == "main" for c in exp.baseline.recorder.calls)
    assert {c.role for c in exp.pal.recorder.calls} == {"main", "helper"}


def test_without_ollama_the_default_falls_back_to_a_labelled_simulation_and_says_how_to_get_a_real_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_ollama", lambda url, wanted: Probe(False, error="could not connect, is `ollama serve` running?"))
    code = cli.main([*QUICK, "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "running the SIMULATED experiment instead" in out and "ollama pull llama3.2:3b" in out and "SIMULATED run" in out


def test_real_mode_without_ollama_is_an_error_that_names_the_commands_to_fix_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_ollama", lambda url, wanted: Probe(False, ["llama3.2:3b", "llama3.2:1b"]))
    code = cli.main([*QUICK, "--real", "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2 and "not usable" in err and "ollama serve" in err and "ollama pull llama3.2:3b" in err and "ollama pull llama3.2:1b" in err
    assert not list(tmp_path.glob("demo-*"))                                                   # no report is written for a run that did not happen


def test_a_missing_helper_model_falls_back_to_the_main_model_and_the_report_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "probe_ollama", lambda url, wanted: Probe(False, [wanted[1]]))
    sims = {}
    monkeypatch.setattr(cli, "build_real_core", lambda model, url, ctx, seed: sims.setdefault(model, SimModel("main", SimScript())))
    monkeypatch.setattr(cli, "warm_up", lambda core: None)
    args = cli.parser().parse_args([*QUICK, "--out", str(tmp_path), "--model", "big:1b", "--helper-model", "tiny:1b"])
    exp, err = cli.run(args, lambda *a: None)
    assert err is None and exp.mode == "real" and exp.helper_model == "big:1b"
    assert any("tiny:1b is not pulled" in n and "ollama pull tiny:1b" in n for n in exp.notes)


def test_the_context_window_is_sized_to_the_longest_prompt_and_too_big_or_too_small_is_handled(tmp_path, monkeypatch):
    sc = build("small", include_code=False)
    need = cli.needed_ctx(sc, 300)
    assert need % 1024 == 0 and need >= runner.baseline_peak_tokens(sc, 300) * 1.25
    assert cli.needed_ctx(build("max", include_code=False), 300) > cli.needed_ctx(build("big", include_code=False), 300) > need
    monkeypatch.setattr(cli, "probe_ollama", lambda url, wanted: Probe(True))
    monkeypatch.setattr(cli, "build_real_core", lambda *a: SimModel("main", SimScript()))
    monkeypatch.setattr(cli, "warm_up", lambda core: None)
    small_ctx = cli.parser().parse_args([*QUICK, "--num-ctx", "2048", "--out", str(tmp_path)])
    exp, err = cli.run(small_ctx, lambda *a: None)
    assert err is None and any("smaller than the longest prompt needs" in n for n in exp.notes)
    monkeypatch.setattr(cli, "MAX_CTX", 1024)
    exp2, err2 = cli.run(cli.parser().parse_args([*QUICK, "--out", str(tmp_path)]), lambda *a: None)
    assert exp2 is None and "over the 1,024 allowed (--max-ctx)" in err2 and "--size small" in err2


def test_without_graphify_the_code_part_is_left_out_and_the_report_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_graph_available", lambda: False)
    args = cli.parser().parse_args(["--size", "small", "--quiet", "--simulate", "--out", str(tmp_path)])
    exp, err = cli.run(args, lambda *a: None)
    assert err is None and exp.scenario.include_code is False and "code" not in exp.scenario.kinds()
    assert any("graphify is not installed" in n for n in exp.notes)
    md = report.render_markdown(report.summarize(exp))
    assert "graphify is not installed" in md and "The source was left out (graphify is not installed)" in md


def test_with_graphify_the_project_is_attached_and_questioned_through_the_code_graph(tmp_path):
    if not cli._graph_available():
        pytest.skip("graphify is not installed")
    args = cli.parser().parse_args(["--size", "small", "--quiet", "--simulate", "--out", str(tmp_path)])
    exp, err = cli.run(args, lambda *a: None)
    s = report.summarize(exp)
    assert err is None and exp.scenario.include_code and s["scenario"]["project_tokens"] > 0
    assert s["pal_estimates"]["code_graph"] > 0
    code_turns = [t.i for t in exp.scenario.turns if t.kind == "code"]
    for i in code_turns[1:]:                      # the small project is only about 900 tokens, so "less than 60%" is the honest bar here
        assert sum(c.prompt_tokens for c in exp.pal.recorder.of_turn(i, "main")) < 0.6 * sum(c.prompt_tokens for c in exp.baseline.recorder.of_turn(i, "main"))


def test_every_size_saves_most_of_the_tokens_and_most_of_the_time(sim, sim_big):
    for _, summary in (sim, sim_big):
        assert summary["savings"]["main_tokens_pct"] > 70 and summary["savings"]["seconds_pct"] > 50 and summary["savings"]["main_calls_avoided"] > 0


def test_with_the_source_attached_a_bigger_session_saves_a_larger_share(tmp_path):
    if not cli._graph_available():
        pytest.skip("graphify is not installed")
    shares = {}
    for size in ("small", "big"):
        exp, err = cli.run(cli.parser().parse_args(["--size", size, "--quiet", "--simulate", "--out", str(tmp_path)]), lambda *a: None)
        assert err is None
        shares[size] = report.summarize(exp)["savings"]
    assert shares["big"]["main_tokens_pct"] > shares["small"]["main_tokens_pct"] and shares["big"]["seconds_pct"] > shares["small"]["seconds_pct"]


def test_the_experiment_is_not_part_of_the_package_nothing_in_src_imports_it():
    offenders = [str(p) for p in (ROOT / "src").rglob("*.py") if "experiments" in p.read_text(encoding="utf-8").replace("experiments/", "")
                 and re.search(r"(?m)^\s*(from|import)\s+experiments", p.read_text(encoding="utf-8"))]
    assert offenders == []
    assert (ROOT / "experiments" / "run_demo.py").is_file()


def test_the_launcher_runs_the_experiment_with_one_command_and_still_starts_the_app_otherwise(tmp_path, monkeypatch):
    import run
    calls = []
    monkeypatch.setattr(run, "ensure_venv", lambda: Path("py"))
    monkeypatch.setattr(run, "ensure_installed", lambda py: None)
    monkeypatch.setattr(run, "_run", lambda cmd: calls.append([str(c) for c in cmd]) or 0)
    monkeypatch.setattr(run.sys, "prefix", run.sys.base_prefix)
    assert run.main(["experiment", "--size", "small"]) == 0
    assert calls[-1][1].endswith("experiments/run_demo.py") and calls[-1][2:] == ["--size", "small"]
    assert run.main([]) == 0 and calls[-1][1:4] == ["-m", "pal", "serve"]


# -- a real Ollama's wire format, with a fake server and the real LiteLLM -------------------------------------------------------------------------
class FakeOllama(BaseHTTPRequestHandler):
    log: list = []
    models = ["llama3.2:3b", "llama3.2:1b"]
    force_full_context = False

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({"models": [{"name": m} for m in FakeOllama.models]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.log.append(body)
        if "messages" not in body:                     # LiteLLM also asks about the model (/api/show): nothing to say
            self._json({})
            return
        msgs = body["messages"]
        system, user = msgs[0]["content"], msgs[-1]["content"]
        m = re.search(r"<text>\n(.*)\n</text>", user, re.S)
        text = ('{"route": "api", "reason": "x"}' if system.startswith("You route") else '{"ok": true, "lost": []}' if system.startswith("You audit")
                else f"<text>\n{m.group(1)}\n</text>" if m else "ready" if user == "Reply with the single word: ready" else "The answer is 42.")
        p = sum(len(x["content"]) for x in msgs) // 4
        if FakeOllama.force_full_context:
            p = body.get("options", {}).get("num_ctx", p)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write((json.dumps({"model": "m", "message": {"role": "assistant", "content": text}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"model": "m", "message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": p,
                                          "eval_count": len(text) // 4}) + "\n").encode())
            return
        self._json({"model": "m", "message": {"role": "assistant", "content": text}, "done": True, "prompt_eval_count": p, "eval_count": len(text) // 4})
        return


@pytest.fixture
def fake_ollama():
    FakeOllama.log, FakeOllama.models, FakeOllama.force_full_context = [], ["llama3.2:3b", "llama3.2:1b"], False
    s = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


@pytest.fixture
def real_litellm():
    """The test suite swaps in a fake LiteLLM by default; these tests need the real library talking real HTTP."""
    pytest.importorskip("litellm")
    from pal.llm_wrapper.components import litellm_wrapper
    litellm_wrapper._module = None
    yield litellm_wrapper._import_litellm()
    litellm_wrapper._module = None


def test_the_probe_tells_ready_missing_and_unreachable_apart(fake_ollama):
    assert probe_ollama(fake_ollama, ["llama3.2:3b", "llama3.2:1b"]).ok
    FakeOllama.models = ["llama3.2:3b"]
    p = probe_ollama(fake_ollama, ["llama3.2:3b", "llama3.2:1b"])
    assert not p.ok and p.missing == ["llama3.2:1b"] and p.error == ""
    FakeOllama.models = ["llama3.2:1b:latest", "tiny:latest"]
    assert probe_ollama(fake_ollama, ["tiny"]).ok                                              # "tiny" matches "tiny:latest"
    down = probe_ollama("http://127.0.0.1:9", ["x"], timeout=1)
    assert not down.ok and "could not connect" in down.error and "ollama serve" in down.error


def test_a_real_run_goes_through_litellm_to_ollama_with_the_right_options_and_reads_the_servers_token_counts(tmp_path, fake_ollama, real_litellm):
    code = cli.main([*QUICK, "--real", "--ollama-url", fake_ollama, "--out", str(tmp_path)])
    assert code == 0
    data = json.loads(next(tmp_path.glob("demo-*.json")).read_text(encoding="utf-8"))
    assert data["mode"] == "real" and data["baseline"]["server_prompt"] > 0 and data["pal"]["server_prompt"] > 0
    chats = [b for b in FakeOllama.log if "messages" in b]
    warm = [b for b in chats if b["messages"][-1]["content"] == "Reply with the single word: ready"]
    assert len(warm) == 2 and {b["model"] for b in warm} == {"llama3.2:3b", "llama3.2:1b"}      # both models were warmed up before timing
    assert chats[0] in warm and chats[1] in warm                                               # ... and before anything else was asked
    ctx = data["models"]["num_ctx"]
    assert all(b["options"]["num_ctx"] == ctx and b["options"]["seed"] == 7 for b in chats)     # the window is large enough, and the run is repeatable
    assert all(b.get("keep_alive") == "30m" for b in chats)
    assert any(b["options"].get("temperature") == 0 for b in chats if b["model"] == "llama3.2:3b" and b not in warm)
    assert {b["model"] for b in chats if b not in warm} == {"llama3.2:3b", "llama3.2:1b"}       # the main model and the helper were both used
    md = next(tmp_path.glob("demo-*.md")).read_text(encoding="utf-8")
    assert "**REAL RUN.**" in md and "Answers containing the expected fact" in md and "Prompt tokens as the server counted them" in md


def test_a_prompt_that_fills_the_context_window_is_flagged_as_truncated(tmp_path, fake_ollama, real_litellm):
    FakeOllama.force_full_context = True
    code = cli.main([*QUICK, "--real", "--ollama-url", fake_ollama, "--out", str(tmp_path)])
    data = json.loads(next(tmp_path.glob("demo-*.json")).read_text(encoding="utf-8"))
    assert code == 0 and data["baseline"]["truncated"] > 0 and data["pal"]["truncated"] > 0
    assert "truncated calls: " in next(tmp_path.glob("demo-*.md")).read_text(encoding="utf-8")


def test_the_session_experiment_also_lists_the_models_you_have_when_asked_for_a_real_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_ollama", lambda url, wanted: Probe(False, ["llama3.2:3b", "llama3.2:1b"], "", ["gemma3:4b"]))
    code = cli.main([*QUICK, "--real", "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2 and "Models you have: gemma3:4b" in err and "--model NAME" in err and "ollama pull llama3.2:3b" in err
