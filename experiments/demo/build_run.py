"""The real build experiment: a local model writes code, the code is written to files in the main directory, and the project's tests judge it.

    python run.py experiment build                 (the one-command way)
    python experiments/run_demo.py build

There is NO simulation here. It needs Ollama running with the models pulled, and it stops with instructions if it does not find them.
The same task is done twice, with the same local models: once by a plain chat client that resends the whole project every turn, and once
through Pal. For each, the report gives the tokens and time used and how many functions came out right.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from experiments.demo import build as B
from experiments.demo.models import (
    DEFAULT_HELPER, DEFAULT_MAIN, Recorder, RecordingCore, build_real_core, missing_model_message, probe_ollama, warm_up,
)
from experiments.demo.report import p95, pct
from experiments.demo.runner import _turn_seconds
from pal.errors import DependencyMissingError
from pal.llm_wrapper.components.base import LLMSyscall
from pal.tokens import estimate_tokens

REPO = Path(__file__).resolve().parents[2]
SYSTEM = "You are an expert Python developer. You work only from the project you are given and answer with code."


@dataclass
class TaskResult:
    task: B.Task
    attempts: int
    ok: bool                         # the project's tests pass with this function replaced by the model's version
    written: bool                    # some usable (parsable, safe) version of the function came out
    seconds: float                   # model and pipeline time; the time spent running tests is not counted
    main_tokens: int
    helper_tokens: int
    problems: list = field(default_factory=list)
    final: B.Extraction | None = None


@dataclass
class ArmBuild:
    name: str
    recorder: Recorder
    results: list = field(default_factory=list)
    replies: list = field(default_factory=list)       # (task, attempt, reply)
    combined: B.TestRun | None = None                 # the whole test suite, with everything the model wrote in place
    folder: str = ""
    notes: list = field(default_factory=list)


# -- solving one task, the same way in both arms ----------------------------------------------------------------------------------------
def solve(task: B.Task, stub: str, ask, attempts: int, params: list, rec: Recorder) -> tuple:
    """Ask for the function; if the reply cannot be used, or the project's tests fail, say why and ask again, up to `attempts` times."""
    prompt, problems, replies, turns, seconds, final, passed = B.task_message(task, stub), [], [], [], 0.0, None, False
    for attempt in range(1, attempts + 1):
        reply, secs, turn = ask(prompt, attempt == 1, stub)
        replies.append((task, attempt, reply))
        turns.append(turn)
        seconds += secs
        ex = B.extract(reply, task.name, params)
        if not ex.ok:
            problem = ex.error
        elif (why := B.unsafe_reason(ex.definition, ex.imports)):
            problem = why
        else:
            final = ex
            run = B.check_candidate(task, ex.definition, ex.imports)
            if run.ok:
                passed = True
                break
            problem = "the project's tests failed with this function:\n" + B.focus_failure(run.output or run.tail)
        problems.append(problem)
        prompt = B.retry_message(problem)
    main = sum(c.prompt_tokens + c.completion_tokens for t in turns for c in rec.of_turn(t, "main"))
    helper = sum(c.prompt_tokens + c.completion_tokens for t in turns for c in rec.of_turn(t, "helper"))
    return TaskResult(task, len(replies), passed, final is not None, seconds, main, helper, problems, final), replies


def project_text(files: dict) -> str:
    return "\n\n".join(f"# {name}\n{code}" for name, code in files.items())


def run_baseline(tasks, files, main: RecordingCore, attempts, max_output, progress=None) -> ArmBuild:
    arm, messages, state = ArmBuild("baseline", main.rec), [{"role": "system", "content": SYSTEM}], {"turn": 0, "shown": False}
    source = project_text(files)

    def ask(prompt, first, stub):
        state["turn"] += 1
        main.rec.turn = state["turn"]
        content = prompt if state["shown"] else f"Here is the full source of the project:\n{source}\n\n{prompt}"
        state["shown"] = True
        messages.append({"role": "user", "content": content})
        t0 = time.perf_counter()
        reply = main.chat(LLMSyscall(messages=list(messages), temperature=0, max_tokens=max_output)).text
        wall = time.perf_counter() - t0
        messages.append({"role": "assistant", "content": reply})
        return reply, _turn_seconds(main.rec, state["turn"], wall)[0], state["turn"]

    for i, task in enumerate(tasks, 1):
        stub = B.stub_of(files[task.path], task.name)
        res, replies = solve(task, stub, ask, attempts, B.param_names(B.find_function(_original(task), task.name)), main.rec)
        arm.results.append(res)
        arm.replies += replies
        if progress:
            progress("baseline", i, len(tasks), res)
    return arm


def _original(task: B.Task) -> str:
    return (B.SRC / task.path).read_text(encoding="utf-8")


def zip_of(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, code in files.items():
            z.writestr(f"localrag/{name}", code)
    return buf.getvalue()


def run_pal(tasks, files, main: RecordingCore, helper: RecordingCore, attempts, progress=None, tools: bool = False) -> ArmBuild:
    from pal.chat import ChatService
    from pal.config import load_settings
    from pal.knowledge_graph import KnowledgeGraph

    arm, state = ArmBuild("pal", main.rec), {"turn": 0, "attached": False}
    home = Path(tempfile.mkdtemp(prefix="pal-build-svc-"))
    settings = load_settings(env={}, overrides={"storage_dir": str(home), "knowledge_graph": {"dir": str(home / "graph")}, "memory": {"enabled": False},
                                                 "tools": {"enabled": tools}, "server": {"session_ttl_minutes": 240}})
    svc = ChatService(settings, local_core=helper, graph=KnowledgeGraph(settings.knowledge_graph))
    svc.api_core = lambda provider, model, key=None: main
    graph_ok = svc.graph.available()
    if not graph_ok:
        arm.notes.append("graphify is not installed, so the project was attached as a document (excerpts) instead of a code graph (pip install graphifyy).")

    def ask(prompt, first, stub):
        state["turn"] += 1
        main.rec.turn = helper.rec.turn = state["turn"]
        t0, attachments = time.perf_counter(), []
        if not state["attached"]:
            attachments = [svc.graph.ingest("build", "localrag.zip", zip_of(files))] if graph_ok else [
                {"name": "localrag_source.md", "text": project_text(files), "kind": "document"}]
            state["attached"] = True
        res = svc.send("build", "local", main.model, None, prompt, attachments, route="api", pasted=[stub] if first and stub else [])
        return res.reply, _turn_seconds(main.rec, state["turn"], time.perf_counter() - t0)[0], state["turn"]

    try:
        for i, task in enumerate(tasks, 1):
            stub = B.stub_of(files[task.path], task.name)
            res, replies = solve(task, stub, ask, attempts, B.param_names(B.find_function(_original(task), task.name)), main.rec)
            arm.results.append(res)
            arm.replies += replies
            if progress:
                progress("pal", i, len(tasks), res)
        arm.notes.append(f"Pal's own estimate of tokens saved: {json.dumps(svc.metrics.chat_summary().get('tokens_saved_estimate', {}))}")
    finally:
        svc.close()
    return arm


# -- writing the result to files ----------------------------------------------------------------------------------------------------------
def write_arm(arm: ArmBuild, tasks, folder: Path) -> None:
    """The project as the model left it, in <folder>/localrag_project, the raw replies in <folder>/replies, and the whole test suite run on it."""
    folder.mkdir(parents=True, exist_ok=True)
    project = B.copy_project(folder / "localrag_project")
    for task, res in zip(tasks, arm.results, strict=True):
        path = project / "src" / "localrag" / task.path
        code = B.apply_stub(path.read_text(encoding="utf-8"), task.name)
        if res.final is not None:
            code = B.replace_function(code, task.name, res.final.definition, list(res.final.imports))
        path.write_text(code, encoding="utf-8")
    (folder / "replies").mkdir(exist_ok=True)
    for n, (task, attempt, reply) in enumerate(arm.replies, 1):
        (folder / "replies" / f"{n:02d}_{task.name}_attempt{attempt}.md").write_text(reply, encoding="utf-8")
    arm.combined = B.run_tests(project, fail_fast=False)
    (folder / "test_output.txt").write_text(arm.combined.output or arm.combined.tail, encoding="utf-8")
    arm.folder = str(folder)


# -- the numbers ---------------------------------------------------------------------------------------------------------------------------
def totals(arm: ArmBuild) -> dict:
    calls = arm.recorder.calls
    main = [c for c in calls if c.role == "main"]
    helper = [c for c in calls if c.role == "helper"]
    correct = sum(1 for r in arm.results if r.ok)
    main_tokens = sum(c.prompt_tokens + c.completion_tokens for c in main)
    return {"tasks": len(arm.results), "correct": correct, "written": sum(1 for r in arm.results if r.written), "attempts": sum(r.attempts for r in arm.results),
            "main_calls": len(main), "main_tokens": main_tokens, "main_prompt": sum(c.prompt_tokens for c in main),
            "main_completion": sum(c.completion_tokens for c in main), "helper_tokens": sum(c.prompt_tokens + c.completion_tokens for c in helper),
            "server_prompt": sum(c.server_prompt for c in main) if main and all(c.server_prompt is not None for c in main) else None,
            "seconds": round(sum(r.seconds for r in arm.results), 2), "seconds_p95": round(p95([r.seconds for r in arm.results]), 2),
            "tokens_per_correct": round(main_tokens / correct) if correct else None, "truncated": sum(1 for c in calls if c.truncated),
            "suite_passed": arm.combined.passed if arm.combined else None, "suite_failed": arm.combined.failed if arm.combined else None,
            "suite_ran": arm.combined.ran if arm.combined else None}


def summarize(meta: dict, base: ArmBuild, mat: ArmBuild) -> dict:
    b, m = totals(base), totals(mat)
    notes = list(meta.get("notes", []))
    if b["server_prompt"] is not None and b["main_prompt"] and b["server_prompt"] < 0.6 * b["main_prompt"]:
        skipped = pct(b["main_prompt"] - b["server_prompt"], b["main_prompt"])
        notes.append(f"Ollama read only {b['server_prompt']:,} of the {b['main_prompt']:,} prompt tokens the plain client sent ({skipped}% skipped): it reused a cached start of the "
                     "prompt, so the plain client's time is far lower than its token count suggests. This is why fewer tokens sent need not mean less time on a local server.")
    if b["main_calls"] > b["attempts"] or m["main_calls"] > m["attempts"]:
        notes.append(f"Main-model calls were more than the attempts (plain {b['main_calls']} vs {b['attempts']}, Pal {m['main_calls']} vs {m['attempts']}): the extra calls are tool rounds.")
    return {**meta, "baseline": b, "pal": m,
            "savings": {"main_tokens_pct": pct(b["main_tokens"] - m["main_tokens"], b["main_tokens"]), "seconds_pct": pct(b["seconds"] - m["seconds"], b["seconds"]),
                        "main_tokens": b["main_tokens"] - m["main_tokens"], "seconds": round(b["seconds"] - m["seconds"], 2)},
            "tasks": [{"function": r.task.name, "file": f"localrag/{r.task.path}",
                       "baseline": {"ok": r.ok, "attempts": r.attempts, "tokens": r.main_tokens, "seconds": round(r.seconds, 2)},
                       "pal": {"ok": q.ok, "attempts": q.attempts, "tokens": q.main_tokens, "seconds": round(q.seconds, 2)}}
                      for r, q in zip(base.results, mat.results, strict=True)],
            "notes": base.notes + mat.notes + notes}


def suite_text(t: dict, short: bool = False) -> str:
    """The whole-suite result of an arm, or an honest statement that the tests did not run (never a silent 0 / 0)."""
    if t["suite_ran"] is False:
        return "did not run: see test_output.txt"
    return f"{t['suite_passed']} passed / {t['suite_failed']} failed" if short else f"{t['suite_passed']} passed, {t['suite_failed']} failed"


def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else f"{x:,.1f}"


def render_markdown(s: dict) -> str:
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    L = ["# Pal build experiment: a local model writes code", "",
         f"**REAL RUN, no simulation.** Local models in Ollama: main `{s['main_model']}`, helper `{s['helper_model']}`, context window {s['num_ctx']:,} tokens. "
         "Seconds are measured on this machine; token counts are counted from the request text and, for the main model, as the server reported them.", "",
         f"Run {s['started']}, size `{s['size']}`: {b['tasks']} functions to write, up to {s['attempts']} attempts each.", ""]
    for note in s["notes"]:
        L += [f"> **Note.** {note}", ""]
    L += ["## What was done", "",
          f"The real LocalRAG project (a RAG engine and MCP server, with {s['reference_tests']} tests) had {b['tasks']} of its functions replaced by stubs "
          "(signature and docstring kept, body removed). The model had to write each one again, one at a time, from the stub and the rest of the project. "
          "A function counts as **correct** only if LocalRAG's real test suite passes with that one function replaced by the model's version, so the project's "
          "tests are the judge. If a reply could not be used (no code, a syntax error, a changed signature, disallowed code) or the tests failed, the model was "
          "told why and tried again.", "",
          "The plain chat client resends the whole project and the whole conversation every turn. Pal attaches the project once as a code graph and sends only what "
          "each question needs. Both use the same local models.", "",
          "## Result at a glance", "", "| | Plain chat client | Pal | Difference |", "|---|---:|---:|---:|",
          f"| **Functions correct** (project's tests pass) | **{b['correct']} of {b['tasks']}** | **{m['correct']} of {m['tasks']}** | {m['correct'] - b['correct']:+d} |",
          f"| Functions with usable code written | {b['written']} | {m['written']} | |",
          f"| Model attempts (answers asked for) | {b['attempts']} | {m['attempts']} | |",
          f"| Main-model calls actually made (tool rounds included) | {b['main_calls']} | {m['main_calls']} | |",
          f"| **Main-model tokens** (prompt + answer) | {_n(b['main_tokens'])} | {_n(m['main_tokens'])} | **{sv['main_tokens_pct']}% saved** |",
          f"| Tokens read by the main model | {_n(b['main_prompt'])} | {_n(m['main_prompt'])} | {pct(b['main_prompt'] - m['main_prompt'], b['main_prompt'])}% |",
          f"| Tokens written by the main model | {_n(b['main_completion'])} | {_n(m['main_completion'])} | {pct(b['main_completion'] - m['main_completion'], b['main_completion'])}% |",
          f"| Local helper tokens (an extra cost) | 0 | {_n(m['helper_tokens'])} | |",
          f"| Main-model tokens per correct function | {_n(b['tokens_per_correct']) if b['tokens_per_correct'] else 'n/a'} | {_n(m['tokens_per_correct']) if m['tokens_per_correct'] else 'n/a'} | |",
          f"| **Time** (model and pipeline; test runs not counted) (s) | {_n(b['seconds'])} | {_n(m['seconds'])} | **{sv['seconds_pct']}% saved** |"]
    if b["server_prompt"] is not None and m["server_prompt"] is not None:
        L += [f"| Prompt tokens as the server counted them | {_n(b['server_prompt'])} | {_n(m['server_prompt'])} | {pct(b['server_prompt'] - m['server_prompt'], b['server_prompt'])}% |"]
    L += [f"| Whole test suite with the model's code in place | {suite_text(b)} | {suite_text(m)} | |", "",
          "Each function is judged on its own, with the others restored to the original; the last row puts everything the model wrote into one project, where functions "
          "that were not written correctly (or at all) fail the tests that use them.", "", "## Function by function", "",
          "| Function | File | Plain: result | tries | tokens | s | Pal: result | tries | tokens | s |", "|---|---|:--:|--:|--:|--:|:--:|--:|--:|--:|"]
    for t in s["tasks"]:
        x, y = t["baseline"], t["pal"]
        L.append(f"| `{t['function']}` | `{t['file']}` | {'pass' if x['ok'] else 'FAIL'} | {x['attempts']} | {_n(x['tokens'])} | {x['seconds']} | "
                 f"{'pass' if y['ok'] else 'FAIL'} | {y['attempts']} | {_n(y['tokens'])} | {y['seconds']} |")
    L += ["", "## The files it wrote", "",
          f"- `{s['folder']}/baseline/localrag_project/`: the project as the plain client's model left it. The raw replies are in `.../baseline/replies/`.",
          f"- `{s['folder']}/pal/localrag_project/`: the same for Pal.",
          "- `report.md` and `report.json` (this report), in the run folder.", "",
          "Open a function, for example `localrag_project/src/localrag/eval/metrics.py`, to read what the model wrote. A function the model could not write still has its stub.", "",
          "## How far to trust this", "",
          "- **It is one run.** Small models vary, and a difference of a function or two between the arms is within what a second run could change. Run it again to see.",
          "- **The comparison is only fair if quality is comparable,** so the correct-function counts matter as much as the token and time rows. Fewer tokens for fewer correct functions is not a saving.",
          "- **It is a best case for Pal on size:** the plain client must resend a whole project every turn. A client with prompt caching or retrieval would narrow the gap, and Ollama may reuse "
          "a repeated prefix, which makes the plain client's measured time lower than it would otherwise be.",
          "- **The main model was asked directly in both arms** (the task is code, so Pal's router was bypassed); the local helper still rewrote and checked what was typed. "
          "Pal's tools were " + ("ON" if s.get("tools") else "off (use --tools to turn them on)") + ".",
          "- **The model's code was run to test it,** in a temporary copy of the project with a time limit and a filter against file, process and network access. That reduces accidents; it is not a sandbox.", "",
          "## Setup", "", f"- Models: main `{s['main_model']}`, helper `{s['helper_model']}`; answers capped at {s['max_output']} tokens; temperature 0; seed {s['seed']}.", ""]
    return "\n".join(L)


def render_terminal(s: dict, folder: str) -> str:
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    line = "=" * 74
    return "\n".join([line, f"  Pal build experiment (REAL): {b['tasks']} functions, size {s['size']}", line,
                      f"  Functions correct    {b['correct']:>8} of {b['tasks']}  ->  {m['correct']} of {m['tasks']}  (plain client -> Pal)",
                      f"  Main-model tokens    {b['main_tokens']:>10,} -> {m['main_tokens']:>9,}   saved {sv['main_tokens_pct']:>5}%",
                      f"  Time (seconds)       {b['seconds']:>10,.1f} -> {m['seconds']:>9,.1f}   saved {sv['seconds_pct']:>5}%",
                      f"  Whole suite with the model's code:  plain {suite_text(b, True)};  Pal {suite_text(m, True)}",
                      *([f"  Tokens the server actually read:    plain {b['server_prompt']:,} -> Pal {m['server_prompt']:,}   (it skips a repeated prompt start)"]
                        if b["server_prompt"] is not None and m["server_prompt"] is not None else []),
                      "", f"  Files written to: {folder}", f"  Report:           {folder}/report.md", line])


# -- the command ----------------------------------------------------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experiment build", description="A local model, through Ollama, re-implements functions removed from LocalRAG; the project's "
                                "tests judge them. Real models only: there is no simulation.")
    p.add_argument("--size", choices=list(B.BUILD_SIZES), default="small", help="small (4 functions, the default), big (9) or max (12)")
    p.add_argument("--functions", default="", help="comma-separated functions to write instead of the size's list, from: " + ", ".join(t.name for t in B.TASKS))
    p.add_argument("--model", default=DEFAULT_MAIN, help=f"the main local model (default {DEFAULT_MAIN}; a larger code model writes better code)")
    p.add_argument("--helper-model", default=DEFAULT_HELPER, help=f"the small local helper model (default {DEFAULT_HELPER})")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--attempts", type=int, default=2, help="tries per function; after a failure the model is told why (default 2)")
    p.add_argument("--max-output", type=int, default=700, help="most tokens either arm may write in one answer")
    p.add_argument("--num-ctx", type=int, default=0, help="context window; by default as large as the longest prompt needs")
    p.add_argument("--max-ctx", type=int, default=65536, help="the largest context window this run may use")
    p.add_argument("--tools", action="store_true", help="let Pal offer its tools (read_source, find_usages, ...) to the model; off by default, because writing a "
                   "function from its stub does not need them and a small model may waste calls on them")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default=str(REPO / "experiment_output"), help="where the run's folder is created (default: experiment_output in the main directory)")
    p.add_argument("--quiet", action="store_true")
    return p


def choose_tasks(args) -> tuple:
    """The functions to write and the parts of the project to show: the size's, or the ones named with --functions (with the parts that hold them)."""
    cfg = B.BUILD_SIZES[args.size]
    if not args.functions:
        return B.TASKS[:cfg["count"]], cfg["packages"]
    by_name = {t.name: t for t in B.TASKS}
    wanted = [n.strip() for n in args.functions.split(",") if n.strip()]
    unknown = [n for n in wanted if n not in by_name]
    if unknown or not wanted:
        raise ValueError(f"unknown function(s) {', '.join(unknown) or '(none given)'}; choose from: {', '.join(by_name)}")
    tasks = [by_name[n] for n in wanted]
    packages = tuple(dict.fromkeys(("core", *(t.path.split("/")[0] for t in tasks))))
    return tasks, packages


def needed_ctx(files: dict, count: int, max_output: int, attempts: int) -> int:
    tokens = estimate_tokens(project_text(files)) + count * attempts * (450 + max_output) + 300
    return int(math.ceil(tokens * 1.25 / 1024.0) * 1024)


def run(args, say=print) -> tuple:
    """Run both arms for real. Returns (summary, folder) or (None, error message)."""
    try:
        tasks, packages = choose_tasks(args)
    except ValueError as e:
        return None, str(e)
    files = B.stubbed_source(tasks, packages)
    probe = probe_ollama(args.ollama_url, [args.model, args.helper_model])
    notes: list = []
    if probe.error:
        return None, (f"Ollama is not usable at {args.ollama_url} ({probe.error}). This experiment needs real models and does not simulate. "
                      f"Start Ollama with `ollama serve`, then:\n  ollama pull {args.model}\n  ollama pull {args.helper_model}")
    if args.model in probe.missing:
        return None, missing_model_message(probe, args.model, "python run.py experiment build")
    helper_model = args.helper_model
    if args.helper_model in probe.missing:
        helper_model = args.model
        notes.append(f"the helper model {args.helper_model} is not pulled, so the main model {args.model} also did the helper's work (ollama pull {args.helper_model}).")
    ctx = args.num_ctx or needed_ctx(files, len(tasks), args.max_output, args.attempts)
    if ctx > args.max_ctx:
        return None, f"This run needs a context window of {ctx:,} tokens, over the {args.max_ctx:,} allowed (--max-ctx). Use --size small or big, or raise --max-ctx."
    if not B.pytest_available():
        return None, (f"pytest is not installed for this Python ({sys.executable}), and it is what judges the model's code. Install it with:\n"
                      f"  {sys.executable} -m pip install pytest\n(`python run.py experiment build` does this for you.)")
    say("Checking that LocalRAG's own tests pass before anything is changed ...")
    scratch = Path(tempfile.mkdtemp(prefix="pal-build-ref-"))
    try:
        ref = B.run_tests(B.copy_project(scratch / "p"), fail_fast=False)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if not ref.ok or ref.passed < 90:
        return None, f"LocalRAG's tests do not pass here ({ref.passed} passed, {ref.failed} failed), so there is nothing to judge the model by:\n{ref.tail[-600:]}"
    folder = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    say(f"Loading {args.model} and {helper_model} (context {ctx:,} tokens) ...")
    try:
        main_core, helper_core = build_real_core(args.model, args.ollama_url, ctx, args.seed), build_real_core(helper_model, args.ollama_url, ctx, args.seed)
        warm_up(main_core)
        warm_up(helper_core)
    except DependencyMissingError as e:
        return None, f"{e} (this experiment reaches Ollama through LiteLLM)"
    except Exception as e:  # noqa: BLE001
        return None, (f"Could not use Ollama: {type(e).__name__}: {str(e)[:200]}\nIf it says the model was not found, run `ollama list` and pass a model "
                      "you have with --model.")
    b_rec, m_rec = Recorder(), Recorder()
    b_main = RecordingCore(main_core, "main", args.model, b_rec, ctx, args.max_output)
    m_main = RecordingCore(main_core, "main", args.model, m_rec, ctx, args.max_output)
    m_helper = RecordingCore(helper_core, "helper", helper_model, m_rec, ctx)

    def progress(arm, i, n, res):
        if not args.quiet:
            say(f"  [{arm:8s} {i:>2}/{n}] {res.task.name:24s} {'pass' if res.ok else 'FAIL'}  {res.attempts} tr{'y' if res.attempts == 1 else 'ies'}  {res.seconds:6.1f}s  {res.main_tokens:>7,} tokens")
    say(f"Baseline: a plain chat client writes {len(tasks)} functions (it resends the whole project every turn) ...")
    base = run_baseline(tasks, files, b_main, args.attempts, args.max_output, progress)
    say("Pal: the same task through the real pipeline ...")
    mat = run_pal(tasks, files, m_main, m_helper, args.attempts, progress, args.tools)
    say("Writing the files and running the project's tests on them ...")
    write_arm(base, tasks, folder / "baseline")
    write_arm(mat, tasks, folder / "pal")
    meta = {"started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "size": args.size, "main_model": args.model, "helper_model": helper_model, "num_ctx": ctx,
            "max_output": args.max_output, "attempts": args.attempts, "seed": args.seed, "tools": args.tools, "reference_tests": ref.passed, "folder": str(folder), "notes": notes}
    summary = summarize(meta, base, mat)
    (folder / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path(args.out_dir) / "LATEST.txt").write_text(str(folder) + "\n", encoding="utf-8")
    return summary, str(folder)


def main(argv: list | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary, out = run(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    if summary is None:
        print(out, file=sys.stderr)
        return 2
    print()
    print(render_terminal(summary, out))
    return 0
