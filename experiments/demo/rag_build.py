"""The RAG build experiment: a local model builds a small RAG engine, file by file, once through a plain chat client and once through Pal.

    python run.py experiment rag                    (the one-command way)

There is NO simulation: it needs Ollama running with a model pulled, and stops with instructions if it does not find one. The same model
builds the same engine twice, in the same steps. The result that matters is how many tokens each way used, and so how many Pal saved.
The code the model wrote is saved as files in the main directory, and a short check says whether the engine it built actually runs.

SAFETY. The model's code is run once, to check that the engine works. Before that, each file must pass a static filter (no file writes, process or
network access, no eval or exec, only standard-library modules that cannot do harm), and it runs in a temporary copy with a time limit and without
your secrets. That reduces accidents; it is not a sandbox.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from experiments.demo.models import DEFAULT_MAIN, Recorder, RecordingCore, build_real_core, missing_model_message, probe_ollama, warm_up
from experiments.demo.report import p95, pct
from experiments.demo.runner import _turn_seconds
from experiments.demo.scenario import load_docs
from pal.tokens import estimate_tokens
from pal.errors import DependencyMissingError
from pal.llm_wrapper.components.base import LLMSyscall

REPO = Path(__file__).resolve().parents[2]
SYSTEM = "You are an expert Python developer. You write complete, working files and reply with code."

INTRO = ("We are building a small retrieval-augmented generation (RAG) engine in Python, in a folder called rag/, using only the standard library. "
         "Its design reference is the documentation of an existing, much larger RAG engine called LocalRAG; ours is a small version of the same idea. "
         "It is built one file at a time, and every file must work with the ones before it. When it is finished it will behave like this:\n\n"
         "```python\nfrom rag import RAG\nengine = RAG()\nengine.add_text(\"Paris is the capital of France.\")\n"
         "print(engine.ask(\"What is the capital of France?\"))   # a prompt for a language model that contains the best matching passage\n```\n\n")


@dataclass(frozen=True)
class Step:
    file: str
    what: str
    api: str = ""


STEPS = [
    Step("chunker.py", "split text into overlapping chunks, cutting at spaces where possible", "def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]"),
    Step("embedder.py", "turn text into a vector by hashing each lowercase word into one of `dim` slots (use hashlib), counting, and scaling to unit length; plus cosine similarity",
         "def embed(text: str, dim: int = 128) -> list[float]\ndef cosine(a: list[float], b: list[float]) -> float"),
    Step("store.py", "a vector store: add a text (split it with chunk_text, embed each chunk) and search for the best chunks for a query",
         "class VectorStore:\n    def add(self, text: str) -> None\n    def search(self, query: str, k: int = 3) -> list[tuple[float, str]]   # (score, chunk), best first"),
    Step("prompt.py", "build the prompt for a language model: the numbered passages, then the question, with an instruction to answer only from the passages",
         "def build_prompt(question: str, chunks: list[str]) -> str"),
    Step("rag.py", "the engine that ties the files together: it holds a VectorStore, and ask() returns the prompt built from the best matching chunks",
         "class RAG:\n    def add_text(self, text: str) -> None\n    def ask(self, question: str, k: int = 3) -> str"),
    Step("cli.py", "a command line: `python cli.py FILE \"question\"` reads FILE, adds it to a RAG, and prints the answer prompt", ""),
    Step("README.md", "documentation: what the engine is, the files, how to use it from Python and from the command line, with one example", ""),
    Step("test_rag.py", "a self-test using unittest: chunk_text, embed (unit length), VectorStore.search ranks the right chunk first, RAG.ask contains the best passage", ""),
    Step("demo.py", "a demo script that builds a RAG from a built-in five-sentence text and prints the prompts for two questions", ""),
]
SIZES = {"small": 5, "big": 7, "max": 9}
ALLOWED_IMPORTS = {"math", "re", "hashlib", "json", "heapq", "collections", "itertools", "functools", "typing", "dataclasses", "string", "textwrap", "operator",
                   "statistics", "argparse", "pathlib", "sys", "unittest", "__future__", "chunker", "embedder", "store", "prompt", "rag", "cli"}
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}
BANNED_ATTRS = {"unlink", "rmdir", "rmtree", "remove", "removedirs", "rename", "replace", "system", "popen", "kill", "write_text", "write_bytes", "mkdir", "touch",
                "chmod", "symlink_to", "__class__", "__subclasses__", "__bases__", "__globals__", "__builtins__", "__dict__", "__mro__", "__code__"}
SMOKE = ("from rag import RAG\nr = RAG()\nr.add_text('Paris is the capital of France. Berlin is the capital of Germany. Madrid is the capital of Spain.')\n"
         "out = r.ask('What is the capital of France?')\nassert isinstance(out, str) and 'Paris' in out, out\nprint('RAG-OK')\n")


def design_block(docs: str) -> str:
    return f"The design documentation of LocalRAG, for reference:\n\n{docs}\n\n"


def step_message(i: int, n: int, step: Step, docs: str = "") -> str:
    """The message for one step. `docs` is the design documentation, pasted in only for a plain client, in its first message."""
    head = (INTRO + (design_block(docs) if docs else "") if i == 1 else "Continue building the same engine. Use the earlier files exactly as written: the same names and signatures.\n\n")
    api = f"\n\nIt must provide:\n```python\n{step.api}\n```" if step.api else ""
    fence = "markdown" if step.file.endswith(".md") else "python"
    return f"{head}Step {i} of {n}: write the file `{step.file}`. It should {step.what}.{api}\n\nReply with the complete file in one ```{fence} block and nothing else."


# -- reading a reply ------------------------------------------------------------------------------------------------------------------------
_FENCE = re.compile(r"```[\w+-]*[ \t]*\n(.*?)```", re.S)


def extract_file(reply: str, filename: str) -> str | None:
    """The file's content from a reply: the first code block (for a .py file, the first that parses), or a bare reply that looks like the file."""
    blocks = _FENCE.findall(reply)
    if filename.endswith(".md"):
        text = blocks[0] if blocks else reply
        return text.strip() + "\n" if text.strip() else None
    for block in blocks or ([reply] if re.search(r"^\s*(def|class|import|from)\b", reply, re.M) else []):
        try:
            ast.parse(block)
            return block.strip() + "\n"
        except SyntaxError:
            continue
    return None


def unsafe_file_reason(code: str) -> str:
    """Why this file may not be run, or an empty string. A best-effort filter, not a sandbox."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for mod in mods:
            if mod.split(".")[0] not in ALLOWED_IMPORTS:
                return f"importing {mod} is not allowed here"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            return f"calling {node.func.id} is not allowed here"
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            return f"using .{node.attr} is not allowed here"
    return ""


# -- the two ways of running it -------------------------------------------------------------------------------------------------------------
@dataclass
class ArmRag:
    name: str
    recorder: Recorder
    replies: list = field(default_factory=list)       # (step, reply, seconds, turn)
    notes: list = field(default_factory=list)
    folder: str = ""
    written: list = field(default_factory=list)       # files that came out
    problems: dict = field(default_factory=dict)      # file -> why it was not written or not run
    smoke_ran: bool = False
    smoke_ok: bool = False
    smoke_detail: str = ""
    estimates: dict = field(default_factory=dict)     # Pal's own estimate of the tokens each of its parts saved


def run_baseline(steps: list, main: RecordingCore, max_output: int, progress=None, docs: str = "") -> ArmRag:
    arm, messages = ArmRag("baseline", main.rec), [{"role": "system", "content": SYSTEM}]
    for i, step in enumerate(steps, 1):
        main.rec.turn = i
        messages.append({"role": "user", "content": step_message(i, len(steps), step, docs)})
        t0 = time.perf_counter()
        reply = main.chat(LLMSyscall(messages=list(messages), temperature=0, max_tokens=max_output)).text
        wall = time.perf_counter() - t0
        messages.append({"role": "assistant", "content": reply})
        secs = _turn_seconds(main.rec, i, wall)[0]
        arm.replies.append((step, reply, secs, i))
        if progress:
            progress("baseline", i, len(steps), step, secs, main.rec)
    return arm


def run_pal(steps: list, main: RecordingCore, helper: RecordingCore, hot_turns: int, progress=None, docs: str = "") -> ArmRag:
    from pal.chat import ChatService
    from pal.config import load_settings
    from pal.knowledge_graph import KnowledgeGraph

    arm = ArmRag("pal", main.rec)
    home = Path(tempfile.mkdtemp(prefix="pal-rag-svc-"))
    settings = load_settings(env={}, overrides={"storage_dir": str(home), "knowledge_graph": {"dir": str(home / "graph")}, "memory": {"enabled": False},
                                                 "tools": {"enabled": False}, "context": {"hot_turns": hot_turns}, "server": {"session_ttl_minutes": 240}})
    svc = ChatService(settings, local_core=helper, graph=KnowledgeGraph(settings.knowledge_graph))
    svc.api_core = lambda provider, model, key=None: main
    try:
        for i, step in enumerate(steps, 1):
            main.rec.turn = helper.rec.turn = i
            t0 = time.perf_counter()
            attachments = [{"name": "localrag_docs.md", "text": docs, "kind": "document"}] if docs and i == 1 else []
            res = svc.send("rag", "local", main.model, None, step_message(i, len(steps), step), attachments, route="api")
            secs = _turn_seconds(main.rec, i, time.perf_counter() - t0)[0]
            arm.replies.append((step, res.reply, secs, i))
            if progress:
                progress("pal", i, len(steps), step, secs, main.rec)
        arm.estimates = svc.metrics.chat_summary().get("tokens_saved_estimate", {})
    finally:
        svc.close()
    return arm


# -- writing the files and checking the engine ---------------------------------------------------------------------------------------------------
def safe_env(folder: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH"))}
    env.update(PYTHONPATH=str(folder), PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


def write_arm(arm: ArmRag, folder: Path, timeout: int = 30) -> None:
    """Save every file the model wrote into <folder>/rag/, the raw replies into <folder>/replies/, and check that the engine runs."""
    rag = folder / "rag"
    rag.mkdir(parents=True, exist_ok=True)
    (folder / "replies").mkdir(exist_ok=True)
    for n, (step, reply, _, _) in enumerate(arm.replies, 1):
        (folder / "replies" / f"{n:02d}_{step.file}.md").write_text(reply, encoding="utf-8")
        content = extract_file(reply, step.file)
        if content is None:
            arm.problems[step.file] = "no usable code in the reply" if step.file.endswith(".py") else "no text in the reply"
            continue
        (rag / step.file).write_text(content, encoding="utf-8")
        arm.written.append(step.file)
    arm.folder = str(folder)
    needed = ["chunker.py", "embedder.py", "store.py", "prompt.py", "rag.py"]
    missing = [f for f in needed if f not in arm.written]
    if missing:
        arm.smoke_detail = "not run: no usable " + ", ".join(missing)
        return
    for f in needed:
        why = unsafe_file_reason((rag / f).read_text(encoding="utf-8"))
        if why:
            arm.problems[f] = why
            arm.smoke_detail = f"not run: {f}: {why}"
            return
    try:
        done = subprocess.run([sys.executable, "-c", SMOKE], cwd=rag, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=safe_env(rag))
    except subprocess.TimeoutExpired:
        arm.smoke_ran, arm.smoke_detail = True, f"did not finish within {timeout} seconds"
        return
    arm.smoke_ran, arm.smoke_ok = True, "RAG-OK" in done.stdout
    errors = [ln for ln in done.stderr.strip().splitlines() if re.match(r"^[\w.]*(Error|Exception)\b", ln)]       # the line that names the exception, not the end of its message
    arm.smoke_detail = ("the engine answered the test question correctly" if arm.smoke_ok
                        else (errors[-1] if errors else (done.stderr.strip().splitlines() or ["it did not answer correctly"])[-1])[:200])
    (folder / "check_output.txt").write_text(done.stdout + "\n" + done.stderr, encoding="utf-8")


# -- the numbers -----------------------------------------------------------------------------------------------------------------------------
def per_step(arm: ArmRag) -> list:
    out = []
    for step, _, secs, turn in arm.replies:
        main = arm.recorder.of_turn(turn, "main")
        helper = arm.recorder.of_turn(turn, "helper")
        out.append({"file": step.file, "prompt": sum(c.prompt_tokens for c in main), "answer": sum(c.completion_tokens for c in main),
                    "main": sum(c.prompt_tokens + c.completion_tokens for c in main), "helper": sum(c.prompt_tokens + c.completion_tokens for c in helper), "seconds": round(secs, 2)})
    return out


def totals(arm: ArmRag) -> dict:
    steps = per_step(arm)
    main = [c for c in arm.recorder.calls if c.role == "main"]
    return {"main_tokens": sum(s["main"] for s in steps), "prompt": sum(s["prompt"] for s in steps), "answer": sum(s["answer"] for s in steps),
            "helper_tokens": sum(s["helper"] for s in steps), "seconds": round(sum(s["seconds"] for s in steps), 2), "calls": len(main),
            "server_prompt": sum(c.server_prompt for c in main) if main and all(c.server_prompt is not None for c in main) else None,
            "files": len(arm.written), "smoke_ran": arm.smoke_ran, "smoke_ok": arm.smoke_ok, "smoke_detail": arm.smoke_detail, "truncated": sum(1 for c in arm.recorder.calls if c.truncated)}


def summarize(meta: dict, base: ArmRag, mat: ArmRag) -> dict:
    b, m = totals(base), totals(mat)
    all_b, all_m = b["main_tokens"] + b["helper_tokens"], m["main_tokens"] + m["helper_tokens"]
    notes = list(meta.get("notes", [])) + base.notes + mat.notes
    if b["server_prompt"] is not None and b["prompt"] and b["server_prompt"] < 0.6 * b["prompt"]:
        notes.append(f"Ollama read only {b['server_prompt']:,} of the {b['prompt']:,} prompt tokens the plain client sent: it reused a cached start of the prompt, "
                     "so the plain client's time is lower than its token count suggests.")
    return {**meta, "baseline": b, "pal": m, "steps": {"baseline": per_step(base), "pal": per_step(mat)}, "estimates": mat.estimates,
            "savings": {"main_tokens": b["main_tokens"] - m["main_tokens"], "main_tokens_pct": pct(b["main_tokens"] - m["main_tokens"], b["main_tokens"]),
                        "all_tokens_pct": pct(all_b - all_m, all_b), "prompt_pct": pct(b["prompt"] - m["prompt"], b["prompt"]),
                        "seconds": round(b["seconds"] - m["seconds"], 2), "seconds_pct": pct(b["seconds"] - m["seconds"], b["seconds"])},
            "notes": notes}


def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else f"{x:,.1f}"


def verdict(t: dict) -> str:
    if not t["smoke_ran"]:
        return "not run (" + t["smoke_detail"].replace("not run: ", "") + ")"
    return "yes" if t["smoke_ok"] else "no (" + t["smoke_detail"] + ")"


def render_markdown(s: dict) -> str:
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    L = ["# Pal RAG build experiment: tokens saved while a local model builds a RAG engine", "",
         f"**REAL RUN, no simulation.** Local model in Ollama: `{s['main_model']}`" + (f", helper `{s['helper_model']}`" if s["helper_model"] != s["main_model"] else " (also used as Pal's helper)")
         + f", context window {s['num_ctx']:,} tokens. Seconds are measured on this machine.", "",
         f"Run {s['started']}, size `{s['size']}`: the model builds the engine in {s['steps_count']} steps (one file each).", ""]
    for note in s["notes"]:
        L += [f"> **Note.** {note}", ""]
    L += ["## The result", "",
          f"### Without Pal: {_n(b['main_tokens'])} tokens.  With Pal: {_n(m['main_tokens'])} tokens.  **Saved: {_n(sv['main_tokens'])} tokens ({sv['main_tokens_pct']}%).**", "",
          "| | Plain chat client | Pal | Difference |", "|---|---:|---:|---:|",
          f"| **Main-model tokens** (what you would pay a model API for) | {_n(b['main_tokens'])} | {_n(m['main_tokens'])} | **{sv['main_tokens_pct']}% saved** |",
          f"| Tokens read by the model (the prompts) | {_n(b['prompt'])} | {_n(m['prompt'])} | {sv['prompt_pct']}% saved |",
          f"| Tokens written by the model (the code) | {_n(b['answer'])} | {_n(m['answer'])} | {pct(b['answer'] - m['answer'], b['answer'])}% |",
          f"| Helper tokens (Pal's own small jobs, an extra cost) | 0 | {_n(m['helper_tokens'])} | |",
          f"| All tokens, main model plus helper | {_n(b['main_tokens'])} | {_n(m['main_tokens'] + m['helper_tokens'])} | {sv['all_tokens_pct']}% saved |",
          f"| Time (s) | {_n(b['seconds'])} | {_n(m['seconds'])} | {sv['seconds_pct']}% {'saved' if sv['seconds'] >= 0 else 'slower'} |",
          f"| Files the model produced | {b['files']} of {s['steps_count']} | {m['files']} of {s['steps_count']} | |",
          f"| The engine it built runs and answers a test question | {verdict(b)} | {verdict(m)} | |"]
    if b["server_prompt"] is not None and m["server_prompt"] is not None:
        L += [f"| Prompt tokens as the server counted them | {_n(b['server_prompt'])} | {_n(m['server_prompt'])} | {pct(b['server_prompt'] - m['server_prompt'], b['server_prompt'])}% |"]
    spec = (f"The model was given the design documentation of LocalRAG ({_n(s['spec_tokens'])} tokens) as its reference. The plain client pastes it into its first message and so "
            "resends it at every step; Pal attaches it once and sends only the parts that match each step. " if s["spec_tokens"] else
            "No design document was attached in this run, so Pal could only save on the conversation history, which needs a long build to pay off. ")
    L += ["", "## Step by step: where the saving comes from", "",
          spec + "A plain client also sends the whole conversation every step, so its prompt grows with every file already written. Pal keeps the latest steps in full and "
          "shortens older ones (code is cut to its signatures).", "",
          "| Step | File | Plain: prompt | Plain: total | Pal: prompt | Pal: total | Saved |", "|--:|---|--:|--:|--:|--:|--:|"]
    for i, (x, y) in enumerate(zip(s["steps"]["baseline"], s["steps"]["pal"], strict=True), 1):
        L.append(f"| {i} | `{x['file']}` | {_n(x['prompt'])} | {_n(x['main'])} | {_n(y['prompt'])} | {_n(y['main'])} | {pct(x['main'] - y['main'], x['main'])}% |")
    est = s.get("estimates") or {}
    if est:
        L += ["", "Pal's own estimate of where it saved tokens (each part's report about itself, not a measurement; the table above is the measured result):", "",
              "| Part | Tokens it reports saving |", "|---|--:|"]
        names = {"documents": "Design-document excerpts instead of the whole document", "context_paging": "History paging", "input_savers": "Trimming what was typed"}
        L += [f"| {names[k]} | {_n(est[k])} |" for k in names if k in est]
    L += ["", "## The files it wrote", "",
          f"- `{s['folder']}/baseline/rag/`: the engine built through the plain chat client.", f"- `{s['folder']}/pal/rag/`: the engine built through Pal.",
          "- `replies/` in each: every raw reply. `check_output.txt`: what happened when the engine was run.", "",
          "Open `rag/rag.py` in either folder to read what the model wrote. A file the model did not produce is simply missing.", "",
          "## How far to trust this", "",
          "- **The headline is tokens, and tokens are counted from the request text** (characters divided by 4) the same way for both, plus the server's own counts where it gives them.",
          "- **It is one run,** and a small model varies. Run it again before relying on the exact percentage.",
          f"- **Pal used its default settings** (it keeps the last {s['hot_turns']} steps in full). Without a design document, a build this short saves little or nothing, because the history "
          "is small and Pal adds a fixed answer-style rule to every request; the saving here comes mostly from not resending the document. Run with `--no-spec` to see that for yourself.",
          "- **The model's code quality is a side result.** A small model may write an engine that does not run; the tokens are still counted.",
          "- **Ollama may reuse a repeated prompt start,** which makes the plain client's *time* lower than its token count suggests; the server's token line above shows how much.",
          "- **The model's code was run once** (the check above), after a filter against file writes, processes, network and eval, in a temporary copy. Not a sandbox.", "",
          "## Setup", "", f"- Model: `{s['main_model']}`; answers capped at {s['max_output']} tokens; temperature 0; seed {s['seed']}; Pal tools off; the main model was asked directly in both arms.", ""]
    return "\n".join(L)


def render_terminal(s: dict, folder: str) -> str:
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    line = "=" * 74
    return "\n".join([line, f"  Pal RAG build (REAL): the model builds a small RAG engine in {s['steps_count']} steps", line,
                      f"  WITHOUT Pal (plain chat client):  {b['main_tokens']:>9,} tokens",
                      f"  WITH Pal:                         {m['main_tokens']:>9,} tokens",
                      f"  TOKENS SAVED:                           {sv['main_tokens']:>9,}  ({sv['main_tokens_pct']}%)",
                      "", f"  Pal's helper used {m['helper_tokens']:,} more tokens; counting them, the saving is {sv['all_tokens_pct']}%.",
                      f"  Time: plain {b['seconds']:,.1f} s, Pal {m['seconds']:,.1f} s ({sv['seconds_pct']}% {'saved' if sv['seconds'] >= 0 else 'slower'})",
                      f"  The engine built runs and answers a test question:  plain {verdict(b)};  Pal {verdict(m)}",
                      "", f"  Files written to: {folder}", f"  Report:           {folder}/report.md", line])


# -- the command ---------------------------------------------------------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experiment rag", description="A local model, through Ollama, builds a small RAG engine twice (plain chat client, then Pal) and "
                                "the report shows the tokens saved. Real models only: there is no simulation.")
    p.add_argument("--size", choices=list(SIZES), default="big", help="small (5 files), big (7, the default) or max (9); a longer build saves a larger share")
    p.add_argument("--model", default=DEFAULT_MAIN, help=f"the local model that builds the engine (default {DEFAULT_MAIN}; a larger code model writes better code)")
    p.add_argument("--helper-model", default="", help="Pal's small helper model; by default the same model, so Ollama never has to swap models in and out of memory")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--hot-turns", type=int, default=6, help="how many of the latest steps Pal keeps in full (6 is its default)")
    p.add_argument("--no-spec", action="store_true", help="do not give the model the design documentation: only the conversation history is left to save on, which needs a long build")
    p.add_argument("--max-output", type=int, default=900, help="most tokens either arm may write in one answer")
    p.add_argument("--num-ctx", type=int, default=0, help="context window; by default as large as the longest prompt needs")
    p.add_argument("--max-ctx", type=int, default=65536, help="the largest context window this run may use")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default=str(REPO / "experiment_output"), help="where the run's folder is created (default: experiment_output in the main directory)")
    p.add_argument("--quiet", action="store_true")
    return p


def needed_ctx(n_steps: int, max_output: int, spec_tokens: int = 0) -> int:
    return int(math.ceil((spec_tokens + n_steps * (500 + max_output) + 800) * 1.3 / 1024.0) * 1024)


def run(args, say=print) -> tuple:
    """Run both ways for real. Returns (summary, folder) or (None, error message)."""
    steps = STEPS[:SIZES[args.size]]
    docs = "" if args.no_spec else load_docs()
    spec_tokens = estimate_tokens(docs)
    helper_model = args.helper_model or args.model
    probe = probe_ollama(args.ollama_url, sorted({args.model, helper_model}))
    notes: list = []
    if probe.error:
        return None, (f"Ollama is not usable at {args.ollama_url} ({probe.error}). This experiment needs a real model and does not simulate. "
                      f"Start Ollama with `ollama serve`, then:\n  ollama pull {args.model}")
    if args.model in probe.missing:
        return None, missing_model_message(probe, args.model, "python run.py experiment rag")
    if helper_model in probe.missing:
        notes.append(f"the helper model {helper_model} is not pulled, so {args.model} also did the helper's work.")
        helper_model = args.model
    ctx = args.num_ctx or needed_ctx(len(steps), args.max_output, spec_tokens)
    if ctx > args.max_ctx:
        return None, f"This run needs a context window of {ctx:,} tokens, over the {args.max_ctx:,} allowed (--max-ctx). Use --size small, or raise --max-ctx."
    folder = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    say(f"Loading {args.model}" + (f" and {helper_model}" if helper_model != args.model else "") + f" (context {ctx:,} tokens) ...")
    try:
        main_core = build_real_core(args.model, args.ollama_url, ctx, args.seed)
        helper_core = main_core if helper_model == args.model else build_real_core(helper_model, args.ollama_url, ctx, args.seed)
        warm_up(main_core)
        if helper_core is not main_core:
            warm_up(helper_core)
    except DependencyMissingError as e:
        return None, f"{e} (this experiment reaches Ollama through LiteLLM)"
    except Exception as e:  # noqa: BLE001
        return None, (f"Could not use Ollama: {type(e).__name__}: {str(e)[:200]}\nIf it says the model was not found, run `ollama list` and pass a model you have with --model.")
    folder.mkdir(parents=True, exist_ok=True)
    b_rec, m_rec = Recorder(), Recorder()
    b_main = RecordingCore(main_core, "main", args.model, b_rec, ctx, args.max_output)
    m_main = RecordingCore(main_core, "main", args.model, m_rec, ctx, args.max_output)
    m_helper = RecordingCore(helper_core, "helper", helper_model, m_rec, ctx)

    def progress(arm, i, n, step, secs, rec):
        if not args.quiet:
            tokens = sum(c.prompt_tokens + c.completion_tokens for c in rec.of_turn(i, "main"))
            say(f"  [{arm:8s} {i}/{n}] {step.file:14s} {secs:6.1f}s  {tokens:>7,} tokens")
    say(f"WITHOUT Pal: a plain chat client builds the engine in {len(steps)} steps (it resends the whole conversation every step) ...")
    base = run_baseline(steps, b_main, args.max_output, progress, docs)
    say("WITH Pal: the same build through the real pipeline ...")
    mat = run_pal(steps, m_main, m_helper, args.hot_turns, progress, docs)
    say("Writing the files and checking that each engine runs ...")
    write_arm(base, folder / "baseline")
    write_arm(mat, folder / "pal")
    meta = {"started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "size": args.size, "steps_count": len(steps), "main_model": args.model, "helper_model": helper_model,
            "num_ctx": ctx, "max_output": args.max_output, "seed": args.seed, "hot_turns": args.hot_turns, "spec_tokens": spec_tokens, "folder": str(folder), "notes": notes}
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
