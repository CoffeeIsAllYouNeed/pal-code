"""Turns the two runs into numbers and a readable report: totals, savings, a per-kind breakdown, where the savings came from, and the
caveats that decide how far the numbers can be trusted."""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from experiments.demo.models import SIM_CALL_OVERHEAD, SIM_SPEED
from experiments.demo.runner import ArmResult
from experiments.demo.scenario import Scenario
from pal.tokens import estimate_tokens


@dataclass
class Experiment:
    scenario: Scenario
    mode: str                              # "real" or "simulated"
    main_model: str
    helper_model: str
    num_ctx: int
    baseline: ArmResult
    pal: ArmResult
    max_output: int
    notes: list = field(default_factory=list)      # things the reader must know (a missing model, a skipped part, ...)
    started: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def pct(saved: float, base: float) -> float:
    return round(100.0 * saved / base, 1) if base else 0.0


def p95(values: list) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, math.ceil(0.95 * len(v)) - 1)]


def arm_stats(arm: ArmResult) -> dict:
    calls, outs = arm.recorder.calls, arm.outcomes
    main = [c for c in calls if c.role == "main"]
    helper = [c for c in calls if c.role == "helper"]
    scored = [o for o in outs if o.ok is not None]
    server = [c.server_prompt for c in main]
    return {
        "turns": len(outs), "main_calls": len(main), "helper_calls": len(helper),
        "main_prompt": sum(c.prompt_tokens for c in main), "main_completion": sum(c.completion_tokens for c in main),
        "main_total": sum(c.prompt_tokens + c.completion_tokens for c in main),
        "helper_total": sum(c.prompt_tokens + c.completion_tokens for c in helper),
        "peak_prompt": max((c.prompt_tokens for c in main), default=0),
        "server_prompt": sum(server) if server and all(s is not None for s in server) else None,
        "seconds": round(sum(o.seconds for o in outs), 2), "seconds_mean": round(sum(o.seconds for o in outs) / max(1, len(outs)), 2),
        "seconds_p95": round(p95([o.seconds for o in outs]), 2), "overhead": round(sum(o.overhead for o in outs), 2),
        "ok": sum(1 for o in scored if o.ok), "scored": len(scored), "truncated": sum(1 for c in calls if c.truncated),
    }


def by_kind(exp: Experiment) -> list:
    rows = []
    for kind in sorted({t.kind for t in exp.scenario.turns}):
        idx = {t.i for t in exp.scenario.turns if t.kind == kind}
        def tok(arm):
            return sum(c.prompt_tokens + c.completion_tokens for c in arm.recorder.calls if c.role == "main" and c.turn in idx)
        def sec(arm):
            return round(sum(o.seconds for o in arm.outcomes if o.turn in idx), 2)
        b_t, m_t, b_s, m_s = tok(exp.baseline), tok(exp.pal), sec(exp.baseline), sec(exp.pal)
        rows.append({"kind": kind, "turns": len(idx), "baseline_tokens": b_t, "pal_tokens": m_t, "tokens_saved_pct": pct(b_t - m_t, b_t),
                     "baseline_seconds": b_s, "pal_seconds": m_s, "seconds_saved_pct": pct(b_s - m_s, b_s)})
    return rows


def summarize(exp: Experiment) -> dict:
    b, m = arm_stats(exp.baseline), arm_stats(exp.pal)
    all_b, all_m = b["main_total"] + b["helper_total"], m["main_total"] + m["helper_total"]
    handled = {}
    for o in exp.pal.outcomes:
        handled[o.handled_by] = handled.get(o.handled_by, 0) + 1
    return {
        "mode": exp.mode, "started": exp.started, "size": exp.scenario.size, "seed": exp.scenario.seed,
        "models": {"main": exp.main_model, "helper": exp.helper_model, "num_ctx": exp.num_ctx, "max_output": exp.max_output},
        "scenario": {"turns": len(exp.scenario.turns), "kinds": exp.scenario.kinds(), "docs_tokens": estimate_tokens(exp.scenario.document),
                     "project_tokens": estimate_tokens(exp.scenario.project_text()) if exp.scenario.include_code else 0,
                     "project_files": len(exp.scenario.project_files), "project_lines": exp.scenario.project_lines(),
                     "project": exp.scenario.project_name, "includes_code": exp.scenario.include_code},
        "baseline": b, "pal": m,
        "savings": {
            "main_tokens": b["main_total"] - m["main_total"], "main_tokens_pct": pct(b["main_total"] - m["main_total"], b["main_total"]),
            "all_tokens": all_b - all_m, "all_tokens_pct": pct(all_b - all_m, all_b),
            "main_calls_avoided": b["main_calls"] - m["main_calls"],
            "seconds": round(b["seconds"] - m["seconds"], 2), "seconds_pct": pct(b["seconds"] - m["seconds"], b["seconds"]),
            "peak_prompt_pct": pct(b["peak_prompt"] - m["peak_prompt"], b["peak_prompt"]),
        },
        "handled_by": handled, "by_kind": by_kind(exp),
        "pal_estimates": (exp.pal.notes.get("metrics") or {}).get("tokens_saved_estimate", {}),
        "sim_speed": {"main": SIM_SPEED["main"], "helper": SIM_SPEED["helper"], "call_overhead": SIM_CALL_OVERHEAD} if exp.mode == "simulated" else None,
        "notes": exp.notes,
    }


# -- rendering ---------------------------------------------------------------------------------------------------------------------------
def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else f"{x:,.1f}"


def banner(s: dict) -> str:
    m = s["models"]
    if s["mode"] == "simulated":
        return ("**SIMULATED RUN.** Ollama was not used. The prompts Pal builds, and so the tokens it sends, are real. The seconds are "
                f"computed from an assumed speed (main model: reads {SIM_SPEED['main'][0]:.0f} and writes {SIM_SPEED['main'][1]:.0f} tokens a second; "
                f"helper: {SIM_SPEED['helper'][0]:.0f} and {SIM_SPEED['helper'][1]:.0f}), not measured on any machine, and the answers are scripted, "
                "so answer quality is not measured. Run with Ollama running for a real measurement.")
    return (f"**REAL RUN.** Local models in Ollama: main `{m['main']}`, helper `{m['helper']}`, context window {m['num_ctx']:,} tokens. "
            "Seconds are measured on this machine.")


def render_markdown(s: dict) -> str:
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    sc = s["scenario"]
    L = ["# Pal demo experiment: tokens and latency saved", "", banner(s), "",
         f"Run {s['started']}, size `{s['size']}`, seed {s['seed']}.", ""]
    for note in s["notes"]:
        L += [f"> **Note.** {note}", ""]
    L += ["## Result at a glance", "",
          "| | Plain chat client | Pal | Saved |", "|---|---:|---:|---:|",
          f"| **Main-model tokens** (prompt + answer) | {_n(b['main_total'])} | {_n(m['main_total'])} | **{_n(sv['main_tokens'])} ({sv['main_tokens_pct']}%)** |",
          f"| Tokens read by the main model | {_n(b['main_prompt'])} | {_n(m['main_prompt'])} | {pct(b['main_prompt'] - m['main_prompt'], b['main_prompt'])}% |",
          f"| Tokens written by the main model | {_n(b['main_completion'])} | {_n(m['main_completion'])} | {pct(b['main_completion'] - m['main_completion'], b['main_completion'])}% |",
          f"| Local helper tokens (route, rewrite, check, easy answers) | 0 | {_n(m['helper_total'])} | (an extra cost) |",
          f"| All tokens, main + helper | {_n(b['main_total'] + b['helper_total'])} | {_n(m['main_total'] + m['helper_total'])} | {sv['all_tokens_pct']}% |",
          f"| Main-model calls | {b['main_calls']} | {m['main_calls']} | {sv['main_calls_avoided']} avoided |",
          f"| Largest prompt sent | {_n(b['peak_prompt'])} | {_n(m['peak_prompt'])} | {sv['peak_prompt_pct']}% |",
          f"| **Latency, whole session** (s) | {_n(b['seconds'])} | {_n(m['seconds'])} | **{_n(sv['seconds'])} s ({sv['seconds_pct']}%)** |",
          f"| Latency per turn, mean (s) | {b['seconds_mean']} | {m['seconds_mean']} | {pct(b['seconds_mean'] - m['seconds_mean'], b['seconds_mean'])}% |",
          f"| Latency per turn, 95th percentile (s) | {b['seconds_p95']} | {m['seconds_p95']} | |",
          f"| of which time outside the models (s) | {b['overhead']} | {m['overhead']} | |"]
    if s["mode"] == "real":
        L += [f"| Answers containing the expected fact | {b['ok']} of {b['scored']} | {m['ok']} of {m['scored']} | |"]
        if b["server_prompt"] is not None and m["server_prompt"] is not None:
            L += [f"| Prompt tokens as the server counted them | {_n(b['server_prompt'])} | {_n(m['server_prompt'])} | {pct(b['server_prompt'] - m['server_prompt'], b['server_prompt'])}% |"]
    lead = "In this SIMULATED run" if s["mode"] == "simulated" else "In this run"
    L += ["", f"In plain words: {lead}, Pal used **{sv['main_tokens_pct']}% fewer tokens on the main model** and finished the session "
          f"**{sv['seconds_pct']}% faster**, while avoiding {sv['main_calls_avoided']} of {b['main_calls']} main-model calls.", "",
          "## What was run", "",
          f"A {sc['turns']}-turn coding-assistant session over **LocalRAG**, a real Python project written for this experiment: a retrieval-augmented "
          "generation (RAG) engine that is also a Model Context Protocol (MCP) server, with its own documentation and test suite. Its documentation "
          f"(about {_n(sc['docs_tokens'])} tokens) is attached in turn 1"
          + (f" and its source (about {_n(sc['project_tokens'])} tokens, {sc['project_files']} files, {_n(sc['project_lines'])} lines) part-way. " if sc["includes_code"]
             else ". The source was left out (graphify is not installed). ")
          + "Turns by kind: " + ", ".join(f"{v} {k}" for k, v in sorted(sc["kinds"].items())) + ". The questions are the kind a developer asks an assistant: which "
          "function does this, what does it take, what does a setting default to, who calls it, what does the guide say. They were generated from the "
          "real files, so every expected answer can be checked. Every message is written with greetings and thanks, as people type, and some turns "
          "are easy questions (arithmetic, the time, small facts). The plain client resends the documentation, the source and the whole conversation on "
          "every turn; Pal is the real chat pipeline (input savers, task router, local answers, built-in tools, document and code-graph excerpts, "
          "history paging and compression).", "",
          "## How each turn was answered in Pal", ""]
    names = {"tool": "answered by a built-in tool, no model at all", "local": "answered by the local helper alone",
             "api": "answered by the main model with a trimmed prompt"}
    for k in ("tool", "local", "api"):
        n = s["handled_by"].get(k, 0)
        L.append(f"- {n} {'turn' if n == 1 else 'turns'} {names[k]}")
    L += ["", "## By kind of turn", "", "| Kind | Turns | Baseline tokens | Pal tokens | Tokens saved | Baseline s | Pal s | Time saved |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["by_kind"]:
        L.append(f"| {r['kind']} | {r['turns']} | {_n(r['baseline_tokens'])} | {_n(r['pal_tokens'])} | {r['tokens_saved_pct']}% | "
                 f"{r['baseline_seconds']} | {r['pal_seconds']} | {r['seconds_saved_pct']}% |")
    est = s["pal_estimates"]
    if est:
        L += ["", "## Where the savings came from (Pal's own estimates)", "",
              "These are the estimates each part reports about itself, not measurements; the table above is the measured result.", "",
              "| Part | Tokens it reports saving |", "|---|---:|"]
        labels = {"input_savers": "Input savers (trimming what you type)", "context_paging": "History paging and compression",
                  "code_graph": "Code graph excerpts instead of the whole project", "documents": "Document excerpts instead of all the documentation"}
        for key, label in labels.items():
            if key in est:
                L.append(f"| {label} | {_n(est[key])} |")
    L += ["", "## How far to trust this", "",
          "- **It is a best case on purpose.** The session is built so that Pal has the most to save: a real project's documentation and source that a "
          "plain client must resend every turn, many easy questions, and wordy messages. A typical chat saves less.",
          "- **The baseline is deliberately naive:** no trimming, no retrieval, no tools, no routing, every turn sends everything. A client "
          "with retrieval or good prompt caching would narrow the gap.",
          "- **Tokens** are counted from the request text (characters divided by 4), the same way for both arms, so the comparison does not depend "
          "on a server. A real run also lists the server's own counts, which can leave out a prefix the model had already seen.",
          "- **Pal's helper is not free:** its tokens and its time are shown and included in the all-tokens and latency rows.",
          "- **Latency includes everything Pal does** around the models, including building the code graph."]
    if s["mode"] == "simulated":
        L += ["- **This was simulated:** see the banner. The seconds follow from the assumed speed and from how many tokens each arm sends, which is "
              "the real part. Treat them as a demonstration of the mechanism, not as a measurement of your hardware."]
    else:
        L += ["- **The answer check is light:** an answer counts if it contains the expected fact. It is not a quality benchmark, and small models "
              "can miss in either arm.",
              "- **Ollama may reuse a repeated prefix** (a prompt cache). If it did, the plain client's measured time is lower than it would be "
              "without one, so the latency saving shown is on the cautious side.",
              "- **Models were loaded and warmed up before anything was timed,** and a prompt that filled the context window would be flagged "
              "(truncated calls: " + f"{b['truncated']} baseline, {m['truncated']} Pal)."]
    L += ["", "## Setup", "", f"- Main model: `{s['models']['main']}`; helper model: `{s['models']['helper']}`; answers capped at {s['models']['max_output']} tokens in both arms.",
          f"- Context window: {s['models']['num_ctx']:,} tokens. Mode: {s['mode']}.", "- Pal ran with its default settings (memory off, nothing persisted).", ""]
    return "\n".join(L)


def render_terminal(s: dict, paths: dict | None = None) -> str:
    b, m, sv = s["baseline"], s["pal"], s["savings"]
    line = "=" * 74
    out = [line, f"  Pal demo: {s['mode'].upper()} run, {s['scenario']['turns']} turns, size {s['size']}", line]
    if s["mode"] == "simulated":
        out.append("  SIMULATED: tokens are real, seconds are computed from an assumed speed, answers are scripted.")
    out += [f"  Main-model tokens   {b['main_total']:>10,} -> {m['main_total']:>9,}   saved {sv['main_tokens_pct']:>5}%",
            f"  All tokens (+helper){b['main_total'] + b['helper_total']:>10,} -> {m['main_total'] + m['helper_total']:>9,}   saved {sv['all_tokens_pct']:>5}%",
            f"  Main-model calls    {b['main_calls']:>10} -> {m['main_calls']:>9}   avoided {sv['main_calls_avoided']}",
            f"  Latency (seconds)   {b['seconds']:>10,.1f} -> {m['seconds']:>9,.1f}   saved {sv['seconds_pct']:>5}%"]
    if s["mode"] == "real":
        out.append(f"  Answers with the expected fact: baseline {b['ok']}/{b['scored']}, Pal {m['ok']}/{m['scored']}")
    out.append("  A best-case workload by design: see the report for what that means.")
    if paths:
        out += ["", f"  Report: {paths['md']}", f"  Data:   {paths['json']}  and  {paths['csv']}"]
    out.append(line)
    return "\n".join(out)


def turns_csv(exp: Experiment) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["turn", "kind", "handled_by", "baseline_main_tokens", "pal_main_tokens", "pal_helper_tokens", "baseline_seconds",
                "pal_seconds", "baseline_ok", "pal_ok"])
    for t in exp.scenario.turns:
        def tok(arm, role):
            return sum(c.prompt_tokens + c.completion_tokens for c in arm.recorder.calls if c.turn == t.i and c.role == role)
        bo, mo = exp.baseline.outcomes[t.i - 1], exp.pal.outcomes[t.i - 1]
        w.writerow([t.i, t.kind, mo.handled_by, tok(exp.baseline, "main"), tok(exp.pal, "main"), tok(exp.pal, "helper"),
                    round(bo.seconds, 3), round(mo.seconds, 3), "" if bo.ok is None else int(bo.ok), "" if mo.ok is None else int(mo.ok)])
    return buf.getvalue()


def write_all(exp: Experiment, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s = summarize(exp)
    paths = {"md": out / f"demo-{stamp}.md", "json": out / f"demo-{stamp}.json", "csv": out / f"demo-{stamp}-turns.csv"}
    paths["md"].write_text(render_markdown(s), encoding="utf-8")
    paths["json"].write_text(json.dumps(s, indent=2), encoding="utf-8")
    paths["csv"].write_text(turns_csv(exp), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}
