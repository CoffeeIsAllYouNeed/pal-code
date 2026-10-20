"""One command: build the session, run it both ways, write the report.

    python run.py experiment                 (sets everything up, then runs; the one-command way)
    python experiments/run_demo.py           (the same, if Pal is already installed)

With Ollama running and the models pulled it is a REAL run on your machine. Without it, the experiment still runs, as a clearly
labelled SIMULATION, so the mechanism can be seen anywhere. Not part of the Pal package: nothing in `src/` imports this.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from experiments.demo import report, runner
from experiments.demo.models import (
    DEFAULT_HELPER, DEFAULT_MAIN, Recorder, RecordingCore, SimModel, SimScript, build_real_core, probe_ollama, warm_up,
)
from experiments.demo.scenario import SIZES, build
from pal.errors import DependencyMissingError

MAX_CTX = 65536
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experiment", description="Run the Pal demo experiment and report the tokens and latency saved.")
    p.add_argument("--size", choices=list(SIZES), default="big", help="small (quick), big (default) or max (the largest session)")
    p.add_argument("--seed", type=int, default=7, help="the session is generated from this, so the same seed gives the same session")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="require Ollama (an error if it is not available)")
    mode.add_argument("--simulate", action="store_true", help="do not use Ollama: scripted models with an assumed speed")
    p.add_argument("--model", default=DEFAULT_MAIN, help=f"the main local model (default {DEFAULT_MAIN})")
    p.add_argument("--helper-model", default=DEFAULT_HELPER, help=f"the small local helper model (default {DEFAULT_HELPER})")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--num-ctx", type=int, default=0, help="context window for the models; by default as large as the longest prompt needs")
    p.add_argument("--max-output", type=int, default=300, help="most tokens either arm may write in one answer")
    p.add_argument("--max-ctx", type=int, default=MAX_CTX, help="the largest context window a real run may use; raise it for --size max if your hardware allows")
    p.add_argument("--no-code", action="store_true", help="leave out the code-project part")
    p.add_argument("--out", default=str(REPORT_DIR), help="where the report is written")
    p.add_argument("--quiet", action="store_true")
    return p


def needed_ctx(scenario, max_output: int) -> int:
    peak = runner.baseline_peak_tokens(scenario, reply_tokens=max_output)
    return int(math.ceil((peak * 1.25 + max_output) / 1024.0) * 1024)


def _graph_available() -> bool:
    try:
        from pal.config import load_settings
        from pal.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph(load_settings(env={}).knowledge_graph).available()
    except Exception:  # noqa: BLE001
        return False


def run(args, say=print):
    """Run the experiment; returns (Experiment, None) or (None, error message)."""
    notes: list = []
    include_code = not args.no_code and _graph_available()
    if not args.no_code and not include_code:
        notes.append("graphify is not installed, so the code-project part of the session was left out (pip install graphifyy to include it).")
    sc = build(args.size, args.seed, include_code)
    ctx = args.num_ctx or needed_ctx(sc, args.max_output)
    mode, main_model, helper_model = "simulated", args.model, args.helper_model

    if not args.simulate:
        probe = probe_ollama(args.ollama_url, [args.model, args.helper_model])
        only_helper_missing = probe.missing == [args.helper_model] and not probe.error
        if probe.ok or only_helper_missing:
            mode = "real"
            if only_helper_missing:
                helper_model = args.model
                notes.append(f"the helper model {args.helper_model} is not pulled, so the main model {args.model} also did the helper's work "
                             f"(ollama pull {args.helper_model} for the intended setup).")
        else:
            why = probe.error or "missing models: " + ", ".join(probe.missing)
            pulls = "".join(f"\n  ollama pull {m}" for m in (probe.missing or [args.model, args.helper_model]))
            if probe.available:
                pulls += f"\nModels you have: {', '.join(probe.available)} (use --model NAME to run with one of them)"
            if args.real:
                return None, f"Ollama is not usable at {args.ollama_url} ({why}). Start it with `ollama serve`, then:{pulls}"
            say(f"Ollama is not usable at {args.ollama_url} ({why}): running the SIMULATED experiment instead. "
                f"For a real run: `ollama serve`, then:{pulls}")
    if mode == "real" and ctx > args.max_ctx:
        return None, (f"This session needs a context window of {ctx:,} tokens, over the {args.max_ctx:,} allowed (--max-ctx). Use --size small or big, "
                      "or raise --max-ctx if your hardware has the memory for it.")
    if mode == "real" and args.num_ctx and args.num_ctx < needed_ctx(sc, args.max_output):
        notes.append(f"--num-ctx {args.num_ctx:,} is smaller than the longest prompt needs ({needed_ctx(sc, args.max_output):,}): the plain client's "
                     "prompts may be cut off, which would flatter it. The report counts truncated calls.")

    script = SimScript()
    b_rec, m_rec = Recorder(), Recorder()
    if mode == "real":
        say(f"Loading {main_model} and {helper_model} (context {ctx:,} tokens) ...")
        try:
            main_core, helper_core = build_real_core(main_model, args.ollama_url, ctx, args.seed), build_real_core(helper_model, args.ollama_url, ctx, args.seed)
            warm_up(main_core)
            warm_up(helper_core)
        except DependencyMissingError as e:
            return None, f"{e} (a real run reaches Ollama through LiteLLM; --simulate needs nothing installed)"
        except Exception as e:  # noqa: BLE001  (the server went away, a model failed to load, ...)
            return None, (f"Could not use Ollama: {type(e).__name__}: {str(e)[:200]}\nIf it says the model was not found, run `ollama list` and pass a "
                          "model you have with --model.")
    else:
        main_core, helper_core = SimModel("main", script), SimModel("helper", script)
    b_main = RecordingCore(main_core, "main", main_model, b_rec, ctx, args.max_output)
    m_main = RecordingCore(main_core, "main", main_model, m_rec, ctx, args.max_output)
    m_helper = RecordingCore(helper_core, "helper", helper_model, m_rec, ctx)

    def progress(arm, turn, o):
        if not args.quiet:
            say(f"  [{arm:8s} {turn.i:>2}/{len(sc.turns)}] {turn.kind:8s} {o.seconds:7.1f}s  {o.handled_by}")
    sim_script = script if mode == "simulated" else None
    say(f"Running the {len(sc.turns)}-turn session as a plain chat client (the baseline) ...")
    baseline = runner.run_baseline(sc, b_main, sim_script, progress)
    say("Running the same session through Pal ...")
    pal = runner.run_pal(sc, m_main, m_helper, sim_script, progress)
    return report.Experiment(sc, mode, main_model, helper_model, ctx, baseline, pal, args.max_output, notes), None


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "build":                       # a local model re-implements removed functions, judged by tests
        from experiments.demo.build_run import main as build_main
        return build_main(argv[1:])
    if argv and argv[0] == "rag":                         # a local model builds a RAG engine; how many tokens did Pal save?
        from experiments.demo.rag_build import main as rag_main
        return rag_main(argv[1:])
    if argv and argv[0] == "suite":                       # 30 varied builds, plain vs Pal; CSV of the ones that saved tokens + 3 bar charts
        from experiments.demo.suite_run import main as suite_main
        return suite_main(argv[1:])
    args = parser().parse_args(argv)
    try:
        exp, err = run(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    if err:
        print(err, file=sys.stderr)
        return 2
    paths = report.write_all(exp, args.out)
    print()
    print(report.render_terminal(report.summarize(exp), paths))
    return 0
