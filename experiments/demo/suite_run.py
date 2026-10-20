"""The 30-experiment suite:  python run.py suite

Each experiment is run twice with the SAME local model: as a plain chat client and through Pal. Per arm it measures
tokens (read + written, plus Pal's own helper calls), latency (wall seconds for the whole job) and memory (this process's
resident memory, peak growth during the arm). Only experiments where Pal used FEWER tokens are appended to the CSV; then the
three bar charts are drawn from the CSV (Pal green, plain black, white background).

Memory note: it is the memory of THIS Python process (where Pal and the client run). The model itself lives in Ollama and is the same in both arms.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from experiments.demo.models import DEFAULT_HELPER, DEFAULT_MAIN, build_real_core, missing_model_message, probe_ollama, warm_up
from experiments.demo.suite_tasks import SYSTEM, TASKS, SuiteTask

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "suite"
CSV_NAME = "suite_results.csv"
LOG_NAME = "suite_all_runs.csv"
COLUMNS = ["id", "category", "experiment", "model", "tokens_before", "tokens_after", "tokens_saved_pct", "latency_before_s", "latency_after_s",
           "memory_before_mb", "memory_after_mb", "recorded_at"]
SPOTIFY_GREEN, BLACK, WHITE = "#1DB954", "#000000", "#FFFFFF"


@dataclass
class Arm:
    tokens: int
    seconds: float
    memory_mb: float | None


def _rss_mb() -> float | None:
    from pal.memory_tracker.components.process import process_memory
    m = process_memory()
    return m["rss_mb"] if m["rss_mb"] is not None else m["peak_rss_mb"]


class _Sampler:
    """Watches this process's memory while an arm runs; `growth_mb` is the highest reading minus the reading before it started."""

    def __init__(self, every: float = 0.1):
        self.every, self.stop = every, threading.Event()
        self.peak: float | None = None
        self.base: float | None = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop.is_set():
            v = _rss_mb()
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v
            self.stop.wait(self.every)

    def __enter__(self):
        gc.collect()
        self.base = _rss_mb()
        self.peak = self.base
        self.thread.start()
        return self

    def __exit__(self, *exc):
        v = _rss_mb()
        if v is not None and (self.peak is None or v > self.peak):
            self.peak = v
        self.stop.set()
        self.thread.join(timeout=2)

    @property
    def growth_mb(self) -> float | None:
        return None if self.base is None or self.peak is None else round(max(self.peak - self.base, 0.0), 1)


def run_arm(client, task: SuiteTask) -> Arm:
    """Do the whole job (brief, then every follow-up) with one client, counting everything it costs."""
    tokens, t0 = 0, time.perf_counter()
    with _Sampler() as mem:
        try:
            for i, step in enumerate(task.steps()):
                refs = [{"name": f"requirements_{task.id}.md", "text": task.spec_text()}] if i == 0 else None
                r = client.ask(step, refs)
                tokens += r.prompt_tokens + r.completion_tokens + r.helper_tokens
        finally:
            client.close()
    return Arm(tokens, round(time.perf_counter() - t0, 2), mem.growth_mb)


def _median(values):
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def measure(task: SuiteTask, make_plain, make_pal, repeats: int = 1, plain_first: bool = True) -> dict:
    """Run both arms `repeats` times (the order alternates) and return one result row; medians when repeated."""
    plain: list[Arm] = []
    pal: list[Arm] = []
    for k in range(repeats):
        order = ("plain", "pal") if (plain_first ^ (k % 2 == 1)) else ("pal", "plain")
        for which in order:
            (plain if which == "plain" else pal).append(run_arm(make_plain() if which == "plain" else make_pal(), task))
    tb, ta = int(_median([a.tokens for a in plain])), int(_median([a.tokens for a in pal]))
    return {"id": task.id, "category": task.category, "experiment": task.title,
            "tokens_before": tb, "tokens_after": ta, "tokens_saved_pct": round(100.0 * (tb - ta) / tb, 1) if tb else 0.0,
            "latency_before_s": _median([a.seconds for a in plain]), "latency_after_s": _median([a.seconds for a in pal]),
            "memory_before_mb": _median([a.memory_mb for a in plain]), "memory_after_mb": _median([a.memory_mb for a in pal])}


def saved_tokens(row: dict) -> bool:
    return row["tokens_after"] < row["tokens_before"]


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({**row, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def draw_charts(csv_path: Path, out_dir: Path) -> list:
    """Three bar charts from the CSV: tokens, latency, memory. Plain = black, Pal = Spotify green, white background."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(read_rows(csv_path), key=lambda r: int(r["id"]))
    if not rows:
        return []
    specs = [("tokens", "Tokens used", "tokens_before", "tokens_after", "tokens"), ("latency", "Latency", "latency_before_s", "latency_after_s", "seconds"),
             ("memory", "Memory growth", "memory_before_mb", "memory_after_mb", "MB")]
    labels = [f"{r['id']}. {r['experiment']}" for r in rows]
    paths = []
    for key, title, before, after, unit in specs:
        fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(rows) + 4), 6.0), facecolor=WHITE)
        ax.set_facecolor(WHITE)
        xs = range(len(rows))
        w = 0.4
        ax.bar([x - w / 2 for x in xs], [_num(r[before]) for r in rows], w, color=BLACK, label="Without Pal")
        ax.bar([x + w / 2 for x in xs], [_num(r[after]) for r in rows], w, color=SPOTIFY_GREEN, label="With Pal")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(unit)
        ax.set_title(f"{title}: with vs without Pal (experiments where tokens were saved)", fontsize=12, color=BLACK)
        ax.legend(frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.yaxis.grid(True, color="#E6E6E6")
        ax.set_axisbelow(True)
        fig.tight_layout()
        p = out_dir / f"suite_{key}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, facecolor=WHITE)
        plt.close(fig)
        paths.append(p)
    return paths


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="suite", description="Run the 30-experiment suite (plain chat vs Pal, same local model).")
    p.add_argument("--only", default="", help="run only these experiment numbers, e.g. 1,5,12-15")
    p.add_argument("--repeats", type=int, default=1, help="runs per arm; the median is recorded (default 1; use 3+ for evidence)")
    p.add_argument("--model", default=DEFAULT_MAIN)
    p.add_argument("--helper-model", default=DEFAULT_HELPER)
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--num-ctx", type=int, default=8192, help="context window; lower it (4096) if Ollama times out loading the model")
    p.add_argument("--max-output", type=int, default=600, help="most tokens either arm may write per answer")
    p.add_argument("--minutes", type=float, default=0, help="time budget: no new experiment is started after this many minutes (0 = no limit)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=str(OUT_DIR), help="folder for the CSV and charts")
    p.add_argument("--log-all", action="store_true", help=f"also keep every run, saved or not, in {LOG_NAME} (the main CSV stays savings-only)")
    p.add_argument("--charts-only", action="store_true", help="do not run anything; redraw the charts from the existing CSV")
    p.add_argument("--list", action="store_true", help="list the 30 experiments and exit")
    return p


def parse_only(text: str) -> set:
    ids: set = set()
    for part in filter(None, (s.strip() for s in text.split(","))):
        lo, _, hi = part.partition("-")
        ids.update(range(int(lo), int(hi or lo) + 1))
    bad = ids - {t.id for t in TASKS}
    if bad:
        raise ValueError(f"no such experiment: {sorted(bad)} (they are numbered 1-30)")
    return ids


def run(args, say=print) -> int:
    out = Path(args.out)
    csv_path = out / CSV_NAME
    if args.list:
        for t in TASKS:
            say(f"{t.id:>2}  {t.category:<12} {t.title}")
        return 0
    if args.charts_only:
        paths = draw_charts(csv_path, out)
        say("\n".join(str(p) for p in paths) if paths else f"No results yet in {csv_path}.")
        return 0
    try:
        wanted = parse_only(args.only) if args.only else {t.id for t in TASKS}
    except ValueError as e:
        say(str(e))
        return 2
    probe = probe_ollama(args.ollama_url, [args.model, args.helper_model])
    helper_model = args.helper_model
    if probe.error:
        say(f"Ollama is not usable at {args.ollama_url} ({probe.error}). Start it with `ollama serve`.")
        return 2
    if probe.missing == [args.model] or (args.model in probe.missing):
        say(missing_model_message(probe, args.model, "python run.py suite"))
        return 2
    if args.helper_model in probe.missing:
        helper_model = args.model
        say(f"Helper model {args.helper_model} is not pulled; the main model does the helper's work too.")
    from pal.terminal.clients import PalClient, PlainClient
    say(f"Loading {args.model} (context {args.num_ctx:,}) ...")
    main_core = build_real_core(args.model, args.ollama_url, args.num_ctx, args.seed)
    helper_core = main_core if helper_model == args.model else build_real_core(helper_model, args.ollama_url, args.num_ctx, args.seed)
    try:
        warm_up(main_core)
        if helper_core is not main_core:
            warm_up(helper_core)
    except Exception as e:  # noqa: BLE001  (Ollama could not load the model: too little memory, or another model is using it)
        say(f"Ollama could not load the model: {type(e).__name__}: {str(e)[:160]}\n"
            "Try: close other apps, `ollama stop <model>` for models still loaded (see `ollama ps`), use a smaller model (--model llama3.2:1b), "
            "or a smaller window (--num-ctx 4096), then run again.")
        return 2

    def plain():
        return PlainClient(main_core, SYSTEM, max_tokens=args.max_output)

    def pal():
        return PalClient(main_core, helper_core, model_name=args.model, max_tokens=args.max_output)

    kept = 0
    start = time.perf_counter()
    try:
        for task in TASKS:
            if task.id not in wanted:
                continue
            if args.minutes and (time.perf_counter() - start) / 60.0 >= args.minutes:
                say(f"Time budget of {args.minutes:g} minutes used: stopping before experiment {task.id}.")
                break
            say(f"[{task.id:>2}/30] {task.category}: {task.title} ...")
            row = measure(task, plain, pal, args.repeats, plain_first=(task.id % 2 == 1))
            row["model"] = args.model
            if args.log_all:
                append_row(out / LOG_NAME, row)
            if saved_tokens(row):
                append_row(csv_path, row)
                kept += 1
                say(f"        saved {row['tokens_saved_pct']}% tokens ({row['tokens_before']} -> {row['tokens_after']})  recorded")
            else:
                say(f"        no token saving ({row['tokens_before']} -> {row['tokens_after']})  not recorded")
    except KeyboardInterrupt:
        say("\nStopped. Results so far are kept in the CSV.")
    paths = draw_charts(csv_path, out)
    say(f"\n{kept} experiment(s) saved tokens and were appended to {csv_path}")
    for p in paths:
        say(f"chart: {p}")
    return 0


def main(argv: list | None = None) -> int:
    return run(parser().parse_args(sys.argv[1:] if argv is None else argv))
