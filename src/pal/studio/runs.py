"""Runs: a build happens on a background thread, reports events as it goes, can be cancelled, and can be done twice to compare.

Only one run at a time (a local model cannot serve two builds well). The model and its key are used for the run and never stored: the plan,
the events and the files never contain the key.
"""
from __future__ import annotations

import re
import shutil
import threading
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from pal.errors import NotFoundError, PalError, UserInputError
from pal.redact import redact
from pal.studio.builder import Build
from pal.studio.editor import EditorSession
from pal.studio.policy import Policy
from pal.terminal.clients import PalClient, PlainClient

RUN_ID = re.compile(r"^[a-f0-9]{12}$")
SYSTEM = "You are a team of expert developers. You write complete, working files and reply with files."


class RunBusy(PalError):
    http_status = 409
    public = True


@dataclass
class ArmState:
    name: str
    label: str
    workspace: Path | None = None
    totals: dict = field(default_factory=dict)
    build: Build | None = None


def pct(before: float, after: float) -> float:
    return round(100.0 * (before - after) / before, 1) if before else 0.0


def compare(base: dict, mat: dict) -> dict:
    """The two builds side by side, and what Pal saved. Honest about what the numbers can and cannot say."""
    out = {"baseline": base, "pal": mat,
           "main_tokens_saved": base["main_tokens"] - mat["main_tokens"], "main_tokens_saved_pct": pct(base["main_tokens"], mat["main_tokens"]),
           "prompt_tokens_saved_pct": pct(base["prompt_tokens"], mat["prompt_tokens"]),
           "all_tokens_saved_pct": pct(base["main_tokens"], mat["main_tokens"] + mat["helper_tokens"]), "seconds_saved_pct": pct(base["seconds"], mat["seconds"]), "notes": []}
    n = out["notes"]
    if out["main_tokens_saved"] <= 0:
        n.append("Pal used as many tokens as the plain client or more on this build. That happens when there is little to resend: it adds a fixed answer-style rule "
                 "to every request, and it pays off when a build has reference documents, a project or a long conversation.")
    if out["seconds_saved_pct"] < 0:
        n.append("Pal was slower. Its small local helper jobs (trimming and checking what is typed) take time, and a local server may reuse the start of the plain "
                 "client's repeated prompt, which makes the plain client faster than its token count suggests.")
    if base["files"] != mat["files"] or base["problems_left"] != mat["problems_left"]:
        n.append(f"The results differ: the plain client wrote {base['files']} file(s) with {base['problems_left']} problem(s) left; Pal wrote {mat['files']} with {mat['problems_left']}.")
    n.append("This is one run of each, with the plain client first. A model varies from run to run, so repeat it before relying on the exact percentages.")
    return out


class Run:
    def __init__(self, run_id: str, plan: dict):
        self.id, self.plan = run_id, plan
        self.status = "queued"
        self.started = time.time()
        self.finished: float | None = None
        self.events: list[dict] = []
        self.arms: dict[str, ArmState] = {}
        self.comparison: dict | None = None
        self.cancel = threading.Event()
        self._lock = threading.Lock()

    def emit(self, kind: str, text: str = "", arm: str = "", **data) -> None:
        with self._lock:
            self.events.append({"id": len(self.events), "t": round(time.time() - self.started, 2), "arm": arm, "kind": kind, "text": text, "data": data})

    def events_since(self, since: int = 0) -> list[dict]:
        with self._lock:
            return self.events[max(0, since):]

    def summary(self) -> dict:
        return {"id": self.id, "status": self.status, "product": self.plan["product"]["name"], "metrics": bool(self.plan.get("metrics")),
                "seconds": round((self.finished or time.time()) - self.started, 1),
                "arms": {a.name: {"label": a.label, "totals": a.build.totals if a.build else a.totals, "files": len(a.build.files) if a.build else 0} for a in self.arms.values()}, "comparison": self.comparison}

    def arm(self, name: str) -> ArmState:
        if name not in self.arms:
            raise NotFoundError("no such build in this run")
        return self.arms[name]


def build_target(project: Path, folder: str) -> Path:
    """Where a build is written in the folder Pal was started in: the folder itself when it is empty, else a new subfolder named after the
    product (with -2, -3 ... when that name is taken), so nothing that is already there is overwritten."""
    project = Path(project)
    if not any(p for p in project.iterdir() if not p.name.startswith(".")):
        return project
    target, n = project / folder, 2
    while target.exists() and (not target.is_dir() or any(target.iterdir())):
        target, n = project / f"{folder}-{n}", n + 1
    return target


class RunStore:
    def __init__(self, root: Path, core_factory: Callable, max_runs: int = 6, hot_turns: int | None = None, project_dir: Path | None = None):
        self.root, self.core_factory, self.max_runs, self.hot_turns = Path(root), core_factory, max_runs, hot_turns
        self.project_dir = Path(project_dir).resolve() if project_dir else None      # pal web's folder: the build is written there
        self._runs: dict[str, Run] = {}
        self._editors: dict[tuple[str, str], EditorSession] = {}
        self._lock = threading.Lock()

    def get(self, run_id: str) -> Run:
        if not RUN_ID.match(run_id or ""):
            raise NotFoundError("no such run")
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise NotFoundError("no such run")
        return run

    def start(self, plan: dict, references: list[dict], api_key: str | None, wait: bool = False) -> Run:
        with self._lock:
            if any(r.status in ("queued", "running") for r in self._runs.values()):
                raise RunBusy("a build is already running: wait for it, or stop it first")
            run = Run(uuid.uuid4().hex[:12], plan)
            self._runs[run.id] = run
            for old in sorted(self._runs.values(), key=lambda r: r.started)[:-self.max_runs]:
                self._runs.pop(old.id, None)
                for key in [k for k in self._editors if k[0] == old.id]:
                    self._editors.pop(key).close()
                shutil.rmtree(self.root / old.id, ignore_errors=True)
        t = threading.Thread(target=self._execute, args=(run, plan, references, api_key), daemon=True, name=f"studio-{run.id}")
        t.start()
        if wait:
            t.join()
        return run

    def _execute(self, run: Run, plan: dict, references: list[dict], api_key: str | None) -> None:
        run.status = "running"
        run.emit("run_start", f"Building {plan['product']['name']} with {len(plan['roles'])} agent(s) and {len(plan['steps'])} step(s)")
        try:
            core = self.core_factory(plan, api_key)
            model_name = plan["model"]["model"] or "model"
            twice = bool(plan.get("metrics"))
            for arm, label in (("baseline", PlainClient.label), ("pal", PalClient.label)) if twice else (("main", PalClient.label),):
                if run.cancel.is_set():
                    break
                policy = Policy(dict(plan["permissions"]))
                save = plan["permissions"].get("write_project")
                where = None
                if save:
                    where = (build_target(self.project_dir, plan["product"]["folder"]) if self.project_dir and arm != "baseline"
                             else self.root / run.id / arm / plan["product"]["folder"])  # the plain client's comparison copy stays in Pal's folder
                state = ArmState(arm, label, where)
                run.arms[arm] = state
                if state.workspace:
                    state.workspace.mkdir(parents=True, exist_ok=True)
                run.emit("arm_start", f"{label}: starting" if twice else "Starting", arm=arm)
                if state.workspace and self.project_dir and arm != "baseline":
                    run.emit("saved_to", f"Files are written to {state.workspace}", arm=arm, path=str(state.workspace))
                client: PlainClient | PalClient
                if arm == "baseline":
                    client = PlainClient(core, SYSTEM)
                else:
                    allow = frozenset() if twice else policy.allowed_tools(plan["tools"])
                    client = PalClient(core, None, allow, network=bool(plan["permissions"].get("network_access")) and not twice, hot_turns=self.hot_turns,
                                          home=self.root / run.id / f"{arm}-pal-home", model_name=model_name)
                try:
                    state.build = Build(plan, client, state.workspace, policy, lambda kind, text="", _a=arm, **d: run.emit(kind, text, arm=_a, **d), run.cancel.is_set, references)
                    state.totals = state.build.run()
                finally:
                    client.close()
                    shutil.rmtree(self.root / run.id / f"{arm}-pal-home", ignore_errors=True)
                run.emit("arm_done", f"{label}: done, {state.totals['files']} file(s), {state.totals['main_tokens']:,} tokens, {state.totals['seconds']:.1f}s", arm=arm)
                if state.totals.get("stopped") == "error":
                    break
            if twice and "baseline" in run.arms and "pal" in run.arms and run.arms["pal"].totals.get("steps_done") == run.arms["pal"].totals.get("steps") \
                    and run.arms["baseline"].totals.get("steps_done") == run.arms["baseline"].totals.get("steps"):
                run.comparison = compare(run.arms["baseline"].totals, run.arms["pal"].totals)
                run.emit("comparison", f"Plain client {run.comparison['baseline']['main_tokens']:,} tokens, Pal {run.comparison['pal']['main_tokens']:,}: "
                         f"{run.comparison['main_tokens_saved_pct']}% saved", comparison=run.comparison)
            stopped = [a.totals.get("stopped") for a in run.arms.values()]
            run.status = "cancelled" if run.cancel.is_set() else ("error" if "error" in stopped else "done")
            run.emit("run_done", {"done": "Finished", "cancelled": "Stopped", "error": "Stopped by an error"}[run.status])
        except Exception as e:  # noqa: BLE001  (a model that cannot be reached, a missing key ...)
            run.status = "error"
            run.emit("error", redact(f"{type(e).__name__}: {e}", api_key)[:300])
            run.emit("run_done", "Stopped by an error")
        finally:
            run.finished = time.time()

    def editable(self, run: Run, arm: str) -> ArmState:
        """The build's state, once the run is over and has files: editing during a build would race with it."""
        if run.status in ("queued", "running"):
            raise RunBusy("the build is still running: open the editor when it has finished")
        state = run.arm(arm)
        if state.build is None:
            raise UserInputError("this build has no files to edit")
        return state

    def editor(self, run: Run, arm: str, api_key: str | None) -> EditorSession:
        """One editing session per build: the same token-saving client for every request, so its history and file index carry over."""
        state = self.editable(run, arm)
        with self._lock:
            session = self._editors.get((run.id, arm))
            if session is None:
                core = self.core_factory(run.plan, api_key)
                client = PalClient(core, None, frozenset(), network=False, hot_turns=self.hot_turns, home=self.root / run.id / f"{arm}-edit-home",
                                      model_name=run.plan["model"]["model"] or "model", max_tokens=6000)
                assert state.build is not None                          # editable() checked it  # noqa: S101
                session = self._editors[(run.id, arm)] = EditorSession(client, state.build.plan["product"]["name"])
        return session

    def zip_bytes(self, run: Run, arm: str) -> bytes:
        state = run.arm(arm)
        if state.workspace is None or not state.workspace.is_dir():
            raise UserInputError("nothing was saved for this build (saving was switched off)")
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(state.workspace.rglob("*")):
                if p.is_file():
                    z.write(p, (Path(state.workspace.name) / p.relative_to(state.workspace)).as_posix())
        return buf.getvalue()
