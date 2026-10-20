"""Plans, context and the change sandbox: the Pal workflow.

    1 Chat (Ask agent)        talk the task through until it is clear
    2 Plan (Plan agent)       broad strokes, the files it needs (loaded into context automatically at "semi" and "full")
    3 Detailed plan           numbered steps, kept with the plan
    4 Build (Build agent)     each step implemented as pending changes in a sandbox: nothing touches your files yet
    5 Review                  /diff, /review (in the browser), /rewind, /reject
    6 Apply and debug         /apply writes the changes, runs the checking commands you approve, rolls everything back if one fails,
                              and (auto-debug) asks the Build agent to fix it and tries again
    7 Commit                  in a git repository, a commit message is written and the changes are committed (auto-commit)

Everything is saved per project folder in ~/.pal/projects/<folder id>/: each plan has branches, and each branch keeps its conversation,
its context, its numbered steps and every version of the sandbox, so /rewind and /checkout can go back or try another way.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pal.paths import pal_home
from pal.terminal.security import Finding, review_code
from pal.terminal.session import MAX_REF_BYTES, PalSession, Turn, read_any
from pal.tokens import estimate_tokens

# Auto modes: what each level does by itself (the same switches for every plan; set-config changes one, and the plan's mode becomes "custom").
_ON = {
    "full": {"auto-continue", "auto-build", "auto-update-context", "auto-load-context", "smart-context", "auto-apply", "auto-commit", "can-exec",
             "auto-exec", "auto-debug", "auto-revert"},
    "semi": {"auto-continue", "auto-build", "auto-update-context", "auto-load-context", "smart-context", "auto-commit", "can-exec", "auto-revert"},
    "plus": {"auto-continue", "auto-build", "auto-update-context", "smart-context", "auto-commit", "can-exec", "auto-revert"},
    "basic": {"auto-continue", "auto-build", "auto-revert"},
    "none": {"auto-revert"},
}
AUTONOMY = _ON
AUTO_MODES = [   # (key, label, description), in the order they are offered
    ("full", "Full Auto", "Fully automated: context, apply, execution and debugging"),
    ("semi", "Semi Auto", "Auto context, manual apply and execution"),
    ("plus", "Basic Plus", "Manual context with auto updates and smart loading, manual apply and execution"),
    ("basic", "Basic", "Manual context, manual apply and execution"),
    ("none", "None", "Fully manual and step-by-step, one response at a time, manual builds"),
    ("custom", "Custom", "Choose settings individually with set-config command"),
]
DEFAULT_AUTO_MODE = "full"          # new plans: Pal applies, runs the checks and fixes failures by itself (\set-auto default changes it)
# every setting: (name, type, description), in display order (auto-mode first, then by name)
CONFIG = [
    ("auto-mode", "string", "Use preset config based on desired level of autonomy"),
    ("auto-apply", "bool", "Automatically apply changes after plan finishes"),
    ("auto-build", "bool", "Automatically generate pending file edits"),
    ("auto-commit", "bool", "Automatically commit changes to git after apply"),
    ("auto-continue", "bool", "Continue iterating until plan is complete"),
    ("auto-debug", "bool", "Automatically debug failed commands"),
    ("auto-debug-tries", "int", "Number of auto-debug attempts"),
    ("auto-exec", "bool", "Automatically execute commands after plan is applied"),
    ("auto-load-context", "bool", "Find and load context automatically"),
    ("auto-revert", "bool", "Automatically update project files when rewinding plan"),
    ("auto-update-context", "bool", "Automatically update context after changes"),
    ("can-exec", "bool", "Allow execution of commands"),
    ("editor", "editor", "Preferred editor"),
    ("skip-changes-menu", "bool", "Skip interactive menu when response finishes and changes are pending"),
    ("skip-commit", "bool", "Skip committing changes to git after apply"),
    ("smart-context", "bool", "Load only necessary context for each task in the plan"),
]
SETTINGS = tuple(n for n, t, _ in CONFIG if t == "bool")
SETTING_ABOUT = {n: d for n, _, d in CONFIG}
MAX_CONTEXT_FILES, MAX_STEPS, DEBUG_TRIES = 1000, 12, 5
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,40}$")


@dataclass
class Version:
    """One state of the sandbox: every pending file (path -> new content) and the commands to check it."""
    id: str
    label: str
    t: float
    pending: dict[str, str] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)


@dataclass
class Branch:
    name: str = "main"
    convo: list[dict] = field(default_factory=list)          # {"role": "you"|"pal", "agent", "text", "t"}
    context: list[str] = field(default_factory=list)         # project-relative paths, or absolute paths outside the folder
    notes: list[str] = field(default_factory=list)           # text loaded with /load -n
    steps: list[str] = field(default_factory=list)           # the detailed plan
    done: int = 0                                            # how many steps are built
    versions: list[Version] = field(default_factory=list)
    applied: list[dict] = field(default_factory=list)        # what apply wrote: {"t", "files", "commit", "before", "after", "msg"}
    added: dict[str, float] = field(default_factory=dict)    # when each context item was loaded
    log: list[dict] = field(default_factory=list)            # every update: {"id", "t", "msg", "state"}; rewind goes back to one
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    @property
    def pending(self) -> dict[str, str]:
        return dict(self.versions[-1].pending) if self.versions else {}

    @property
    def commands(self) -> list[str]:
        return list(self.versions[-1].commands) if self.versions else []


@dataclass
class Plan:
    name: str
    created: float = field(default_factory=time.time)
    branch: str = "main"
    autonomy: str = "semi"
    custom: dict = field(default_factory=dict)
    branches: dict[str, Branch] = field(default_factory=lambda: {"main": Branch()})
    archived: bool = False
    updated: float = field(default_factory=time.time)

    @property
    def b(self) -> Branch:
        return self.branches[self.branch]

    def setting(self, name: str):
        if name in self.custom:
            return self.custom[name]
        if name == "auto-debug-tries":
            return 5 if self.autonomy == "full" else 0
        if name == "editor":
            return ""
        return name in _ON.get(self.autonomy, set())

    def config(self) -> dict:
        """Every setting with its value now."""
        out: dict = {"auto-mode": self.autonomy}
        for name, _, _ in CONFIG[1:]:
            out[name] = self.setting(name)
        return out

    def set_auto(self, mode: str) -> None:
        """A preset: every switch follows the mode (skip-commit, the editor and the debug tries are kept)."""
        keep = {k: v for k, v in self.custom.items() if k in ("skip-commit", "editor", "auto-debug-tries")}
        keep.setdefault("auto-debug-tries", self.setting("auto-debug-tries"))
        if mode == "full":
            keep["auto-debug-tries"] = 5
        self.autonomy, self.custom = mode, (keep if mode in _ON else {**self.config(), **keep, "auto-mode": mode})
        self.custom.pop("auto-mode", None)

    def set(self, name: str, value) -> None:
        """One setting, with what it implies (as the reference config does): any change but skip-* makes the mode custom."""
        if name == "auto-mode":
            self.set_auto(str(value))
            return
        before = self.setting(name)
        if self.autonomy != "custom" and name not in ("skip-changes-menu", "skip-commit", "editor") and before != value:
            self.custom = self.config()
            self.custom.pop("auto-mode", None)
            self.autonomy = "custom"
        self.custom[name] = value
        if name == "auto-apply" and value:
            self.custom["auto-build"] = True
        elif name == "auto-build" and not value:
            self.custom["auto-apply"] = False
        elif name == "auto-debug" and value:
            self.custom.update({"can-exec": True, "auto-exec": True})
            if not self.setting("auto-debug-tries"):
                self.custom["auto-debug-tries"] = 5
        elif name == "auto-debug-tries":
            if not value:
                self.custom["auto-debug"] = False
            else:
                self.custom.update({"can-exec": True, "auto-exec": True, "auto-debug": True})
        elif name == "auto-exec":
            if value:
                self.custom["can-exec"] = True
            else:
                self.custom["auto-debug"] = False
        elif name == "can-exec" and not value:
            self.custom.update({"auto-exec": False, "auto-debug": False})


def default_plan(name: str) -> Plan:
    """A new plan with the default config (\\set-config default and \\set-auto default change it)."""
    plan = Plan(name)
    cfg = default_config()
    plan.autonomy = cfg.pop("auto-mode", DEFAULT_AUTO_MODE)
    plan.custom = cfg
    return plan


def default_config() -> dict:
    """{"auto-mode": mode, **overrides} for new plans."""
    from pal.terminal.backend import load_prefs
    raw = load_prefs().get("default_config", "")
    try:
        cfg = json.loads(raw) if raw else {}
    except ValueError:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault("auto-mode", DEFAULT_AUTO_MODE)
    return cfg


def save_default_config(plan: Plan) -> None:
    from pal.terminal.backend import save_prefs
    save_prefs(default_config=json.dumps({"auto-mode": plan.autonomy, **plan.custom}))


def pid_alive(pid: int) -> bool:
    """Is that process still running? (Never sends it a signal: on Windows os.kill would end it.)"""
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259    # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                                      # alive, owned by someone else
        return True
    except OSError:
        return False
    return True


def project_id(folder: Path) -> str:
    folder = folder.resolve()
    return f"{folder.name[:24]}-{hashlib.sha1(str(folder).encode('utf-8')).hexdigest()[:10]}"   # noqa: S324 (a folder id, not security)


class Store:
    """Plans for one project folder, saved as JSON."""

    def __init__(self, folder: Path, home: Path | None = None):
        self.folder = Path(folder).resolve()
        self.root = (home or pal_home()) / "projects" / project_id(self.folder)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.plan.json"

    def names(self) -> list[str]:
        return sorted(p.name[: -len(".plan.json")] for p in self.root.glob("*.plan.json"))

    def load(self, name: str) -> Plan:
        raw = json.loads(self._path(name).read_text(encoding="utf-8"))
        branches = {k: Branch(**{**v, "versions": [Version(**x) for x in v.get("versions", [])]}) for k, v in raw.pop("branches").items()}
        return Plan(**raw, branches=branches)

    def save(self, plan: Plan, current: bool = True) -> None:
        data = asdict(plan)
        tmp = self._path(plan.name).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self._path(plan.name))
        if current:
            (self.root / "current").write_text(plan.name, encoding="utf-8")

    def current_name(self) -> str | None:
        f = self.root / "current"
        name = f.read_text(encoding="utf-8").strip() if f.is_file() else ""
        return name if name in self.names() else None

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)
        self.release(name)

    # -- one terminal per plan: a second one would overwrite the first one's changes ---------------------------------------
    def _lock(self, name: str) -> Path:
        return self.root / f"{name}.lock"

    def acquire(self, name: str) -> int | None:
        """Mark the plan as open in this process. Returns the process id of another live Pal that has it open (the plan is then not taken),
        or None when it is ours now. A lock left by a Pal that has exited is taken over."""
        f = self._lock(name)
        try:
            pid = int(f.read_text(encoding="utf-8").split()[0]) if f.is_file() else 0
        except (OSError, ValueError, IndexError):
            pid = 0
        if pid and pid != os.getpid() and pid_alive(pid):
            return pid
        try:
            f.write_text(f"{os.getpid()} {time.time():.0f}", encoding="utf-8")
        except OSError:
            pass
        return None

    def release(self, name: str) -> None:
        f = self._lock(name)
        try:
            if f.is_file() and f.read_text(encoding="utf-8").split()[0] == str(os.getpid()):
                f.unlink()
        except (OSError, IndexError):
            pass

    def new_name(self) -> str:
        if "draft" not in self.names():
            return "draft"
        n = 2
        while f"draft-{n}" in self.names():
            n += 1
        return f"draft-{n}"


@dataclass
class ApplyResult:
    written: list[str] = field(default_factory=list)
    failed_command: str = ""
    output: str = ""
    rolled_back: bool = False
    commit: str = ""
    message: str = ""


class Workflow:
    """The current plan and branch, tied to a session: context goes with every request, Build's changes go to the sandbox."""

    def __init__(self, session: PalSession, store: Store | None = None, plan: str | None = None):
        self.session, self.store = session, store or Store(session.cwd)
        self.notice = ""                            # said once when the session opens (a plan open in another terminal)
        name = plan or self.store.current_name()
        other = self.store.acquire(name) if name else None
        if other:
            fresh = self.store.new_name()
            self.notice = (f"Plan {name} is open in another Pal, so this one started a new plan, {fresh}, "
                           "and the two cannot overwrite each other. \\cd to it after closing the other.")
            name = None
            self.store.acquire(fresh)
            self.plan = default_plan(fresh)
        else:
            self.plan = self.store.load(name) if name else default_plan(self.store.new_name())
            self.store.acquire(self.plan.name)
        self.store.save(self.plan)
        self._attach()

    def close(self) -> None:
        """Let other terminals open this plan."""
        self.store.release(self.plan.name)

    def _take(self, name: str) -> None:
        other = self.store.acquire(name)
        if other:
            raise ValueError(f"plan {name} is open in another Pal: close that one first, or \\new")
        if self.plan.name != name:
            self.store.release(self.plan.name)

    # -- wiring -------------------------------------------------------------------------------------------------------------
    def _attach(self) -> None:
        s = self.session
        s.context = self.context_refs
        s.overlay = self.plan.b.pending
        if self.plan.b.convo:                       # resuming: the model gets the earlier conversation once, compactly
            s.preamble = self.recap()

    def save(self) -> None:
        self.session.overlay = self.plan.b.pending
        self.plan.updated = self.plan.b.updated = time.time()
        self.store.save(self.plan)

    # -- the plan's history: every update is logged with the state it left, so rewind can go back to it -------------------------
    def _state(self) -> dict:
        b = self.plan.b
        return {"convo": len(b.convo), "context": list(b.context), "notes": list(b.notes), "steps": list(b.steps), "done": b.done,
                "versions": len(b.versions), "applied": len(b.applied), "added": dict(b.added)}

    def log_event(self, msg: str) -> str:
        sha = hashlib.sha1(f"{time.time()}{msg}{len(self.plan.b.log)}".encode()).hexdigest()[:7]   # noqa: S324 (an id, not security)
        self.plan.b.log.append({"id": sha, "t": time.time(), "msg": msg, "state": self._state()})
        self.save()
        return sha

    def rewind_to(self, sha: str) -> list[dict]:
        """Put the plan back as it was right after log entry `sha`. Returns the applies that were undone (newest last)."""
        b = self.plan.b
        idx = next((i for i, e in enumerate(b.log) if e["id"] == sha or (len(sha) >= 4 and e["id"].startswith(sha))), None)
        if idx is None:
            raise ValueError(f"no update {sha}")
        st = b.log[idx]["state"]
        undone = b.applied[st["applied"]:]
        del b.convo[st["convo"]:]
        del b.versions[st["versions"]:]
        del b.applied[st["applied"]:]
        b.context, b.notes, b.steps, b.done, b.added = list(st["context"]), list(st["notes"]), list(st["steps"]), st["done"], dict(st.get("added", {}))
        del b.log[idx + 1:]
        self.save()
        return undone

    def tokens(self) -> tuple[int, int]:
        """(context tokens, conversation tokens) of the current branch."""
        ctx = sum(r[2] for r in self.context_rows())
        convo = sum(int(m.get("tokens") or estimate_tokens(m.get("text", ""))) for m in self.plan.b.convo)
        return ctx, convo

    def mark_applied(self, written: list[str], before: dict, after: dict, commit: str = "", msg: str = "") -> None:
        """The pending changes are on disk now: recorded (with the files as they were, for rewind) and cleared from the sandbox."""
        b = self.plan.b
        b.applied.append({"t": time.time(), "files": written, "commit": commit, "before": before, "after": after, "msg": msg})
        b.versions.append(Version(hashlib.sha1(f"{time.time()}applied".encode()).hexdigest()[:7], "applied " + ", ".join(written)[:70], time.time(), {}, []))   # noqa: S324
        self.save()
        self.log_event("\u2705 Marked pending results as applied" + "".join(f"\n \u2022 \U0001F4C4 {p}" for p in sorted(after)) + (f"\n\u270F\uFE0F  {msg}" if msg else ""))

    def recap(self, limit: int = 12) -> str:
        lines = [f"{m['role']}: {m['text'][:600]}" for m in self.plan.b.convo[-limit:]]
        steps = "\n".join(f"{i}. {t}" for i, t in enumerate(self.plan.b.steps, 1))
        return ("Earlier in this plan (for context, do not answer it again):\n" + "\n".join(lines)
                + (f"\nThe plan's steps ({self.plan.b.done} built):\n{steps}" if steps else ""))

    # -- plans and branches ----------------------------------------------------------------------------------------------------
    def new(self, name: str = "") -> str:
        name = name or self.store.new_name()
        if not _NAME.match(name):
            raise ValueError("a plan name is letters, digits, dot, dash or underscore (up to 40)")
        if name in self.store.names():
            raise ValueError(f"there is already a plan called {name}")
        self._take(name)
        self.plan = default_plan(name)
        self.session.preamble = ""
        self.store.save(self.plan)
        self._attach()
        return name

    def switch(self, which: str) -> str:
        names = self.store.names()
        name = which if which in names else names[int(which) - 1] if which.isdigit() and 0 < int(which) <= len(names) else which
        if name not in names:
            raise ValueError(f"no plan called {which}")
        self._take(name)
        self.plan = self.store.load(name)
        self.store.save(self.plan)
        self._attach()
        return name

    def rename(self, new: str) -> None:
        if not _NAME.match(new) or new in self.store.names():
            raise ValueError("pick another name (letters, digits, dot, dash, underscore; not already used)")
        old = self.plan.name
        self.store.acquire(new)
        self.plan.name = new
        self.store.save(self.plan)
        self.store.delete(old)

    def delete(self, name: str) -> None:
        if name not in self.store.names():
            raise ValueError(f"no plan called {name}")
        if name != self.plan.name and (other := self.store.acquire(name)):
            raise ValueError(f"plan {name} is open in another Pal")
        self.store.delete(name)
        if name == self.plan.name:
            others = [n for n in self.store.names() if not self.store.acquire(n)]
            self.plan = self.store.load(others[0]) if others else default_plan(self.store.new_name())
            self.store.acquire(self.plan.name)
            for n in others[1:]:
                self.store.release(n)
            self.store.save(self.plan)
            self._attach()

    def checkout(self, name: str) -> bool:
        """Switch to a branch; a new one starts as a copy of the current branch. True when it was created."""
        if not _NAME.match(name):
            raise ValueError("a branch name is letters, digits, dot, dash or underscore")
        created = name not in self.plan.branches
        if created:
            self.plan.branches[name] = Branch(**json.loads(json.dumps(asdict(self.plan.b), default=str)) | {"name": name})
            self.plan.branches[name].versions = [Version(**v) if isinstance(v, dict) else v for v in self.plan.branches[name].versions]
        self.plan.branch = name
        self.save()
        return created

    def delete_branch(self, name: str) -> None:
        if name == "main" or name not in self.plan.branches:
            raise ValueError("pick a branch other than main that exists")
        if self.plan.branch == name:
            self.plan.branch = "main"
        del self.plan.branches[name]
        self.save()

    # -- context ------------------------------------------------------------------------------------------------------------------
    def _resolve(self, raw: str) -> list[Path]:
        root = self.session.cwd
        raw = raw.strip().strip("\"'")
        p = (root / raw) if not Path(raw).expanduser().is_absolute() else Path(raw).expanduser()
        if any(c in raw for c in "*?["):
            return sorted(x for x in root.glob(raw) if x.is_file())
        if p.is_dir():
            from pal.terminal.session import SKIP_DIRS
            return sorted(x for x in p.rglob("*") if x.is_file() and not any(part in SKIP_DIRS or part.startswith(".") for part in x.relative_to(p).parts))
        return [p] if p.is_file() else []

    def _key(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.session.cwd).as_posix()
        except ValueError:
            return str(p.resolve())

    def load(self, items: list[str], note: str = "") -> tuple[list[str], list[str]]:
        """Add files, folders or globs (and an optional note) to the context. (added, problems)."""
        added, problems = [], []
        if note:
            self.plan.b.notes.append(note[:4000])
            added.append("note")
        for raw in items:
            found = self._resolve(raw)
            if not found:
                problems.append(f"{raw}: nothing found")
            for p in found:
                key = self._key(p)
                if key in self.plan.b.context:
                    continue
                if len(self.plan.b.context) >= MAX_CONTEXT_FILES:
                    problems.append(f"context is full ({MAX_CONTEXT_FILES} files): \\rm something first")
                    break
                self.plan.b.context.append(key)
                self.plan.b.added[key] = time.time()
                added.append(key)
        self.save()
        return added, problems

    def remove(self, items: list[str]) -> list[str]:
        """Remove context by number (from /ls), range (2-4), name or glob."""
        ctx = self.plan.b.context
        drop: set[str] = set()
        for raw in items:
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", raw)
            if m:
                a, b = int(m.group(1)), int(m.group(2) or m.group(1))
                drop |= {ctx[i - 1] for i in range(a, b + 1) if 0 < i <= len(ctx)}
            else:
                from fnmatch import fnmatch
                drop |= {c for c in ctx if c == raw or fnmatch(c, raw)}
        self.plan.b.context = [c for c in ctx if c not in drop]
        self.save()
        return sorted(drop)

    def clear(self) -> int:
        n = len(self.plan.b.context) + len(self.plan.b.notes)
        self.plan.b.context, self.plan.b.notes = [], []
        self.save()
        return n

    def context_rows(self) -> list[tuple[int, str, int, str]]:
        """(number, name, tokens, note) for /ls."""
        rows = []
        for i, key in enumerate(self.plan.b.context, 1):
            p = Path(key) if Path(key).is_absolute() else self.session.cwd / key
            text, problem = read_any(p) if p.is_file() else (None, "missing")
            rows.append((i, key, estimate_tokens(text or ""), problem or ""))
        for j, note in enumerate(self.plan.b.notes, len(rows) + 1):
            rows.append((j, f"note: {note[:40]}", estimate_tokens(note), ""))
        return rows

    def context_refs(self) -> list[dict]:
        """What goes with every request: the loaded files as they are now (with pending changes on top), secrets hidden, and the notes."""
        from pal.terminal.security import guard_outgoing
        refs, pending = [], self.plan.b.pending
        text: str | None
        for key in self.plan.b.context:
            if key in pending:
                text = pending[key]
            else:
                p = Path(key) if Path(key).is_absolute() else self.session.cwd / key
                text, _ = read_any(p) if p.is_file() else (None, None)
            if text is None:
                continue
            safe, _ = guard_outgoing(key, text)
            if safe is not None and len(safe.encode("utf-8")) <= MAX_REF_BYTES:
                refs.append({"name": key, "text": safe})
        if self.plan.b.notes:
            refs.append({"name": "notes.md", "text": "\n\n".join(self.plan.b.notes)})
        return refs

    # -- talking ------------------------------------------------------------------------------------------------------------------
    def tell(self, text: str) -> Turn:
        """One request with the session's current agent. Plan replies become the plan's steps; Build replies become a sandbox version."""
        b = self.plan.b
        asked = time.time()
        turn = self.session.ask(text)
        n = len(b.convo)
        you = estimate_tokens(text)
        b.convo.append({"role": "you", "agent": self.session.mode, "text": text, "t": asked, "tokens": you})
        self.log_event(f"Message #{n + 1} | \U0001F4AC User prompt | {you} \U0001FA99")
        reply_text = turn.text if not (self.session.mode == "build" and turn.changes) else (turn.text + "".join(f"\n\n\U0001F4C4 {c.path}" for c in turn.changes)).strip()
        out = turn.completion_tokens or estimate_tokens(reply_text)
        b.convo.append({"role": "pal", "agent": self.session.mode, "text": reply_text[:20000], "t": time.time(), "tokens": out,
                        "tags": (["\U0001F468\u200D\U0001F4BB Wrote Code"] if turn.changes else []) + (["\U0001F4CB Made Plan"] if self.session.mode == "plan" else [])})
        tags = "".join(f" | {t}" for t in b.convo[-1]["tags"])
        if self.session.mode == "plan":
            steps = parse_steps(turn.text)
            if steps:
                b.steps, b.done = steps[:MAX_STEPS], 0
            if self.plan.setting("auto-load-context"):
                named = [x for x in mentioned_files(turn.text, self.session.cwd) if x not in b.context]
                if named:
                    self.load(named)
                    turn.warnings.append("loaded into context: " + ", ".join(named))
        if self.session.mode == "build" and (turn.changes or turn.commands):
            self.add_version(turn, label=text[:80])
        self.log_event(f"Message #{n + 2} | \U0001F916 Pal reply{tags} | {out} \U0001FA99")
        return turn

    def add_version(self, turn: Turn, label: str) -> Version:
        pending = self.plan.b.pending
        for c in turn.changes:
            pending[c.path] = c.new
        commands = turn.commands or self.plan.b.commands
        v = Version(hashlib.sha1(f"{time.time()}{label}".encode()).hexdigest()[:7], label, time.time(), pending, commands)   # noqa: S324
        self.plan.b.versions.append(v)
        self.save()
        return v

    def next_step(self) -> tuple[int, str] | None:
        b = self.plan.b
        return (b.done + 1, b.steps[b.done]) if b.done < len(b.steps) else None

    def build_step(self) -> Turn | None:
        """/continue: implement the next step of the plan in the sandbox."""
        nxt = self.next_step()
        if nxt is None:
            return None
        i, step = nxt
        mode = self.session.mode
        self.session.mode = "build"
        try:
            turn = self.tell(f"Implement step {i} of {len(self.plan.b.steps)} of the plan: {step}. Only this step; keep everything else as it is.")
        finally:
            self.session.mode = mode
        self.plan.b.done = i
        self.save()
        return turn

    # -- the sandbox ----------------------------------------------------------------------------------------------------------------
    def disk(self, path: str) -> str | None:
        p = self.session.cwd / path
        try:
            return p.read_text(encoding="utf-8") if p.is_file() else None
        except (OSError, UnicodeDecodeError):
            return None

    def diffs(self) -> list[tuple[str, bool, list[list[str]], list[Finding]]]:
        from pal.terminal.session import diff_lines
        out = []
        for path, new in sorted(self.plan.b.pending.items()):
            old = self.disk(path)
            out.append((path, old is None, diff_lines(old or "", new), review_code(path, new)))
        return out

    def reject(self, paths: list[str] | None = None) -> list[str]:
        """Drop pending changes for some files (or all). Recorded as a new version, so /rewind can bring them back."""
        pending = self.plan.b.pending
        gone = sorted(pending) if not paths else sorted(p for p in paths if p in pending)
        for p in gone:
            pending.pop(p)
        if gone:
            self.plan.b.versions.append(Version(hashlib.sha1(f"{time.time()}reject".encode()).hexdigest()[:7], "rejected " + ", ".join(gone)[:70], time.time(),   # noqa: S324
                                                pending, self.plan.b.commands if pending else []))
            self.save()
        return gone

    def rewind(self, steps: int = 1, to: str = "") -> int:
        """Go back n versions, or to a version id from /log. Returns how many were undone."""
        vs = self.plan.b.versions
        if to:
            idx = next((i for i, v in enumerate(vs) if v.id.startswith(to)), None)
            if idx is None:
                raise ValueError(f"no version {to}: see \\log")
            n = len(vs) - 1 - idx
        else:
            n = min(max(steps, 1), len(vs))
        del vs[len(vs) - n:]
        self.save()
        return n

    def log(self) -> list[Version]:
        return list(self.plan.b.versions)

    # -- apply, check, commit -------------------------------------------------------------------------------------------------------
    def apply(self, run_command, debug=None, commit_message=None) -> ApplyResult:
        """Write the pending changes, run the checking commands (`run_command(cmd) -> (ok, output)`, which asks the person when it must),
        and roll every file back if one fails. With `debug` (auto-debug), the Build agent is asked to fix the failure and it is tried again."""
        res = ApplyResult()
        for attempt in range(DEBUG_TRIES + 1):
            pending, commands = self.plan.b.pending, self.plan.b.commands
            if not pending:
                res.message = "nothing pending"
                return res
            snapshot = {p: self.disk(p) for p in pending}
            written = self.session.apply([_change(p, snapshot[p], new) for p, new in pending.items()])
            res.written = [p for p, status in written if status == "written"]
            failed, output = "", ""
            if self.plan.setting("can-exec"):
                for cmd in commands:
                    ok, output = run_command(cmd)
                    if ok is None:                  # the person declined: the changes stay applied, unchecked
                        continue
                    if not ok:
                        failed = cmd
                        break
            if not failed:
                break
            self._roll_back(snapshot)
            res.failed_command, res.output, res.rolled_back = failed, output[-4000:], True
            if not debug or attempt == DEBUG_TRIES:
                return res
            fixed = debug(failed, output[-3000:])
            if not fixed:
                return res
        self.plan.b.applied.append({"t": time.time(), "files": res.written, "commit": ""})
        self.plan.b.versions.append(Version(hashlib.sha1(f"{time.time()}applied".encode()).hexdigest()[:7], "applied " + ", ".join(res.written)[:70], time.time(), {}, []))   # noqa: S324
        res.failed_command, res.rolled_back = "", False
        if self.plan.setting("auto-commit") and commit_message and is_git_repo(self.session.cwd):
            res.commit = git_commit(self.session.cwd, res.written, commit_message(res.written))
            self.plan.b.applied[-1]["commit"] = res.commit
        self.save()
        return res

    def _roll_back(self, snapshot: dict[str, str | None]) -> None:
        for path, old in snapshot.items():
            p = self.session.cwd / path
            if old is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(old, encoding="utf-8", newline="")


def _change(path: str, old: str | None, new: str):
    from pal.terminal.session import Change, diff_lines
    return Change(path, old or "", new, old is None, diff_lines(old or "", new), review_code(path, new))


def parse_steps(text: str) -> list[str]:
    """Numbered lines (1. / 1) / Step 1:) from a plan, in order."""
    steps = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:step\s*)?(\d{1,2})[.):]\s+(.+)$", line, re.I)
        if m:
            steps.append(re.sub(r"\*\*|`", "", m.group(2)).strip())
    return steps


def mentioned_files(text: str, root: Path) -> list[str]:
    """Paths in the text that are files in the project (what a plan says it needs)."""
    found = []
    for m in re.finditer(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,6})(?![\w/])", text):
        rel = m.group(1).strip("`'\".,;:()")
        p = (root / rel)
        try:
            inside = root.resolve() in p.resolve().parents
        except OSError:
            inside = False
        if inside and p.is_file() and rel not in found:
            found.append(rel)
    return found[:12]


def is_git_repo(folder: Path) -> bool:
    """Inside a git work tree (the folder itself or any folder above it)."""
    git = shutil.which("git")
    if git:
        try:
            r = subprocess.run([git, "-C", str(folder), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True, timeout=10)
            return r.returncode == 0 and r.stdout.strip() == "true"
        except (OSError, subprocess.SubprocessError):
            pass
    return _has_dot_git(folder)


def _has_dot_git(folder: Path) -> bool:
    return shutil.which("git") is not None and (folder / ".git").exists()


def git_commit(folder: Path, files: list[str], message: str) -> str:
    """Commit just these files. Returns the short commit id, or '' if git refused."""
    git = shutil.which("git")
    if not git or not files:
        return ""
    try:
        subprocess.run([git, "-C", str(folder), "add", "--", *files], check=True, capture_output=True, timeout=30)
        r = subprocess.run([git, "-C", str(folder), "commit", "-m", message, "--", *files], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return ""
        return subprocess.run([git, "-C", str(folder), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def run_shell(cmd: str, cwd: Path, timeout: int = 300) -> tuple[bool, str]:
    """Run one approved command in the project folder. The person has seen and accepted the exact text."""
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}     # a file Pal just rewrote must not be run from a stale .pyc of the same size and second
        r = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env)   # noqa: S602  (shown to and approved by the person)
        return r.returncode == 0, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except OSError as e:
        return False, str(e)
