"""pal cli: the REPL. The welcome, the prompt (⚡️ tell, 💬 chat), backslash commands with suggestions as you type, multi-line mode, and what
runs when a prompt is sent: the live reply and build view, then the changes menu or the automatic apply."""
from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pal import __version__
from pal.paths import pal_home
from pal.terminal.repl import changes, cmds, stream, term
from pal.terminal.repl.term import BOLD, RED_HI, WHITE, Exit, accent, c, out, print_cmds
from pal.terminal.workflow import Store, Workflow, project_id

BUILTIN_DESCS = [("tell", "switch to tell mode (the Build agent writes code)"), ("chat", "switch to chat mode (the Ask agent, no changes)"),
                 ("multi", "multi-line mode on or off"), ("send", "send the current prompt"), ("run", "run a file through tell/chat"),
                 ("help", "REPL info and list of commands"), ("quit", "exit")]
BUILTINS = {"quit": "q", "help": "h", "multi": "m", "send": "s", "tell": "t", "chat": "ch", "run": "r"}
BUILTIN_ALIAS = {v: k for k, v in BUILTINS.items()}
EXTRA_ALIASES = {"diffs": "diff", "add": "load", "remove": "rm", "unload": "rm", "list-context": "ls", "history": "log", "logs": "log",
                 "set-plan": "cd", "set-models": "set-model", "n": "new"}
SWITCH = re.compile(r"(switch|start|continue|begin|change|chang|move|proceed|go|transition)(ing)?( to | with | into )?(tell|implementation|coding|development)( mode)?", re.I)


def tok(text: str) -> str:
    """A command token as the welcome and hints show it."""
    return c(f"{BOLD};{term.HI_CYAN}", text)


# -- state kept per folder ---------------------------------------------------------------------------------------------------------------
class State:
    """The mode follows the agent chosen in the chatbox: Build is tell, Ask is chat, Plan plans. Multi-line is kept per folder."""
    AGENT = {"tell": "build", "chat": "ask", "plan": "plan"}

    def __init__(self, folder: Path, who=None):
        self.path = pal_home() / "projects" / project_id(folder) / "repl.json"
        self.who = who
        self._mode, self.multi = "tell", False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.multi = bool(raw.get("multi"))
        except (OSError, ValueError, AttributeError):
            pass

    @property
    def mode(self) -> str:
        if self.who is None:
            return self._mode
        agent = self.who().mode
        return {"build": "tell", "ask": "chat", "plan": "plan"}.get(agent, "tell")

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value
        if self.who is not None:
            from pal.terminal import backend
            self.who().mode = self.AGENT.get(value, "build")
            backend.save_prefs(mode=self.AGENT.get(value, "build"))

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"multi": self.multi}), encoding="utf-8")
        except OSError:
            pass


# -- the context every command works with ---------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, args, core_factory: Callable | None, boot, ready: dict, prep: threading.Thread):
        self.args, self.core_factory, self.boot, self.ready, self.prep = args, core_factory, boot, ready, prep
        self.session: Any = None                           # the PalSession, once the background setup is done
        store = Store(boot.cwd)
        self.after_new = store.current_name() is None
        name = store.current_name()
        self.wf = Workflow(boot, store, plan=name)          # the plan is ready at once; the model joins when the background setup is done
        self.state = State(boot.cwd, who=self.session_or_boot)
        self.preserved = ""
        self.found_models: list | None = None

    # -- the model -------------------------------------------------------------------------------------------------------------------------
    def session_or_boot(self):
        return self.session or self.boot

    def model_name(self) -> str:
        prov, name = self.session_or_boot().model_for("build")
        return f"{prov}:{name}" if prov and prov != "local" else name

    def ensure(self):
        """The real session (waits for the background setup the first time)."""
        if self.session is None:
            from pal.terminal import cli
            if self.prep.is_alive():
                term.spinner.start("")
                self.prep.join()
                term.spinner.stop()
            if self.ready.get("need_model"):
                self._first_model()
            quiet = cli.ui.make_console()
            s = cli.finish_session(self.args, self.core_factory, self.ready, quiet, self.boot)
            if s is None:
                raise Exit(1)
            self.session = s
            self.wf.session = s
            self.wf._attach()
        return self.session

    def _first_model(self) -> None:
        """No model yet: pick one from the same list as \\set-model (downloaded if needed), then the session starts with it."""
        from pal.terminal import backend, cli
        out(c(f"{BOLD};{WHITE}", "\U0001F9E0 Choose a model to start"))
        choices = self.model_choices()
        labels = [label + (f"  ({about})" if about else "") for label, about, _ in choices]
        try:
            pick = term.select("Select a model:", labels)
        except term.Interrupted:
            raise Exit(1) from None
        spec = choices[labels.index(pick)][2]
        if spec.startswith("\x00"):
            try:
                spec = term.ask_text("Model (provider:model, for example openai:gpt-4o)")
            except term.Interrupted:
                raise Exit(1) from None
        provider, name = backend.split_spec(spec)
        if provider in backend.PROVIDER_KEYS and not os.environ.get(backend.PROVIDER_KEYS[provider]):
            if not cli.ask_key(cli.ui.make_console(), provider):
                raise Exit(1)
        if provider == backend.LOCAL and name not in backend.chat_models(backend.local_info(self.args.host or backend.DEFAULT_HOST)):
            from pal.terminal.setup import pull
            out(f"\U0001F4E5 Downloading {name} (once)...")
            if not pull(name):
                term.fail(f"The download of {name} did not finish")
        backend.save_prefs(provider=provider, model=name)
        self.args.provider, self.args.model = provider, name
        self.boot.provider, self.boot.model = provider, name
        self.ready["need_model"] = False

    def model_choices(self) -> list:
        from pal.terminal import cli
        return cli.model_choices(self.args, self.session_or_boot().model_for())

    def switch_model(self, spec: str) -> str:
        from rich.console import Console

        from pal.terminal import backend, cli
        s = self.ensure()
        provider, name = backend.split_spec(spec)
        if provider in backend.PROVIDER_KEYS and not os.environ.get(backend.PROVIDER_KEYS[provider]) and self.core_factory is None:
            if not cli.ask_key(cli.ui.make_console(), provider):
                return f"No key for {provider}: set {backend.PROVIDER_KEYS[provider]}, or \\key {provider} <key>"
        if provider == backend.LOCAL and self.core_factory is None and name not in backend.chat_models(backend.local_info(self.args.host or backend.DEFAULT_HOST)):
            from pal.terminal.setup import pull
            out(f"\U0001F4E5 Downloading {name} (once)...")
            if not pull(name):
                return f"The download of {name} did not finish"
        buf = io.StringIO()
        fresh = cli._switch(self.args, s, spec, self.core_factory, Console(file=buf, width=200))
        if fresh is s:
            return term.strip(buf.getvalue()).strip() or f"Could not switch to {spec}"
        self.session = fresh
        self.wf.session = fresh
        self.wf._attach()
        return ""

    # -- plans ---------------------------------------------------------------------------------------------------------------------------
    def _fresh_session(self) -> None:
        if self.session is not None:
            from pal.terminal import cli
            self.session = cli._fresh(self.session, self.args, self.core_factory)
            self.wf.session = self.session
            self.wf._attach()

    def new_plan(self, name: str = "") -> str:
        made = self.wf.new(name)
        self._fresh_session()
        return made

    def switch_plan(self, name: str) -> None:
        self.wf.switch(name)
        self._fresh_session()

    def after_plan_change(self) -> None:
        self._fresh_session()

    # -- sending a prompt ------------------------------------------------------------------------------------------------------------------
    def tell(self, send: str, shown: str, mode: str = "build", apply_debug: bool = False) -> str:
        """One request in the live view. Returns how it ended: "finished", "stopped", "background" or "error"."""
        self.ensure()
        running = stream.active()
        if running is not None:
            term.fail("Prompt error: a reply is still being written: \\connect shows it, \\stop stops it")
        cfg = self.wf.plan.config()
        can_bg = not (cfg["auto-apply"] or cfg["auto-load-context"] or apply_debug)
        st = stream.start(self.wf, send, shown, mode, can_bg=can_bg)
        how = stream.show(st)
        if how == "error":
            from pal.errors import PalError
            e = st.error
            msg = str(e) if isinstance(e, PalError) else f"{type(e).__name__}: {e}"
            term.fail(f"Stream error: {msg}")
        if how in ("stopped", "background"):
            raise Exit(0)
        return how

    def follow(self, st, mode: str = "build") -> None:
        how = stream.show(st)
        if how == "error":
            term.fail(f"Stream error: {st.error}")
        if how == "finished" and st.mode == "build":
            cfg = self.wf.plan.config()
            changes.after_reply(self, cfg["auto-apply"], cfg["skip-changes-menu"])


# -- the welcome --------------------------------------------------------------------------------------------------------------------------
def print_logo(kind: str = "cli") -> None:
    from pal.terminal.logo import banner
    rows = term.term_height()
    cols = term.term_width()
    size = "small" if rows >= 40 and cols >= 90 else "word"
    lines = banner(size, kind)
    if sum(len(ch) for ch, _ in lines[0]) > cols:
        return
    for line in lines:
        s = ""
        for chars, style in line:
            fg, _, bg = (style or "").partition("|")
            params = []
            if fg:
                params.append("38;2;" + ";".join(str(int(fg[i:i + 2], 16)) for i in (1, 3, 5)))
            if bg:
                params.append("48;2;" + ";".join(str(int(bg[i:i + 2], 16)) for i in (1, 3, 5)))
            s += c(";".join(params), chars) if params else chars
        out(s.rstrip())


def show_mode(ctx: Ctx) -> None:
    out()
    if ctx.state.mode == "tell":
        out(term.label_white(" ⚡️ Tell mode is enabled "))
        out(f"{tok(chr(92) + 'chat')} (\\ch) switch to chat mode to chat without writing code or making changes")
    else:
        out(term.label_white(" \U0001F4AC Chat mode is enabled "))
        out(f"{tok(chr(92) + 'tell')} (\\t) switch to tell mode to start writing code and implementing tasks")
    out()


def show_multi(ctx: Ctx) -> None:
    if ctx.state.multi:
        out(term.label_white(" \U0001F522 Multi-line mode is enabled "))
        out(f"{tok(chr(92) + 'multi')} to exit multi-line mode")
        out(f"{tok('enter')} for line breaks")
        out(f"{tok(chr(92) + 'send')} to send prompt")
    else:
        out(term.label_white(" 1️⃣  Multi-line mode is disabled "))
        out(f"{tok(chr(92) + 'multi')} for multi-line editing mode")
        out(f"{tok('enter')} to send prompt")


def welcome(ctx: Ctx, is_help: bool = False) -> None:
    from pal.terminal import ui
    wf = ctx.wf
    cfg = wf.plan.config()
    if not ctx.after_new or is_help:
        out()
    head = term.label(" \U0001F44B Welcome to Pal Code ") + c(f"{WHITE};{term.BG_HI_BLACK}", f" v{__version__} ")
    out(head + "  " + term.muted(ui.short_path(ctx.boot.cwd)))
    out()
    out(cmds.current_plan_table(wf))
    out(f"{tok(chr(92))} for commands")
    out(f"{tok('@')} for loading files into context" + (" manually (optional)" if cfg["auto-load-context"] else ""))
    out(f"{tok(chr(92) + 'help')} (\\h) for help")
    out(f"{tok(chr(92) + 'quit')} (\\q) to exit")
    out()
    out(cmds.auto_mode_table(cfg), end="")
    out(f"{tok(chr(92) + 'set-auto')} to change auto mode")
    out(f"{tok(chr(92) + 'config')} to see config")
    out(f"{tok(chr(92) + 'set-config')} to customize config")
    out()
    out(cmds.model_table(ctx.model_name()), end="")
    out(f"{tok(chr(92) + 'models')} to see model details")
    out(f"{tok(chr(92) + 'set-model')} to change models")
    show_mode(ctx)
    show_multi(ctx)
    out()
    if not is_help:
        out(term.label(" Describe a coding task \U0001F447 " if ctx.state.mode == "tell" else " Ask a question or chat \U0001F447 "))
        out()


def startup(ctx: Ctx) -> None:
    """Flags only: the first screen is the chatbox with the logo (the plan, its auto mode and the model are in \\current, \\config, \\models)."""
    flag = next((m for m in ("none", "basic", "plus", "semi", "full") if getattr(ctx.args, f"auto_{m}", False)), "")
    if getattr(ctx.args, "chat", False) and getattr(ctx.args, "tell", False):
        term.err(c(f"{RED_HI};{BOLD}", "\U0001F6A8 Cannot specify both --chat and --tell flags"))
        raise Exit(1)
    if getattr(ctx.args, "chat", False):
        ctx.state.mode = "chat"
    elif getattr(ctx.args, "tell", False):
        ctx.state.mode = "tell"
    wf = ctx.wf
    if flag and flag != wf.plan.autonomy:
        wf.plan.set_auto(flag)
        wf.save()
    if ctx.wf.notice:
        ctx.boot.notice = ctx.wf.notice


# -- help -----------------------------------------------------------------------------------------------------------------------------------
SECTIONS = [
    ("Key Commands", ["new", "load", "diff", "diff --ui", "apply", "reject", "debug"]),
    ("Plans", ["new", "plans", "cd", "current", "delete-plan", "rename", "archive", "plans --archived", "unarchive"]),
    ("Changes", ["diff", "diff --ui", "apply", "reject"]),
    ("Context", ["load", "ls", "rm", "update", "clear", "show"]),
    ("Branches", ["branches", "checkout", "delete-branch"]),
    ("History", ["log", "rewind", "convo", "summary"]),
    ("Control", ["continue", "build", "debug"]),
    ("Streams", ["ps", "connect", "stop"]),
    ("Config", ["config", "set-config", "config default", "set-config default"]),
    ("Autonomy", ["set-auto", "set-auto default", "set-auto full", "set-auto semi", "set-auto plus", "set-auto basic", "set-auto none"]),
    ("AI Models", ["models", "models available", "providers", "set-model", "set-model default", "key", "endpoint"]),
    ("Tokens", ["tokens", "budget"]),
    ("Voice and Security", ["voice", "security"]),
    ("New Plan Shortcuts", ["new --full", "new --semi", "new --plus", "new --basic", "new --none"]),
]


def print_help() -> None:
    for i, (title, names) in enumerate(SECTIONS):
        bg = (term.BG_MAGENTA if i == 0 else term.BG_CYAN) + ";" + WHITE
        out(c(f"{BOLD};{bg}", f" {title} "))
        for n in names:
            style = f"{BOLD};{term.HI_MAGENTA}" if i == 0 else f"{BOLD};{term.HI_CYAN}"
            out(" " + c(style, f" {term.cmd_label(n)} ") + " \U0001F449 " + term.DESCS.get(n, ""))
        out()
    out()


# -- suggestions while typing ----------------------------------------------------------------------------------------------------------
def project_paths(root: Path, limit: int = 3000) -> list[tuple[str, bool]]:
    from pal.terminal.session import SKIP_DIRS
    found: list[tuple[str, bool]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            rel = Path(dirpath).relative_to(root)
            for d in dirnames:
                found.append(((rel / d).as_posix(), True))
            for f in sorted(filenames):
                if not f.startswith("."):
                    found.append(((rel / f).as_posix(), False))
            if len(found) > limit:
                break
    except OSError:
        pass
    return found


def suggestions(ctx: Ctx, paths: list[tuple[str, bool]]) -> list[tuple[str, str]]:
    st = ctx.state
    out_: list[tuple[str, str]] = []
    if st.multi:
        out_ += [("\\send", "(\\s) Send the current prompt"), ("\\multi", "(\\m) Turn multi-line mode off"),
                 ("\\run", "(\\r) Run a file through tell/chat based on current mode"), ("\\quit", "(\\q) Exit the REPL")]
    if st.mode == "tell":
        out_.append(("\\chat", "(\\ch) Switch to 'chat' mode to have a conversation without making changes"))
    else:
        out_.append(("\\tell", "(\\t) Switch to 'tell' mode for implementation"))
    if not st.multi:
        out_ += [("\\multi", "(\\m) Turn multi-line mode on"), ("\\run", "(\\r) Run a file through tell/chat based on current mode"),
                 ("\\quit", "(\\q) Exit the REPL")]
    out_.append(("\\help", "(\\h) REPL info and list of commands"))
    for cmd, alias, desc in term.COMMANDS:
        out_.append(("\\" + cmd, f"(\\{alias}) {desc}" if alias else desc))
    for p, is_dir in paths:
        shown = p + ("/" if is_dir else "")
        out_.append(("@" + shown, ""))
        out_.append(("\\load " + shown + (" -r" if is_dir else ""), ""))
        if is_dir:
            out_.append(("\\load " + shown + " --tree", ""))
        if p.endswith((".md", ".txt")):
            out_.append(("\\run " + shown, ""))
    return out_


def fuzzy(needle: str, hay: str) -> bool:
    it = iter(hay.upper())
    return all(ch in it for ch in needle.upper())


def complete(ctx: Ctx, text: str, cursor: int, paths: list[tuple[str, bool]]) -> tuple[list[tuple[str, str]], int]:
    """(suggestions, start index) for the text before the cursor."""
    lines = text.split("\n")
    if text[:cursor].count("\n") != len(lines) - 1:
        return [], 0
    last = lines[-1]
    if not last.strip() and len(lines) >= 2:
        last = lines[-2]
    valid = [i for i, ch in enumerate(last) if ch in "\\@" and (i == 0 or last[i - 1].isspace())]
    if not valid:
        return [], 0
    w = last[valid[-1]:]
    start = len(text) - len(w)
    if not last.strip().endswith(w.strip()):
        return [], 0
    w_trim = w[1:].strip() if w.startswith("\\") else w.strip()
    w_cmd = w_trim.split(" ")[0] if w_trim else ""
    if w.startswith("\\"):
        ok = any(cmd.startswith(w_cmd) for cmd, _, _ in term.COMMANDS) or any(a.startswith(w_cmd) for a in term.ALIASES) \
            or any(b.startswith(w_cmd) for b in ("quit", "multi", "tell", "chat", "send", "run", "help"))
        if not ok and w_cmd:
            return [], 0
    items = suggestions(ctx, paths)
    fz = [s for s in items if fuzzy(w, s[0])]
    px = [s for s in items if s[0].lower().startswith(w.lower())]
    if w_cmd != "run":
        fz = [s for s in fz if not s[0].startswith("\\run ")]
        px = [s for s in px if not s[0].startswith("\\run ")]
    if w_cmd != "load":
        fz = [s for s in fz if not s[0].startswith("\\load ")]
        px = [s for s in px if not s[0].startswith("\\load ")]
    if w.strip() != "\\":
        first = ("\\chat", "\\tell", "\\multi", "\\quit", "\\send", "\\run")
        px.sort(key=lambda s: (s[0] not in first, s[0]))
    result = px + [s for s in fz if s not in px] if px else fz
    target = ""
    if w_trim in BUILTIN_ALIAS:
        target = "\\" + BUILTIN_ALIAS[w_trim]
    elif w_trim in term.ALIASES:
        target = "\\" + term.ALIASES[w_trim]
    if target:
        hit = [s for s in result if s[0] == target]
        if hit:
            result = hit + [s for s in result if s[0] != target]
    return result, start


# -- the prompt -------------------------------------------------------------------------------------------------------------------------
def prefix(ctx: Ctx) -> str:
    if ctx.state.mode == "tell":
        cfg = ctx.wf.plan.config()
        return "⚡️❗️ " if cfg["auto-apply"] and cfg["auto-exec"] else "⚡️ "
    return "\U0001F4AC "


def parse_command(ctx: Ctx, text: str, paths) -> str:
    """Would Enter run this as a command (rather than add a line in multi-line mode)?"""
    text = text.strip()
    if not text:
        return ""
    last = text.split("\n")[-1].strip()
    bs, at = last.rfind("\\"), last.rfind("@")
    if at > bs:
        names = {p for p, _ in paths} | {p + "/" for p, d in paths if d}
        segs = [s.strip() for s in last.split("@")[1:]]
        if any(s in names for s in segs):
            return "@"
    if bs >= 0:
        cmd_string = last[bs + 1:].strip()
        if not cmd_string:
            return ""
        first = cmd_string.split()[0]
        if first in BUILTINS or first in BUILTIN_ALIAS:
            return first
        if resolve(first):
            return first
        if any(cmd.split()[0].startswith(first) for cmd, _, _ in term.COMMANDS):
            return first
    return ""


def resolve(word: str) -> str:
    """The command a typed word names (exact name or alias), or ''."""
    names = {cmd.split()[0] for cmd, _, _ in term.COMMANDS}
    if word in names:
        return word
    if word in term.ALIASES:
        return term.ALIASES[word].split()[0]
    return EXTRA_ALIASES.get(word, "")


def build_prompt(ctx: Ctx):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    from pal.terminal.history import History
    hist = History(ctx.boot.cwd)
    mem = InMemoryHistory()
    for item in hist.items:
        mem.append_string(item)
    cache: dict = {"paths": None, "t": 0.0}

    def paths():
        if cache["paths"] is None or time.time() - cache["t"] > 10:
            cache["paths"], cache["t"] = project_paths(ctx.boot.cwd), time.time()
        return cache["paths"]

    class C(Completer):
        def get_completions(self, document, complete_event):
            if document.cursor_position_row != document.line_count - 1:
                return
            before = document.text_before_cursor
            items, start = complete(ctx, before, len(before), paths())
            if start > len(before):
                return
            for text, meta in items[:400]:
                yield Completion(text, start_position=start - len(before), display_meta=meta or None)

    kb = KeyBindings()

    @kb.add("enter")
    @kb.add("c-j")
    def _(e):
        buf = e.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
        if parse_command(ctx, buf.text, paths()) or not ctx.state.multi:
            buf.validate_and_handle()
        else:
            buf.insert_text("\n")

    @kb.add("c-c")
    def _(e):
        e.app.exit(exception=KeyboardInterrupt())

    style = Style.from_dict({
        "completion-menu.completion": "bg:#3a3a3a #ffffff",
        "completion-menu.completion.current": "bg:#d0d0d0 #000000 bold",
        "completion-menu.meta.completion": "bg:#E50914 #ffffff",
        "completion-menu.meta.completion.current": "bg:#78060c #ffffff",
        "scrollbar.background": "bg:#78060c",
        "scrollbar.button": "bg:#3a3a3a",
    })

    def message():
        return ANSI(c(term.HI_BLUE, prefix(ctx)))

    def continuation(width, line_number, is_soft_wrap):
        n = term.width(prefix(ctx).rstrip())
        return "." * n + " "

    session: PromptSession = PromptSession(message=message, completer=C(), complete_while_typing=True, history=mem, key_bindings=kb, style=style,
                            prompt_continuation=continuation, reserve_space_for_menu=6, enable_history_search=False, multiline=True)
    return session, hist, paths


def similar(word: str, targets: list[str]) -> list[str]:
    w = word.lower().strip().strip("/")
    found = []
    for t in targets:
        if not fuzzy(w, t) or not w:
            continue
        if w in t or t in w or _lev(w, t) <= (1 if max(len(w), len(t)) < 5 else 4):
            found.append(t)
    return found


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def prompt_option(s: str) -> str:
    return f"Send '{s[:20] + '...' if len(s) > 20 else s}' as a prompt to the AI model"


def suggest_cmds(matches: list[str], typed: str) -> str | None:
    """'Did you mean' menu. Returns the command, "" to send as a prompt, or None to cancel."""
    out()
    opt = prompt_option(typed)
    try:
        pick = term.select("\U0001F914 Did you mean to type one of these commands?", ["\\" + m for m in matches] + ["Cancel", opt])
    except term.Interrupted:
        return None
    if pick == "Cancel":
        return None
    if pick == opt:
        return ""
    return pick[1:]


# -- running what was typed ------------------------------------------------------------------------------------------------------------------
HANDLERS: dict[str, Callable] = {
    "new": cmds.cmd_new, "plans": cmds.cmd_plans, "cd": cmds.cmd_cd, "current": cmds.cmd_current, "rename": cmds.cmd_rename,
    "delete-plan": cmds.cmd_delete_plan, "archive": cmds.cmd_archive, "unarchive": cmds.cmd_unarchive, "config": cmds.cmd_config,
    "set-config": cmds.cmd_set_config, "set-auto": cmds.cmd_set_auto, "load": cmds.cmd_load, "ls": cmds.cmd_ls, "rm": cmds.cmd_rm,
    "clear": cmds.cmd_clear, "show": cmds.cmd_show, "update": cmds.cmd_update, "branches": cmds.cmd_branches, "checkout": cmds.cmd_checkout,
    "delete-branch": cmds.cmd_delete_branch, "log": cmds.cmd_log, "rewind": cmds.cmd_rewind, "convo": cmds.cmd_convo, "summary": cmds.cmd_summary,
    "models": cmds.cmd_models, "set-model": cmds.cmd_set_model, "providers": cmds.cmd_providers, "ps": cmds.cmd_ps, "stop": cmds.cmd_stop,
    "connect": cmds.cmd_connect,
}


SAFE_WHILE_ACTIVE = {"ps", "stop", "connect", "help", "log", "convo", "ls", "show", "config", "models", "providers", "plans", "current",
                     "branches", "summary", "diff", "tokens", "voice", "key"}


def run_command(ctx: Ctx, name: str, args: list[str]) -> None:
    if name not in SAFE_WHILE_ACTIVE and stream.active() is not None:
        if name == "apply":
            out("This plan is currently active. Please wait for it to finish before applying.")
        else:
            out("This plan is currently active. Please wait for it to finish, or stop it.")
        out()
        print_cmds("ps", "connect", "stop")
        return
    if name in HANDLERS:
        HANDLERS[name](ctx, args)
    elif name == "diff":
        if "--ui" in args:
            changes.diff_ui(ctx.wf, side="--line" not in args and "-l" not in args)
        else:
            changes.diff_git(ctx.wf, plain="--plain" in args or "-p" in args)
    elif name == "apply":
        debug = None
        for a in args:
            if a.startswith("--debug"):
                debug = int(a.split("=", 1)[1]) if "=" in a else 5
        ctx.ensure()
        changes.apply_cmd(ctx, full="--full" in args, commit=True if ("--commit" in args or "-c" in args) else None,
                          no_exec=True if "--no-exec" in args else None, auto_exec=True if "--auto-exec" in args else None, debug=debug,
                          skip="--skip-commit" in args)
    elif name == "reject":
        changes.reject(ctx.wf, [a for a in args if not a.startswith("-")], all_="--all" in args or "-a" in args)
    elif name == "debug":
        ctx.ensure()
        changes.debug_cmd(ctx, args)
    elif name == "continue":
        cont(ctx)
    elif name == "build":
        out("\U0001F937‍♂️ This plan has no pending changes to build")
        out()
        print_cmds("log", "continue")
    elif name == "voice":
        voice(ctx)
    elif name == "key":
        key(args)
    elif name == "security":
        security(ctx)
    elif name == "tokens":
        tokens(ctx)
    elif name == "budget":
        budget(ctx, args)
    elif name == "endpoint":
        endpoint(args)
    else:
        raise KeyError(name)


def cont(ctx: Ctx) -> None:
    wf = ctx.wf
    nxt = wf.next_step()
    if nxt is not None:
        i, step = nxt
        send = f"Implement step {i} of {len(wf.plan.b.steps)} of the plan: {step}. Only this step; keep everything else as it is."
        how = ctx.tell(send, "", mode="build")
        wf.plan.b.done = i
        wf.save()
    elif wf.plan.b.convo:
        how = ctx.tell("Continue the plan: implement what is left, or say that it is done.", "", mode="build")
    else:
        out("\U0001F937‍♂️ There's no plan yet to continue")
        out()
        raise Exit(0)
    if how == "finished":
        cfg = wf.plan.config()
        changes.after_reply(ctx, cfg["auto-apply"], cfg["skip-changes-menu"])


def voice(ctx: Ctx) -> None:
    from pal.terminal import cli
    heard = cli.voice_flow(cli.ui.make_console())
    if heard:
        ctx.preserved = (ctx.preserved + " " + heard).strip()


def key(args: list[str]) -> None:
    from pal.terminal import backend
    if len(args) == 2 and args[0] == "custom":
        os.environ["PAL_CUSTOM_API_KEY"] = args[1]
        out("✅ Key for the custom endpoint set for this session (not saved)")
        return
    if len(args) != 2 or args[0] not in backend.PROVIDER_KEYS:
        term.fail(f"Use: \\key <provider> <key>   providers: {', '.join(backend.PROVIDER_KEYS)}")
    os.environ[backend.PROVIDER_KEYS[args[0]]] = args[1]
    out(f"✅ Key for {args[0]} set for this session (not saved)")
    out()
    print_cmds("set-model")


def tokens(ctx: Ctx) -> None:
    from pal.terminal.ui import LABELS
    s = ctx.ensure()
    rows = [[LABELS.get(k, k), str(v)] for k, v in sorted(s.saved_by_total().items(), key=lambda kv: -kv[1]) if v]
    out(accent("\U0001FA99 Tokens this session"))
    out(term.table([["Used by Pal", str(s.pal_tokens)], ["A plain chat would use (estimate)", str(s.plain_tokens)],
                    ["Saved", f"{s.saved} ({s.saved_pct:g}%)"], ["Pal's helper jobs", str(s.helper_tokens)]], header=["", "Tokens"]), end="")
    if rows:
        out(term.table(rows, header=["Saved by", "Tokens"]), end="")


def budget(ctx: Ctx, args: list[str]) -> None:
    s = ctx.ensure()
    if args and args[0] in ("off", "0", "none"):
        s.budget = 0
        out("\u2705 No token cap for this session")
    elif args and args[0].replace(",", "").replace("_", "").isdigit():
        s.budget = int(args[0].replace(",", "").replace("_", ""))
        out(f"\u2705 Token cap: {s.budget} for this session ({s.pal_tokens} used so far)")
    else:
        out(f"Token cap: {s.budget or 'none'}. Set one with \\budget 50000, or \\budget off")


def endpoint(args: list[str]) -> None:
    from pal.terminal import backend
    if args and args[0].startswith(("http://", "https://")):
        backend.save_prefs(custom_base_url=args[0])
        out(f"\u2705 Custom endpoint: {args[0]} (the key comes from PAL_CUSTOM_API_KEY or \\key custom <key>)")
        out()
        print_cmds("set-model")
    else:
        out(f"Custom endpoint: {backend.custom_base_url() or 'none'}. Set one with \\endpoint https://your-server/v1, then \\set-model custom:<model>")


def security(ctx: Ctx) -> None:
    s = ctx.ensure()
    term.spinner.start("")
    findings, checked = s.audit()
    term.spinner.stop()
    if not findings:
        out(f"✅ {checked} files checked, nothing found")
        return
    rows = [[c(f"{BOLD};{RED_HI}", f.severity), f"{f.path}:{f.line}", f.title] for f in findings[:60]]
    out(c(f"{BOLD};{WHITE}", f"\U0001F6A8 {len(findings)} findings in {checked} files"))
    out(term.table(rows, header=["Severity", "Where", "What"], align=["L", "L", "L"]))


def send_prompt(ctx: Ctx, text: str) -> None:
    """A prompt (not a command): tell (Build) or chat (Ask), in the live view."""
    if getattr(ctx.args, "bench_screen", False):
        from pal.terminal import cli
        s = ctx.ensure()
        console = cli.ui.make_console()
        result = cli.bench_run(console, s.cwd, text, s.provider, s.model, ctx.args.host or cli.backend.DEFAULT_HOST, core_factory=ctx.core_factory)
        cli.show_bench(console, result, s.provider, s.model, s.cwd)
        cli.keep_bench_files(console, result, s.cwd)
        return
    if ctx.state.mode == "tell":
        how = ctx.tell(text, text, mode="build")
        if how == "finished":
            cfg = ctx.wf.plan.config()
            changes.after_reply(ctx, cfg["auto-apply"], cfg["skip-changes-menu"])
        return
    if ctx.state.mode == "plan":
        ctx.tell(text, text, mode="plan")
        if ctx.wf.plan.b.steps:
            out()
            print_cmds("continue")
        return
    ctx.tell(text, text, mode="ask")
    reply = ctx.wf.plan.b.convo[-1]["text"] if ctx.wf.plan.b.convo else ""
    if SWITCH.search(re.sub(r"['\"*\\`_]", "", reply)):
        out()
        if term.confirm("Switch to tell mode for implementation?"):
            ctx.state.mode = "tell"
            ctx.state.save()
            out()
            out(term.label_white(" ⚡️ Tell mode is enabled "))
            out()
            out("Now that you're in tell mode, you can either begin the implementation based on the conversation so far, or you can send another "
                "prompt to begin the implementation with additional information or instructions.")
            out()
            pick = term.select("What would you like to do?", ["Begin implementation", "Send another prompt"])
            if pick == "Begin implementation":
                out()
                how = ctx.tell("Go ahead with the plan based on what we've discussed so far.", "", mode="build")
                if how == "finished":
                    cfg = ctx.wf.plan.config()
                    changes.after_reply(ctx, cfg["auto-apply"], cfg["skip-changes-menu"])


def execute(ctx: Ctx, raw: str, paths: list[tuple[str, bool]]) -> bool:
    """One Enter. False to quit."""
    text = raw.strip()
    if not text:
        return True
    lines = text.split("\n")
    last = lines[-1].strip()
    " ".join(text.split())
    trimmed = text
    preserved = ("\n".join(lines[:-1]) + "\n") if len(lines) > 1 else ""
    bs, at = last.rfind("\\"), last.rfind("@")
    sugg, _ = complete(ctx, text, len(text), paths) if "\n" not in text else ([], 0)
    names = {p for p, _ in paths} | {p + "/" for p, d in paths if d}

    if at > bs:                                            # @path loads files into context
        segs = last.split("@")
        preserved += segs[0] + " " if segs[0].strip() else ""
        found = []
        for i, seg in enumerate(segs[1:], 1):
            p = seg.strip()
            if (not p or p not in names) and sugg and i == len(segs) - 1:
                found.append(sugg[0][0][1:])
            elif p in names:
                found.append(p)
        if found:
            out()
            cmds.cmd_load(ctx, found + ["-r"])
            out()
            ctx.preserved = preserved if preserved.strip() else ""
            return True

    def restore() -> None:
        nonlocal preserved
        if bs > 0:
            preserved += last[:bs]
        ctx.preserved = preserved if preserved.strip() else ""

    if bs != -1:
        cmd_string = last[bs + 1:].strip()
        if not cmd_string:
            return True
        parts = cmd_string.split()
        word, args = parts[0], parts[1:]
        if word in ("quit", "q"):
            return False
        if word in ("help", "h"):
            restore()
            welcome(ctx, is_help=True)
            print_help()
            return True
        if word in ("multi", "m"):
            restore()
            out()
            ctx.state.multi = not ctx.state.multi
            show_multi(ctx)
            ctx.state.save()
            out()
            return True
        if word in ("tell", "t"):
            restore()
            ctx.state.mode = "tell"
            ctx.state.save()
            show_mode(ctx)
            return True
        if word in ("chat", "ch"):
            restore()
            ctx.state.mode = "chat"
            ctx.state.save()
            show_mode(ctx)
            return True
        if word in ("run", "r"):
            restore()
            out()
            if len(args) != 1:
                out(c(RED_HI, "Run command failed: run command requires exactly one file path argument"))
            else:
                f = ctx.boot.cwd / args[0]
                if not f.is_file():
                    out(c(RED_HI, f"Run command failed: file does not exist: {args[0]}"))
                else:
                    send_prompt(ctx, f.read_text(encoding="utf-8", errors="replace"))
            out()
            return True
        if word in ("send", "s"):
            body = trimmed.split("\\s")[0].strip()
            if not body:
                out()
                out("\U0001F937‍♂️ No prompt to send")
                out()
                return True
            out()
            send_prompt(ctx, body)
            out()
            return True
        name = resolve(word)
        if not name and sugg:
            pick = sugg[0][0]
            if pick.startswith("\\"):
                parts = pick[1:].split()
                if parts[0] in BUILTINS:                    # \mul -> \multi
                    return execute(ctx, (last[:bs] + "\\" + parts[0]) if len(lines) == 1 else "\n".join(lines[:-1] + [last[:bs] + "\\" + parts[0]]), paths)
                name, args = resolve(parts[0]), parts[1:]
        if name:
            restore()
            out()
            run_command(ctx, name, args)
            out()
            return True
        builtins = list(BUILTINS)
        targets = builtins + [cmd for cmd, _, _ in term.COMMANDS]
        near = similar(word, [t for t in targets if t != word])
        if near:
            choice = suggest_cmds(near, word)
            if choice is None:
                return True
            if choice:
                return execute(ctx, "\\" + choice, paths)
        elif len(lines) == 1 and trimmed.startswith("\\"):
            try:
                pick = term.select("\U0001F914 Couldn't find a matching command. What do you want to do?",
                                   ["Cancel", "Show available commands", prompt_option(word)])
            except term.Interrupted:
                return True
            if pick == "Cancel":
                return True
            if pick == "Show available commands":
                welcome(ctx, is_help=True)
                print_help()
                return True
    elif len(lines) == 1:                                  # a command typed without the backslash?
        targets = list(BUILTINS) + [cmd for cmd, _, _ in term.COMMANDS]
        near = similar(last, targets)
        if near and len(last.split()) <= 3:
            choice = suggest_cmds(near, last)
            if choice is None:
                return True
            if choice:
                return execute(ctx, "\\" + choice, paths)
    dropped = _dropped(text)
    if dropped:                                            # files dropped on the window: into the context
        out()
        cmds.cmd_load(ctx, dropped)
        out()
        return True
    out()
    send_prompt(ctx, trimmed)
    out()
    return True


def _dropped(text: str) -> list[str]:
    from pal.terminal.cli import dropped_files
    return dropped_files(text)


def run(args, core_factory: Callable | None = None) -> int:
    from pal.terminal import backend, cli
    term.enable_vt()
    boot = cli.Boot(args, backend.load_prefs())
    ready: dict = {}
    prep = threading.Thread(target=cli.prepare_quietly, args=(args, core_factory, ready), daemon=True, name="pal-setup")
    prep.start()
    if core_factory is None:
        from pal.terminal import voice as _voice
        threading.Thread(target=_voice.preload, daemon=True, name="pal-voice").start()
    try:
        ctx = Ctx(args, core_factory, boot, ready, prep)
        startup(ctx)
    except Exit as e:
        return e.code
    sys.stdout.write(f"\x1b]2;pal code {__version__}\x07")
    from pal.terminal import ui
    from pal.terminal.history import History
    console = ui.make_console()
    ui.CMD_PREFIX = "\\"
    ui.MENU_SOURCE = lambda: [("\\" + b, d) for b, d in BUILTIN_DESCS] + [("\\" + cmd, d) for cmd, _, d in term.COMMANDS]
    hist = History(boot.cwd)
    found: dict = {"items": None}

    def look_for_models() -> None:
        try:
            found["items"] = ctx.model_choices()
        except Exception:  # noqa: BLE001  (Ollama gone: at least the API choice)
            found["items"] = [("API model\u2026", "type provider:model", cli.API_ITEM)]
    threading.Thread(target=look_for_models, daemon=True, name="pal-models").start()
    cache: dict = {"paths": None, "t": 0.0}

    def paths():
        if cache["paths"] is None or time.time() - cache["t"] > 10:
            cache["paths"], cache["t"] = project_paths(boot.cwd), time.time()
        return cache["paths"]

    first, open_models, listen_now = True, False, False
    code = 0
    try:
        while True:
            shown = ctx.session_or_boot()
            draft = ctx.preserved
            ctx.preserved = ""
            text = ui.read_input(shown, console, backend.save_prefs, landing=first, version=__version__, history=hist, initial=draft,
                                 models=lambda: found["items"], open_models=open_models, start_listening=listen_now)
            landing, open_models, listen_now = first, False, False
            first = False
            if text is None:
                break
            if text.startswith("\x00pick\x00"):                 # a model chosen in the box's dropdown
                spec, _, rest = text[len("\x00pick\x00"):].partition("\x00")
                ctx.preserved, first = rest, landing
                if spec == cli.API_ITEM:
                    ctx.preserved = "\\set-model "
                    continue
                try:
                    problem = ctx.switch_model(spec)
                    if problem:
                        out(c(f"{RED_HI};{BOLD}", f"\U0001F6A8 {problem}"))
                except Exit:
                    pass
                threading.Thread(target=look_for_models, daemon=True).start()
                continue
            if text.startswith("\x00model\x00"):               # the model clicked before its list was ready
                ctx.preserved, open_models, first = text[len("\x00model\x00"):], True, landing
                continue
            if text.startswith("\x00voice\x00"):
                ctx.preserved, listen_now, first = text[len("\x00voice\x00"):], True, landing
                continue
            if text.strip() in ("\\voice", "\\v", "/voice"):
                listen_now, first = True, landing
                continue
            if landing:                                           # leaving the first screen: Pal's header, the work goes on below it
                ui.show_header(console, boot.cwd, __version__, boot.brand)
            hist.add(text)
            raw = text.strip()
            if raw.startswith("/") and resolve(raw[1:].split()[0] if raw[1:].split() else ""):   # the old /commands still work
                raw = "\\" + raw[1:]
            out(c(term.HI_BLUE, prefix(ctx)) + raw.replace("\n", "\n" + "." * term.width(prefix(ctx).rstrip()) + " "))
            try:
                if not execute(ctx, raw, paths()):
                    break
            except Exit:
                out()
            except term.Interrupted:
                out()
            except KeyboardInterrupt:
                term.spinner.stop()
                out()
            except Exception as e:  # noqa: BLE001  (a command that broke never ends the REPL)
                term.spinner.stop()
                out(c(RED_HI, f"Error executing command: {e}"))
                out()
    finally:
        sys.stdout.write("\x1b]2;\x07")
        ctx.wf.close()
        if ctx.session is not None:
            ctx.session.close()
    return code
