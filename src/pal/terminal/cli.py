"""The `pal` command.

    pal cli                open a terminal session in this folder (Tab switches between the Build and Plan agents); plain `pal` does the same
    pal web                open the web interface in your default browser (it runs on this computer only)
    pal ask "question"     one answer from the Ask agent, then exit (text piped in is included; --plan or --build for the others)
    pal doctor             check this computer; also: pal config, pal tools, pal models (OpenRouter), pal agent, pal chat
    pal --provider openrouter --model <name>     use an API model (the key comes from the environment)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text

from pal import __version__
from pal.errors import PalError
from pal.llm_wrapper.modes import providers_with_keys
from pal.terminal import backend, flow_cli, theme, ui
from pal.terminal.local_models import api_only
from pal.terminal.session import AGENT_ABOUT, MODES, BudgetExceeded, PalSession, make_session
from pal.terminal.workflow import Workflow

if TYPE_CHECKING:
    from pal.terminal.project import Project

NO_MODEL = ("No model to talk to yet. Either:\n"
            "  1. start Ollama and pull a model:   ollama pull llama3.1:8b      (everything stays on this computer)\n"
            "  2. or use an API model:             set OPENROUTER_API_KEY (or OPENAI_API_KEY, ANTHROPIC_API_KEY ...), then\n"
            "                                      pal cli --provider openrouter --model <model name>")


def _common(p: argparse.ArgumentParser, suppress: bool = False) -> None:
    """The session options, accepted before or after `cli`/`ask` (`pal --model x cli` and `pal cli --model x` both work)."""
    d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
    p.add_argument("--provider", default=d(None), help="local (default), an API provider (openrouter, openai, anthropic, gemini, groq, deepseek, mistral, together), litellm (any LiteLLM model), custom (any OpenAI-compatible endpoint), lmstudio or vllm")
    p.add_argument("--model", default=d(None), help="the model name")
    p.add_argument("--mode", choices=MODES, default=d(None), help="start with the Build, Plan or Ask agent (default build)")
    p.add_argument("--host", default=d(None), help="Ollama address (default http://localhost:11434)")
    p.add_argument("--dir", default=d("."), help="the folder Pal works in (default: here)")
    p.add_argument("--budget", type=int, default=d(0), help="stop before this session uses more than this many tokens (default: no cap)")
    p.add_argument("--chat", "-c", action="store_true", default=d(False), help="start in chat mode (for conversation without making changes)")
    p.add_argument("--tell", "-t", action="store_true", default=d(False), help="start in tell mode (for implementation)")
    p.add_argument("--no-auto", dest="auto_none", action="store_true", default=d(False), help="None: step-by-step, no automation")
    p.add_argument("--basic", dest="auto_basic", action="store_true", default=d(False), help="Basic: auto-continue plans")
    p.add_argument("--plus", dest="auto_plus", action="store_true", default=d(False), help="Plus: auto-update context, smart context, auto-commit changes")
    p.add_argument("--semi", dest="auto_semi", action="store_true", default=d(False), help="Semi-Auto: auto-load context")
    p.add_argument("--full", dest="auto_full", action="store_true", default=d(False), help="Full-Auto: auto-apply, auto-exec, auto-debug")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pal", description="Pal: build and plan with fewer tokens, in your terminal (pal cli) or your browser (pal web).")
    p.add_argument("--version", action="version", version=f"pal {__version__}")
    _common(p)
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("cli", help="open a terminal session in this folder")
    _common(c, suppress=True)
    w = sub.add_parser("web", help="open the web interface in your default browser")
    w.add_argument("--port", type=int, default=None, help="the port to use on this computer (default 5000)")
    w.add_argument("--page", choices=["studio", "chat"], default="studio", help="which page to open: the Studio (default) or the chat")
    w.add_argument("--no-open", action="store_true", help="start the server without opening a browser")
    w.add_argument("--dir", default=".", help="the folder to set up for the chat (default: here)")
    w.add_argument("--host", default=None, help="Ollama address (default http://localhost:11434)")
    st = sub.add_parser("setup", help="look at this computer's GPU and set up the local models that suit it")
    st.add_argument("--list", action="store_true", help="show every local model by the GPU memory it needs, and how each fits this computer")
    st.add_argument("--yes", action="store_true", help="download without asking")
    st.add_argument("--hardware", help="plan for another computer instead: cpu:16, gpu:12, gpu:24:64, mac:16")
    st.add_argument("--host", default=None, help="Ollama address (default http://localhost:11434)")
    a = sub.add_parser("ask", help="ask one question and print the answer")
    _common(a, suppress=True)
    a.add_argument("question", nargs="+")
    a.add_argument("--plan", action="store_true", help="use the Plan agent: a plan instead of an answer")
    a.add_argument("--build", action="store_true", help="use the Build agent: it may propose file changes (asks before writing)")
    b = sub.add_parser("bench", help="run one task as a plain chat and through Pal, on the same model, and compare the real tokens")
    _common(b, suppress=True)
    b.add_argument("task", nargs="*", help="what to build, as you would type it (leave it out to open the bench screen and type tasks there)")
    b.add_argument("--then", action="append", default=None, metavar="MESSAGE", help="a follow-up message (repeat for more; default: two standard ones)")
    b.add_argument("--no-follow-ups", action="store_true", help="only the task itself")
    b.add_argument("--json", action="store_true", help="print the result as JSON")
    p.epilog = ("more: pal doctor (check this computer), pal models (browse OpenRouter models), pal config, pal tools. "
                "In a session, /help lists the commands.")
    return p


def bench(args, core_factory: Callable | None = None) -> int:
    """pal bench: the same task, plain chat versus Pal, on the same model. Without a task it opens the usual Pal screen (\"pal bench\"), and every
    task typed there is run both ways, with the results as a table and bar charts."""
    import json as _json

    if not args.task:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            print("pal bench: give a task, e.g.  pal bench \"make a todo app\"", file=sys.stderr)
            return 2
        args.bench_screen = True
        return interactive(args, core_factory)
    console = ui.make_console()
    host = args.host or backend.DEFAULT_HOST
    prefs = backend.load_prefs()
    if args.provider and args.provider != backend.LOCAL:
        provider, model = args.provider, args.model or prefs.get("model", "")
    else:
        info = backend.local_info(host)
        provider, model = backend.pick_default(info, {**prefs, **({"provider": backend.LOCAL, "model": args.model} if args.model else {})})
        model = args.model or model
    if not model:
        print("pal bench: choose a model, e.g.  pal bench --provider openrouter --model <name> \"your task\"", file=sys.stderr)
        return 2
    try:
        backend.make_core(provider, model, host)
    except PalError as e:
        print(f"pal bench: {e}", file=sys.stderr)
        return 2
    result = bench_run(console, Path(args.dir), " ".join(args.task), provider, model, host, [] if args.no_follow_ups else args.then)
    if args.json:
        print(_json.dumps(result.to_dict(), indent=1))
        return 0
    show_bench(console, result, provider, model, Path(args.dir))
    return 0


def bench_run(console, folder: Path, task: str, provider: str, model: str, host: str, follow=None, core_factory: Callable | None = None):
    from pal.terminal.bench import run_bench
    status = console.status(Text("Starting\u2026", style=theme.MUTED), spinner=theme.SPINNER, spinner_style=theme.RED)
    with status:                                                 # the progress shows under the task, inside Pal's screen
        return run_bench(folder, task, lambda: (core_factory or backend.make_core)(provider, model, host), model, follow,
                         progress=lambda m: status.update(Text(m, style=theme.MUTED)))


def bars(console, title: str, rows: list[tuple[str, float, str]], width: int = 46) -> None:
    """A horizontal bar chart in the terminal: (label, value, shown value). Plain chat in grey, Pal in red."""
    top = max((v for _, v, _ in rows), default=0) or 1
    ui.say(console, Text(title, style=f"bold {theme.WHITE}"))
    for label, value, shown in rows:
        n = max(1, round(width * value / top)) if value > 0 else 0
        colour = theme.RED if label.lower().startswith("pal") else theme.MUTED
        ui.say(console, Text.assemble(("  " + f"{label:<11}", theme.INK), ("\u2588" * n, colour), (" " * (width - n + 2), ""), (shown, f"bold {theme.WHITE}")))


def show_bench(console, result, provider: str, model: str, folder: Path) -> None:
    """The comparison, split the way it should be read: input tokens (what Pal controls: what is sent) and output tokens (what the model
    chooses to write, which varies from run to run), then the totals and the time."""
    from rich.table import Table

    from pal.terminal.bench import source_files
    ui.say(console, Text.assemble(("pal bench  ", f"bold {theme.WHITE}"), (f"{model}  \u00B7  {ui.short_path(Path(folder).resolve())}", theme.MUTED)))
    ui.say(console, Text(f"Task: {result.task}", style=theme.INK))
    console.print()
    grid = Table(box=None, padding=(0, 2), header_style=f"bold {theme.WHITE}")
    for col in ("step", "agent", "plain in", "Pal in", "plain out", "Pal out"):
        grid.add_column(col, justify="left" if col in ("step", "agent") else "right")
    for i in range(max(len(result.plain.steps), len(result.pal.steps))):
        a = result.plain.steps[i] if i < len(result.plain.steps) else None
        b = result.pal.steps[i] if i < len(result.pal.steps) else None
        first = a or b
        grid.add_row(str(i + 1), first.mode if first else "", f"{a.prompt:,}" if a else "-", f"{b.prompt:,}" if b else "-",
                     f"{a.completion:,}" if a else "-", f"{b.completion:,}" if b else "-")
    pin, pout = sum(s.prompt for s in result.plain.steps), sum(s.completion for s in result.plain.steps)
    qin, qout = sum(s.prompt for s in result.pal.steps), sum(s.completion for s in result.pal.steps)
    grid.add_row("", Text("total", style=f"bold {theme.WHITE}"), f"{pin:,}", f"{qin:,}", f"{pout:,}", f"{qout:,}")
    ui.say(console, grid)
    console.print()
    bars(console, "Input tokens (what is sent: Pal's part)", [("plain chat", pin, f"{pin:,}"), ("Pal", qin, f"{qin:,}")])
    console.print()
    bars(console, "Output tokens (what the model wrote)", [("plain chat", pout, f"{pout:,}"), ("Pal", qout, f"{qout:,}")])
    console.print()
    bars(console, "Time", [("plain chat", result.plain.seconds, f"{result.plain.seconds:g}s"), ("Pal", result.pal.seconds, f"{result.pal.seconds:g}s")])
    console.print()
    saved_in = pin - qin
    pct_in = round(100.0 * saved_in / pin, 1) if pin else 0.0
    verdict = (f"Input: Pal sent {saved_in:,} fewer tokens ({pct_in:g}% less)." if saved_in > 0 else
               f"Input: Pal sent {-saved_in:,} more tokens (its fixed rules, about 100 tokens a request, with nothing to save against here).")
    ui.say(console, Text(verdict, style=f"bold {theme.RED_HI}"))
    if qout != pout:
        ui.say(console, Text(f"Output: the model wrote {abs(qout - pout):,} {'more' if qout > pout else 'fewer'} tokens in Pal's run. Output is the model's own "
                             "choice and changes from run to run; a longer answer is also resent in every later step, in both ways.", style=theme.INK))
    if not source_files(Path(folder)) and not any(p.suffix.lower() in (".pdf", ".docx", ".md", ".txt") for p in Path(folder).glob("*")):
        ui.say(console, Text("This folder is empty, so the plain chat had no code or documents to paste: there was nothing for Pal to save. "
                             "Run the bench in a real project (and with longer conversations) to see what Pal saves.", style=f"bold {theme.WHITE}"))
    for arm in (result.plain, result.pal):
        files = sorted({f for s in arm.steps for f in s.files})
        ui.say(console, Text(f"{arm.name:<6} {arm.seconds:g}s \u00B7 {arm.calls} model call{'s' if arm.calls != 1 else ''} \u00B7 wrote: {', '.join(files) or 'no files'}"
                             + (f" \u00B7 stopped: {arm.error}" if arm.error else ""), style=theme.MUTED))
    est = result.plain.estimated_calls + result.pal.estimated_calls
    if est:
        ui.say(console, Text(f"{est} call(s) had no token count from the model, so those are estimates (characters / 4).", style=theme.MUTED))
    ui.say(console, Text("Pal's local helper (trimming what you type, answering easy questions) is off in the bench, so this compares what is sent "
                         "and nothing else.", style=theme.MUTED))
    console.print()


def keep_bench_files(console, result, folder: Path) -> list[str]:
    """Both ways build in temporary copies; offer to write Pal's version into the folder (asking first, never overwriting without saying so)."""
    if not result.pal_files:
        return []
    folder = Path(folder).resolve()
    clash = [p for p in result.pal_files if (folder / p).exists()]
    question = f"Save Pal's version ({len(result.pal_files)} file{'s' if len(result.pal_files) != 1 else ''}) in {ui.short_path(folder)}?" + (
        f" It replaces {len(clash)} existing file{'s' if len(clash) != 1 else ''}." if clash else "")
    if not ui.confirm(console, question):
        return []
    written = []
    for rel, data in result.pal_files.items():
        target = (folder / rel).resolve()
        if folder not in target.parents:
            continue                                           # never outside the folder
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(rel)
    ui.say(console, Text(f"Saved: {', '.join(written)}", style=theme.INK))
    return written


def setup(args) -> int:
    from pal.terminal.hardware import from_spec
    from pal.terminal.setup import pull, run_setup, show_table
    console = ui.make_console()
    try:
        hw = from_spec(args.hardware) if args.hardware else backend.hardware()
    except ValueError as e:
        print(f"pal setup: {e}", file=sys.stderr)
        return 2
    if args.list:
        show_table(console, hw)
        return 0
    run_setup(console, hw, host=args.host or backend.DEFAULT_HOST, assume_yes=args.yes, puller=pull, dry=bool(args.hardware))
    return 0


def first_run() -> None:
    """The first `pal cli` or `pal web` in a terminal runs setup once."""
    from pal.terminal.setup import needs_setup, run_setup
    if needs_setup() and sys.stdin.isatty() and sys.stdout.isatty():
        console = ui.make_console()
        run_setup(console, backend.hardware())
        console.print()


def prepare_folder(args, console, core_factory: Callable | None = None, choose: bool = True) -> tuple[str, Project | None]:
    """Before a session: get local models ready (install and start Ollama, download the models for this GPU, asking first), then set up the
    folder: documents to markdown, the project map, the code graph (in the background). Returns (a line about the local model, the Project)."""
    from pal.terminal.project import Project
    from pal.terminal.setup import ensure_local
    local = ""
    if core_factory is None:
        tty = sys.stdin.isatty() and sys.stdout.isatty()
        local = ensure_local(console, backend.hardware(), args.host or backend.DEFAULT_HOST, interactive=tty, choose=choose).detail
    folder = Path(getattr(args, "dir", ".") or ".")
    if os.environ.get("PAL_SKIP_PROJECT"):
        return local, None
    try:
        with console.status(Text(f"Setting up {ui.short_path(folder.resolve())}\u2026", style=theme.MUTED), spinner=theme.SPINNER, spinner_style=theme.RED):
            project = Project(folder)
            project.prepare()
    except Exception as e:  # noqa: BLE001  (the session still starts without the folder's map, graph and documents)
        ui.say(console, Text(f"Could not set up this folder: {type(e).__name__}: {str(e)[:200]}", style=theme.RED_HI))
        return local, None
    return local, project


def web(args) -> int:
    """Start the web interface on this computer and open it in the default browser, at once. It listens on 127.0.0.1 only, unless the settings
    say otherwise. The setup (Ollama, the helper, this folder's documents, map and graph) runs in the background; the chat gets the folder's
    documents as soon as they are ready."""
    from pal.cli import _serve, load_dotenv
    from pal.config import load_settings
    from pal.errors import ConfigError
    from pal.terminal.project import Project, ProjectGraph
    load_dotenv()
    overrides: dict = {"server": {"port": args.port}} if args.port else {}
    helper = backend.load_prefs().get("helper_model")
    overrides["local_llm"] = {"backend": "ollama", "model": helper} if helper else {"enabled": False}    # no helper yet: API models only
    try:
        settings = load_settings(overrides=overrides or None)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    folder = Path(getattr(args, "dir", ".") or ".").resolve()
    project = None if os.environ.get("PAL_SKIP_PROJECT") else Project(folder)
    graph = ProjectGraph(project.root / "graph", settings.knowledge_graph) if project is not None else None

    def in_background(service) -> None:
        import io

        from rich.console import Console

        from pal.terminal.setup import ensure_local
        try:
            host = args.host or backend.DEFAULT_HOST
            hw = backend.hardware()
            ensure_local(Console(file=io.StringIO()), hw, host, interactive=False, choose=False)
            background_helper(hw, backend.local_info(host))
            from pal.terminal import voice
            voice.preload()                                            # the page's mic answers at once
            if project is not None:
                project.prepare()
                service.preload_docs = list(project.documents())       # new chats get the folder's documents and map
        except Exception:  # noqa: BLE001  (the web interface works without the folder's setup)
            pass
    return _serve(settings, open_browser=not args.no_open, page="studio" if args.page == "studio" else "", graph=graph,
                  project_dir=folder, background=in_background)


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")      # type: ignore[union-attr]   # old Windows consoles default to a legacy code page
        except (AttributeError, ValueError):
            pass


NO_GPU = ("No GPU on this computer, so Pal uses an API model. Choose one with:\n"
          "  pal cli --provider openrouter --model <name>     (set OPENROUTER_API_KEY, or OPENAI_API_KEY, ANTHROPIC_API_KEY ...)\n"
          "  browse models:  pal models --free")


def start(args, core_factory: Callable | None = None, project=None, prompt: bool = True) -> PalSession | None:
    """The session for these arguments, or None (after saying why) when there is no model to talk to.
    On a computer without a GPU there is no local model: the session uses an API model and Pal's local helper jobs are off."""
    console = ui.make_console()
    host = args.host or backend.DEFAULT_HOST
    prefs = backend.load_prefs()
    no_gpu = core_factory is None and api_only(backend.hardware())
    if no_gpu:
        return _start_api_only(args, prefs, console, project, prompt)
    want = args.provider or prefs.get("provider", "")
    if want in backend.PROVIDER_KEYS and not os.environ.get(backend.PROVIDER_KEYS[want]) and core_factory is None and sys.stdin.isatty():
        if not prompt:
            return None                              # the key is asked for in the foreground
        ask_key(console, want)                        # the main model is an API model: its key (never saved) for this session
    info = backend.local_info(host)
    if args.provider and args.provider != backend.LOCAL:
        provider, model = args.provider, args.model or prefs.get("model", "")
    else:
        provider, model = backend.pick_default(info, {**prefs, **({"provider": backend.LOCAL, "model": args.model} if args.model else {})})
        if args.model:
            model = args.model
    if not args.provider and not model:           # nothing local: maybe a key is set, and the last API model still works
        provider, model = backend.pick_default(info, prefs)
    mode = args.mode or prefs.get("mode", "build")
    try:
        core = (core_factory or backend.make_core)(provider, model, host) if (model or core_factory) else None
    except PalError as e:
        console.print(Text(f"\n{e}\n", style=theme.RED_HI))
        return None
    if core is None:
        console.print(Text("\n" + NO_MODEL + "\n", style=theme.MUTED), soft_wrap=True)
        return None
    helper = backend.make_helper(prefs, info, host) if core_factory is None and info.get("ok") else None
    session = make_session(core, model, provider, Path(args.dir), mode, helper=helper, project=project)
    session.budget = max(0, getattr(args, "budget", 0) or 0)
    session.helper_model = prefs.get("helper_model", "") if helper else ""
    apply_role_models(session, core_factory, host, console)
    return session


def ask_key(console, provider: str) -> bool:
    """Ask for a provider's key for this session only (it is never written anywhere). True when one was given."""
    import getpass
    var = backend.PROVIDER_KEYS[provider]
    console.print(Text(f"\n  Your main model uses {provider}. Paste your {provider} API key (not shown, not saved; or set {var} to skip this):",
                       style=theme.WHITE))
    try:
        key = getpass.getpass("  key: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    if key:
        os.environ[var] = key
    return bool(key)


def _start_api_only(args, prefs: dict, console, project=None, prompt: bool = True) -> PalSession | None:
    from pal.terminal.setup import ask_api
    keyed = lambda p: p in backend.OPEN_PROVIDERS or (p in backend.PROVIDER_KEYS and bool(os.environ.get(backend.PROVIDER_KEYS[p])))  # noqa: E731
    if args.provider == backend.LOCAL:
        console.print(Text("\nThis computer has no GPU, so Pal does not run local models here. Use an API model instead.\n", style=theme.RED_HI))
    provider, model = (args.provider, args.model) if args.provider and args.provider != backend.LOCAL else (prefs.get("provider", ""), args.model or prefs.get("model", ""))
    if provider == backend.LOCAL:
        provider, model = "", ""
    if not (provider and model and keyed(provider)):
        if not prompt:
            return None
        if sys.stdin.isatty() and sys.stdout.isatty():
            console.print(Text("\n  No GPU on this computer, so Pal uses an API model.\n", style=f"bold {theme.RED_HI}"))
            chosen = ask_api(console, backend.hardware().system)
            if not chosen:
                console.print(Text("\n" + NO_GPU + "\n", style=theme.MUTED), soft_wrap=True)
                return None
            provider, model = chosen
        else:
            console.print(Text("\n" + NO_GPU + "\n", style=theme.MUTED), soft_wrap=True)
            return None
    try:
        core = backend.make_core(provider, model)
    except PalError as e:
        console.print(Text(f"\n{e}\n", style=theme.RED_HI))
        return None
    session = make_session(core, model, provider, Path(args.dir), args.mode or prefs.get("mode", "build"), local=False, project=project)
    session.budget = max(0, getattr(args, "budget", 0) or 0)
    apply_role_models(session, None, args.host or backend.DEFAULT_HOST, console)
    return session


API_CHOICE = "An API model…"


def pick_model(console, session: PalSession, info: dict) -> str | None:
    """/model: the models that suit this GPU (installed ones marked), or an API model. A local model that is not installed yet is downloaded
    (after asking). Returns the model to switch to (a local name, or provider:model), or None."""
    from pal.terminal import decide
    from pal.terminal.local_models import suitable
    from pal.terminal.setup import ollama_installed, pull
    hw = backend.hardware()
    installed = backend.chat_models(info) if info.get("ok") else []
    options = [] if not session.local else suitable(hw, [m for m in installed if m != session.helper_model])
    choices = [decide.Option(tag, about) for tag, about, _ in options]
    for name in installed:                                   # anything else already installed (not the helper)
        if name != session.helper_model and all(o.label != name for o in choices):
            choices.append(decide.Option(name, "installed"))
    choices = choices[:7] + [decide.Option(API_CHOICE, "OpenAI, Anthropic, Gemini, OpenRouter, any LiteLLM model or your own endpoint")]
    current = session.model if session.provider == backend.LOCAL else ""
    q = decide.Question(f"Main model (now: {session.model}, {session.provider}). The best for your GPU is first; type any Ollama name to get it.",
                        choices, "Model", 0, other=True)
    answer = decide.ask(console, q)
    decide.show_answer(console, q, answer)
    if not answer or answer == current:
        return None
    if answer == API_CHOICE:
        try:
            spec = console.input(Text("  provider:model (for example openai:gpt-5, anthropic:<model>, openrouter:<model>, litellm:<model>): ", style=theme.WHITE)).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        provider, _ = backend.split_spec(spec)
        if provider in backend.PROVIDER_KEYS and not os.environ.get(backend.PROVIDER_KEYS[provider]) and not ask_key(console, provider):
            return None
        return spec or None
    if not session.local:
        ui.say(console, Text("This computer has no GPU, so local models are off.", style=theme.RED_HI))
        return None
    if answer not in installed:
        if not ollama_installed():
            ui.say(console, Text("Ollama is not installed: run  pal setup  first.", style=theme.RED_HI))
            return None
        if not ui.confirm(console, f"{answer} is not installed. Download it now?"):
            return None
        ui.say(console, Text(f"Downloading {answer} ...", style=theme.WHITE))
        if not pull(answer):
            ui.say(console, Text(f"The download of {answer} did not finish.", style=theme.RED_HI))
            return None
    return answer


def set_role_model(session: PalSession, role: str, spec: str, core_factory=None, host: str = backend.DEFAULT_HOST) -> str:
    """Give an agent its own model ('default' gives it back the session's). Returns '' or what went wrong."""
    if spec in ("default", "off", "none", ""):
        session.role_models.pop(role, None)
        setter = getattr(session.client, "set_role_core", None)
        if setter:
            setter(role, None)
        return ""
    provider, name = backend.split_spec(spec)
    if provider == backend.LOCAL and not session.local:
        return "this computer has no GPU, so local models are off: use  /model <agent> <provider>:<model>"
    try:
        core = (core_factory or backend.make_core)(provider, name, host)
    except PalError as e:
        return str(e)
    setter = getattr(session.client, "set_role_core", None)
    if setter is None:
        return "this session cannot give agents their own models"
    setter(role, core)
    session.role_models[role] = (provider, name)
    return ""


def apply_role_models(session: PalSession, core_factory=None, host: str = backend.DEFAULT_HOST, console=None) -> None:
    """The per-agent models saved with /model <agent> <model>, for a new session."""
    prefs = backend.load_prefs()
    for role in MODES:
        spec = prefs.get(f"model_{role}", "")
        if spec:
            problem = set_role_model(session, role, spec, core_factory, host)
            if problem and console is not None:
                ui.say(console, Text(f"{role.capitalize()} agent keeps the session's model ({spec}: {problem})", style=theme.MUTED))


def _switch(args, session: PalSession, spec: str, core_factory, console) -> PalSession:
    """/model name  (local model)   or   /model provider:name"""
    provider, name = backend.split_spec(spec)
    host = args.host or backend.DEFAULT_HOST
    if provider == backend.LOCAL and not session.local:
        ui.say(console, Text("This computer has no GPU, so local models are off. Use  /model <provider>:<name>, for example /model openrouter:<name>", style=theme.RED_HI))
        return session
    if provider == backend.LOCAL:
        installed = backend.chat_models(backend.local_info(host))
        if installed and name not in installed:
            ui.say(console, Text(f"{name} is not installed. Installed: {', '.join(installed)}", style=theme.RED_HI))
            return session
    try:
        core = (core_factory or backend.make_core)(provider, name, host)
    except PalError as e:
        ui.say(console, Text(str(e), style=theme.RED_HI))
        return session
    mode, cwd, budget, helper, helper_name = session.mode, session.cwd, session.budget, session.helper_core, session.helper_model
    session.close()
    backend.save_prefs(provider=provider, model=name)
    ui.say(console, Text(f"Now using {name}. This starts a new conversation.", style=theme.MUTED))
    console.print()
    fresh = make_session(core, name, provider, cwd, mode, helper=helper, local=session.local, project=session.project)
    fresh.budget, fresh.helper_model, fresh.local_status = budget, helper_name, session.local_status
    apply_role_models(fresh, core_factory, host)
    return fresh


def split_paths(text: str) -> list[str]:
    """File paths as a terminal types them when files are dropped on it: quoted, or with escaped spaces."""
    import shlex
    try:
        parts = shlex.split(text, posix=os.name != "nt")
    except ValueError:
        return []
    return [x.strip("\"'") for x in parts if x.strip("\"'")]


def dropped_files(text: str) -> list[str]:
    """The paths, when everything typed is one or more existing files (what dropping files on the window produces); else []."""
    paths = split_paths(text)
    return paths if paths and all(Path(x).expanduser().is_file() for x in paths) else []


def attach(session: PalSession, paths: list[str], console) -> None:
    for raw in paths:
        name, note = session.attach(raw)
        if name:
            ui.say(console, Text.assemble(("  \u2713 attached ", theme.RED_HI), (name, f"bold {theme.WHITE}"), (f"   {note}" if note else "", theme.MUTED)))
        else:
            ui.say(console, Text(f"  \u2717 {note}", style=theme.RED_HI))
    if session.pending:
        ui.say(console, Text(f"  {len(session.pending)} file{'s' if len(session.pending) != 1 else ''} will go with your next message.", style=theme.MUTED))


def _fresh(session: PalSession, args, core_factory) -> PalSession:
    """A new session with the same model, agent, cap and helper: a clean conversation for the engine."""
    mode, cwd, provider, model, budget = session.mode, session.cwd, session.provider, session.model, session.budget
    helper, helper_name, local = session.helper_core, session.helper_model, session.local
    session.close()
    core = (core_factory or backend.make_core)(provider, model, args.host or backend.DEFAULT_HOST)
    fresh = make_session(core, model, provider, cwd, mode, helper=helper, local=local, project=session.project)
    fresh.budget, fresh.helper_model, fresh.local_status = budget, helper_name, session.local_status
    apply_role_models(fresh, core_factory, args.host or backend.DEFAULT_HOST)
    return fresh


def _bind_status(wf: Workflow) -> None:
    """What the status line shows about the plan: only the changes waiting to be applied (no plan names or ids)."""
    def note() -> str:
        n = len(wf.plan.b.pending)
        return f"{n} file{'s' if n != 1 else ''} pending" if n else ""
    wf.session.status_note = note


def handle(line: str, args, session: PalSession, console, core_factory, box: dict | None = None) -> tuple[PalSession, bool]:
    """Run a /command. Returns (the session to use next, whether to keep going)."""
    cmd, _, rest = line.strip().partition(" ")
    rest = rest.strip()
    if cmd in ("/exit", "/quit"):
        return session, False
    from pal.terminal.commands import handle_extra
    extra = handle_extra(cmd, rest, args, session, console, core_factory, box)
    if extra is not None:
        return extra
    if box is not None and cmd not in ("/new", "/cd") and flow_cli.handle_flow(cmd, rest, box["wf"], console):
        return session, True
    if cmd == "/help":
        ui.show_help(console)
    elif cmd == "/agent":
        from pal.terminal.setup import choose_agent
        mode = rest if rest in MODES else choose_agent(console, current=session.mode)
        if mode:
            session.mode = mode
            backend.save_prefs(mode=mode)
            ui.say(console, Text(f"Agent: {mode.capitalize()} ({AGENT_ABOUT[mode]})", style=theme.MUTED))
        console.print()
    elif cmd == "/mode":
        if rest in MODES:
            session.mode = rest
        else:
            session.toggle_mode()
        backend.save_prefs(mode=session.mode)
        ui.say(console, Text(f"Agent: {session.mode.capitalize()} ({AGENT_ABOUT[session.mode]})", style=theme.MUTED))
        console.print()
    elif cmd == "/model" and rest.split(" ", 1)[0] in MODES:
        role, _, spec = rest.partition(" ")
        if not spec.strip():
            prov, name = session.model_for(role)
            ui.say(console, Text(f"{role.capitalize()} agent: {name} ({prov})" + ("" if role in session.role_models else "  (the session's model)")
                                 + f"   change: /model {role} <provider>:<model>   back: /model {role} default", style=theme.MUTED))
        else:
            problem = set_role_model(session, role, spec.strip(), core_factory, args.host or backend.DEFAULT_HOST)
            if problem:
                ui.say(console, Text(problem, style=theme.RED_HI))
            else:
                backend.save_prefs(**{f"model_{role}": "" if spec.strip() in ("default", "off", "none") else spec.strip()})
                prov, name = session.model_for(role)
                ui.say(console, Text(f"{role.capitalize()} agent now uses {name} ({prov}). Same conversation; saved for next time.", style=theme.MUTED))
        console.print()
    elif cmd == "/endpoint":
        if rest.startswith(("http://", "https://")):
            backend.save_prefs(custom_base_url=rest.strip())
            ui.say(console, Text(f"Custom endpoint: {rest.strip()}  (saved; the key comes from PAL_CUSTOM_API_KEY or  /key custom <key>). Then:  /model custom:<model>", style=theme.MUTED))
        else:
            ui.say(console, Text(f"Custom endpoint: {backend.custom_base_url() or 'none'}.  Set one with  /endpoint https://your-server/v1  (any OpenAI-compatible API)", style=theme.MUTED))
        console.print()
    elif cmd == "/model":
        if rest:
            switched = _switch(args, session, rest, core_factory, console)
            if box is not None and switched is not session:
                box["wf"] = Workflow(switched, box["wf"].store, plan=box["wf"].plan.name)
                _bind_status(box["wf"])
            return switched, True
        info = backend.local_info(args.host or backend.DEFAULT_HOST)
        if sys.stdin.isatty() and sys.stdout.isatty() and core_factory is None:
            chosen = pick_model(console, session, info)
            if chosen:
                switched = _switch(args, session, chosen, core_factory, console)
                if box is not None and switched is not session:
                    box["wf"] = Workflow(switched, box["wf"].store, plan=box["wf"].plan.name)
                    _bind_status(box["wf"])
                return switched, True
        else:
            ui.show_models(console, session, info, backend.hardware(), [p for p, ok in providers_with_keys().items() if ok])
    elif cmd == "/add":
        paths = split_paths(rest)
        if not paths:
            ui.say(console, Text("Use: /add <file> [more files]. Any file on this computer: code, text, PDF, Word, PowerPoint, Excel. Or drop files on this window.", style=theme.MUTED))
        attach(session, paths, console)
        console.print()
    elif cmd == "/key":
        provider, _, key = rest.partition(" ")
        if provider == "custom" and key.strip():
            os.environ["PAL_CUSTOM_API_KEY"] = key.strip()
            ui.say(console, Text("Key for the custom endpoint set for this session (not saved).", style=theme.MUTED))
            console.print()
            return session, True
        if provider not in backend.PROVIDER_KEYS or not key.strip():
            ui.say(console, Text(f"Use: /key <provider> <key>   providers: {', '.join(backend.PROVIDER_KEYS)}. The key is kept for this session only, never saved.", style=theme.MUTED))
        else:
            os.environ[backend.PROVIDER_KEYS[provider]] = key.strip()
            ui.say(console, Text(f"Key for {provider} set for this session (not saved). Now pick a model:  /model {provider}:<name>", style=theme.MUTED))
        console.print()
    elif cmd == "/files":
        ui.say(console, Text("\n".join(session.listing()) or "(this folder has no files Pal can see)", style=theme.MUTED))
        console.print()
    elif cmd == "/security":
        findings, checked = session.audit()
        counts = {sev: sum(1 for f in findings if f.severity == sev) for sev in ("critical", "high", "medium", "low")}
        ui.say(console, Text.assemble(("Security check  ", f"bold {theme.WHITE}"),
                                      (f"{checked} files read \u00B7 " + " \u00B7 ".join(f"{v} {k}" for k, v in counts.items()), theme.MUTED)))
        if findings:
            ui.show_findings(console, findings)
        else:
            ui.say(console, Text("Nothing found. These are pattern checks: they catch common mistakes, they cannot prove the code is safe.", style=theme.MUTED))
        if rest == "ai":
            console.print()
            turn_ai(session, findings, console)
        else:
            ui.say(console, Text("/security ai  asks the Plan agent for a deeper review (uses tokens; secrets are hidden first)", style=theme.MUTED))
        console.print()
    elif cmd == "/budget":
        if rest in ("off", "0", "none"):
            session.budget = 0
            ui.say(console, Text("No token cap for this session.", style=theme.MUTED))
        elif rest.replace(",", "").replace("_", "").isdigit():
            session.budget = int(rest.replace(",", "").replace("_", ""))
            ui.say(console, Text(f"Token cap: {session.budget:,} for this session ({session.pal_tokens:,} used so far). Pal stops before going over.", style=theme.MUTED))
        else:
            ui.say(console, Text(f"Token cap: {f'{session.budget:,}' if session.budget else 'none'}. Set one with /budget 50000, or /budget off.", style=theme.MUTED))
        console.print()
    elif cmd in ("/tokens", "/stats"):
        ui.show_tokens(console, session, rest)
    elif cmd == "/project":
        if rest == "rebuild" and session.project is not None:
            with console.status(Text("Setting up this folder again\u2026", style=theme.MUTED), spinner=theme.SPINNER, spinner_style=theme.RED):
                session.project.prepare()
            session.client_add_documents()
        ui.show_project(console, session)
        if rest != "rebuild":
            ui.say(console, Text("/project rebuild  reads the folder again (new or changed documents and code)", style=theme.MUTED))
        console.print()
    elif cmd in ("/new", "/cd"):
        if box is None:
            return session, True
        store = box["wf"].store
        try:
            if cmd == "/cd" and not rest:
                raise ValueError("Use: /cd <plan name or number>   (/plans lists them)")
            fresh = _fresh(session, args, core_factory)
            wf = Workflow(fresh, store, plan=box["wf"].plan.name)
            name = wf.new(rest) if cmd == "/new" else wf.switch(rest)
        except ValueError as e:
            ui.say(console, Text(str(e), style=theme.RED_HI))
            console.print()
            return session, True
        session, box["wf"] = fresh, wf
        _bind_status(wf)
        ui.say(console, Text(f"{'New plan' if cmd == '/new' else 'Now on plan'} {name}." + (" Its conversation is picked up where it stopped." if cmd == "/cd" and wf.plan.b.convo else ""), style=theme.MUTED))
        console.print()
    else:
        ui.say(console, Text(f"Unknown command {cmd}. Type /help.", style=theme.RED_HI))
        console.print()
    return session, True


def _live(session: PalSession, seconds: float) -> Text:
    """The line next to the spinner: what Pal is doing, for how long, and what it has saved so far."""
    doing = {"build": "building", "plan": "planning", "ask": "thinking"}.get(session.mode, "working")
    out = Text(f"Pal is {doing}\u2026 {seconds:.0f}s", style=theme.MUTED)
    if session.saved > 0:
        out.append("  \u00B7  saved so far ", style=theme.MUTED)
        out.append(f"{session.saved:,} tokens ({session.saved_pct:g}%)", style=f"bold {theme.RED_HI}")
    elif session.turns:
        out.append(f"  \u00B7  {session.pal_tokens:,} tokens used so far", style=theme.MUTED)
    return out


def turn_ai(session: PalSession, findings, console) -> None:
    """/security ai: the Plan agent reviews the project, with the scanner's findings and the files they point at."""
    try:
        with console.status(Text("Pal is reviewing the project\u2026", style=theme.MUTED), spinner=theme.SPINNER, spinner_style=theme.RED):
            result = session.audit_ai(findings)
    except BudgetExceeded as e:
        ui.say(console, Text(f"Stopped: {e}", style=theme.RED_HI))
        return
    except KeyboardInterrupt:
        ui.say(console, Text("Stopped.", style=theme.MUTED))
        return
    except Exception as e:  # noqa: BLE001
        ui.say(console, Text(f"Something went wrong: {type(e).__name__}: {str(e)[:300]}", style=theme.RED_HI))
        return
    ui.show_reply(console, result, session)


AUTONOMY_CYCLE = ("none", "basic", "plus", "semi")      # Shift+Tab; "full" (runs commands without asking) only with /autonomy full


def cycle_autonomy(wf: Workflow) -> str:
    """Shift+Tab: the next autonomy level. Returns the line the status bar shows."""
    now = wf.plan.autonomy
    nxt = AUTONOMY_CYCLE[(AUTONOMY_CYCLE.index(now) + 1) % len(AUTONOMY_CYCLE)] if now in AUTONOMY_CYCLE else "none"
    wf.plan.autonomy, wf.plan.custom = nxt, {}
    wf.save()
    about = {"none": "asks before everything", "basic": "builds, you apply", "plus": "loads files it needs, you apply",
             "semi": "writes changes into the folder after its checks, asks before running commands"}[nxt]
    return f"autonomy: {nxt}  ({about})   shift+tab next"


def voice_flow(console) -> str:
    """Record until Enter, then turn it into text. Returns the words, or '' (after saying why)."""
    import threading

    from pal.terminal import voice
    if not voice.can_record():
        ui.say(console, Text("The mic is not available here: no microphone was found, or (on Linux) PortAudio is missing: sudo apt install libportaudio2. "
                             "Windows voice typing (Win+H) and apps such as Wispr Flow type into the box too.", style=theme.MUTED))
        console.print()
        return ""
    if not voice.engines():
        ui.say(console, Text("Speech to text is not available: reinstall Pal (pip install -U pal-code), or set OPENAI_API_KEY or GROQ_API_KEY.", style=theme.MUTED))
        console.print()
        return ""
    stop, out, err = threading.Event(), {}, {}

    def rec() -> None:
        try:
            out["wav"] = voice.record(stop)
        except voice.VoiceUnavailable as e:
            err["e"] = str(e)
            stop.set()
    th = threading.Thread(target=rec, daemon=True)
    th.start()
    try:
        console.input(Text("   ● listening… press Enter when you are done ", style=f"bold {theme.RED_HI}"))
    except (EOFError, KeyboardInterrupt):
        stop.set()
        th.join(2)
        console.print()
        return ""
    stop.set()
    th.join(5)
    if err or not out.get("wav"):
        ui.say(console, Text(err.get("e", "Nothing was recorded."), style=theme.RED_HI))
        return ""
    try:
        first_time = "local" in voice.engines() and not voice._LOCAL
        label = "Turning speech into text (the first time, the speech model is downloaded once, about 150 MB)…" if first_time else "Turning speech into text…"
        with console.status(Text(label, style=theme.MUTED), spinner=theme.SPINNER, spinner_style=theme.RED):
            heard = voice.transcribe(out["wav"])
    except voice.VoiceUnavailable as e:
        ui.say(console, Text(str(e), style=theme.RED_HI))
        return ""
    ui.say(console, Text(f"   heard ({heard.engine}): {heard.text or '(nothing)'}", style=theme.MUTED))
    return heard.text


def automatic(args, core_factory: Callable | None = None) -> bool:
    """A real terminal session on a computer with a GPU: Pal gets its helper ready by itself and asks nothing at start."""
    return (core_factory is None and sys.stdin.isatty() and sys.stdout.isatty() and not os.environ.get("PAL_SKIP_SETUP")
            and not api_only(backend.hardware()))


def ensure_ready_line(args) -> str:
    """The local-model line again, after a model was downloaded."""
    from pal.terminal.setup import ensure_local
    return ensure_local(ui.make_console(), backend.hardware(), args.host or backend.DEFAULT_HOST, interactive=False, choose=False).detail


def has_main_model(args, host: str) -> bool:
    """Is there a model to start with: one on the command line, or the last one used (local and installed, or an API model with its key)?"""
    if args.provider or args.model:
        return True
    _, model = backend.pick_default(backend.local_info(host), backend.load_prefs())
    return bool(model)


class Boot:
    """What the first screen needs before the real session exists: the agent and the model from the saved settings, nothing probed.
    It lets pal cli draw its screen at once while the setup runs in the background."""

    def __init__(self, args, prefs: dict):
        self.mode = args.mode or prefs.get("mode", "build")
        if self.mode not in MODES:
            self.mode = "build"
        self.provider = args.provider or prefs.get("provider", "") or backend.LOCAL
        self.model = args.model or prefs.get("model", "") or "choose a model"
        self.cwd = Path(getattr(args, "dir", ".") or ".").resolve()
        self.brand = "bench" if getattr(args, "bench_screen", False) else "cli"
        self.pending: list = []
        self.role_models: dict = {}
        self.turns, self.pal_tokens, self.saved, self.budget = 0, 0, 0, 0
        self.notice, self.helper_model, self.local, self.project, self.local_status = "", "", True, None, ""

    def model_for(self, mode: str | None = None) -> tuple[str, str]:
        return self.provider, self.model

    def toggle_mode(self) -> str:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode


def quiet_pull(tag: str) -> bool:
    """`ollama pull` with no output: for the helper, downloaded in the background."""
    import shutil
    import subprocess
    exe = shutil.which("ollama")
    if not exe:
        return False
    try:
        return subprocess.call([exe, "pull", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def background_helper(hw, info: dict) -> None:
    """Pal's helper: the best small model for this computer. Installed: used. Missing: downloaded in a background thread (used from the next
    start), so nothing waits for it."""
    import threading

    from pal.terminal.local_models import pick_helper
    if api_only(hw) or not info.get("ok"):
        return
    best = pick_helper(hw).tag
    if best in backend.chat_models(info):
        backend.save_prefs(helper_model=best)
        return

    def get() -> None:
        if quiet_pull(best):
            backend.save_prefs(helper_model=best)
    threading.Thread(target=get, daemon=True, name="pal-helper-download").start()


def prepare_quietly(args, core_factory: Callable | None, out: dict) -> None:
    """Everything a session needs, without a word on the screen: the hardware, Ollama (started if needed), the folder (documents to markdown,
    the map, the graph), the helper, and the session itself. What needs the person (a model to choose, a key) is left to the foreground."""
    import io

    from rich.console import Console

    from pal.terminal.project import Project
    from pal.terminal.setup import ensure_local
    quiet = Console(file=io.StringIO())
    try:
        host = args.host or backend.DEFAULT_HOST
        if core_factory is None:
            hw = backend.hardware()
            out["local"] = ensure_local(quiet, hw, host, interactive=False, choose=False).detail
            info = backend.local_info(host)
            if automatic(args, core_factory):
                background_helper(hw, info)
        project = None
        if not os.environ.get("PAL_SKIP_PROJECT"):
            try:
                project = Project(Path(getattr(args, "dir", ".") or "."))
                project.prepare()
            except Exception:  # noqa: BLE001  (the session still starts without the folder's map, graph and documents)
                project = None
        out["project"] = project
        out["need_model"] = automatic(args, core_factory) and not has_main_model(args, host)
        if not out["need_model"]:
            out["session"] = start(args, core_factory, project, prompt=False)
    except Exception as e:  # noqa: BLE001  (the foreground tries again and shows what went wrong)
        out["error"] = e


def finish_session(args, core_factory: Callable | None, ready: dict, console, boot: Boot) -> PalSession | None:
    """The real session, once the background setup is done. Asks the person only what it must (a first model, an API key)."""
    session = ready.get("session")
    if session is None:
        host = args.host or backend.DEFAULT_HOST
        if ready.get("need_model"):
            from pal.terminal.setup import choose_models
            if not choose_models(console, backend.hardware(), host, helper=False):
                console.print(Text("\nNo model chosen. Run  pal cli  again to choose one.\n", style=theme.MUTED))
                return None
        session = start(args, core_factory, ready.get("project"))
        if session is None:
            return None
    session.mode = boot.mode
    session.local_status = ready.get("local", "")
    session.brand = boot.brand
    return session


API_ITEM = "\x00api"


def model_choices(args, current: tuple[str, str]) -> list[tuple[str, str, str]]:
    """The model dropdown: (label, about, spec), about being only "installed" or the download size. Local models that suit this GPU,
    installed ones first (the helper left out), the best one for building marked "recommended"; the API model in use; then any API model."""
    from pal.terminal.local_models import BY_TAG, lookup, recommend
    from pal.terminal.setup import main_choices
    out: list[tuple[str, str, str]] = []
    host = args.host or backend.DEFAULT_HOST
    hw = backend.hardware()
    info = backend.local_info(host)
    if not api_only(hw) and info.get("ok"):
        helper = backend.load_prefs().get("helper_model", "")
        installed = [n for n in backend.chat_models(info) if n != helper]
        rec = recommend(hw)
        best = rec.main.tag if rec else ""
        for tag, about in main_choices(hw, installed):
            known = lookup(tag)
            short = "installed" if about.startswith("installed") else (f"{BY_TAG[tag].size_gb:g} GB" if tag in BY_TAG else "download")
            if known is not None and known.tag == best:
                short += "  recommended"
            out.append((tag, short, f"local:{tag}"))
    provider, model = current
    if provider != backend.LOCAL and model and model != "choose a model":
        out.insert(0, (model, provider.capitalize(), f"{provider}:{model}"))
    out.append(("API model\u2026", "OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral", API_ITEM))
    return out


def use_model(args, session: PalSession, spec: str, core_factory, console, box) -> PalSession:
    """A model chosen in the dropdown: downloaded first when it is a local model that is not installed yet, then switched to."""
    from pal.terminal.setup import pull
    provider, name = backend.split_spec(spec)
    host = args.host or backend.DEFAULT_HOST
    if provider == backend.LOCAL and core_factory is None and name not in backend.chat_models(backend.local_info(host)):
        ui.say(console, Text(f"Downloading {name} (once)\u2026", style=theme.WHITE))
        if not pull(name):
            ui.say(console, Text(f"The download of {name} did not finish.", style=theme.RED_HI))
            return session
    if provider in backend.PROVIDER_KEYS and not os.environ.get(backend.PROVIDER_KEYS[provider]) and core_factory is None and not ask_key(console, provider):
        return session
    switched = _switch(args, session, spec, core_factory, console)
    if box is not None and switched is not session:
        box["wf"] = Workflow(switched, box["wf"].store, plan=box["wf"].plan.name)
        _bind_status(box["wf"])
    return switched


def interactive(args, core_factory: Callable | None = None) -> int:
    """pal cli: the REPL (see pal.terminal.repl)."""
    from pal.terminal.repl.main import run
    return run(args, core_factory)


def interactive_box(args, core_factory: Callable | None = None) -> int:
    """The earlier input-box screen: pal cli: the screen is drawn at once; the setup (Ollama, the folder, the helper, the model) runs in the background while you type.
    The model is changed by clicking it (or /model). Only when there is no model at all yet is the person asked for one."""
    import threading

    from pal.terminal.history import History
    console = ui.make_console()
    boot = Boot(args, backend.load_prefs())
    ready: dict = {}
    prep = threading.Thread(target=prepare_quietly, args=(args, core_factory, ready), daemon=True, name="pal-setup")
    prep.start()
    history = History(boot.cwd)
    state: dict = {"session": None, "box": None}
    found: dict = {"items": None}

    def look_for_models() -> None:
        try:
            shown = state["session"] or boot
            found["items"] = model_choices(args, shown.model_for())
        except Exception:  # noqa: BLE001  (Ollama gone: at least the API choice)
            found["items"] = [("An API model\u2026", "type provider:model", API_ITEM)]
    threading.Thread(target=look_for_models, daemon=True, name="pal-models").start()
    if core_factory is None:
        from pal.terminal import voice
        threading.Thread(target=voice.preload, daemon=True, name="pal-voice").start()   # the speech model is ready before the mic is used

    def session_now() -> PalSession | None:
        """The real session: waits for the background setup the first time (only if it has not finished yet)."""
        if state["session"] is None:
            if prep.is_alive():
                with console.status(Text("Getting ready\u2026", style=theme.MUTED), spinner=theme.SPINNER, spinner_style=theme.RED):
                    prep.join()
            s = finish_session(args, core_factory, ready, console, boot)
            if s is None:
                return None
            state["session"], state["box"] = s, {"wf": Workflow(s)}
            _bind_status(state["box"]["wf"])
        return state["session"]

    first, draft, code, listen_now = True, "", 0, False
    open_models: bool | str = False
    try:
        while True:
            shown = state["session"] or boot
            box = state["box"]
            text = ui.read_input(shown, console, backend.save_prefs, landing=first, version=__version__, history=history, initial=draft,
                                 on_shift_tab=(lambda: cycle_autonomy(state["box"]["wf"])) if box else None,
                                 models=lambda: found["items"], open_models=open_models, start_listening=listen_now)
            landing, first, shown.notice, draft, open_models, listen_now = first, False, "", "", False, False
            if text is None:
                break
            if text.startswith("\x00pick\x00"):            # a model chosen in the dropdown
                spec, _, draft = text[len("\x00pick\x00"):].partition("\x00")
                first = landing
                if spec == API_ITEM:
                    draft = "/model "                           # type provider:model, e.g. /model openai:gpt-4o
                    continue
                session = session_now()
                if session is None:
                    code = 1
                    break
                state["session"] = use_model(args, session, spec, core_factory, console, state["box"])
                threading.Thread(target=look_for_models, daemon=True).start()
                continue
            if text.strip() in ("/model", "/agent"):          # on their own: the dropdown in the box
                open_models, first = text.strip()[1:], landing
                continue
            if text.startswith("\x00model\x00"):           # the model clicked before its list was ready: open the dropdown
                draft, open_models, first = text[len("\x00model\x00"):], True, landing
                continue
            if text.startswith("\x00voice\x00"):           # the mic (click or Ctrl+Space): record, then back to the box with the words added
                before = text[len("\x00voice\x00"):]
                heard = voice_flow(console)
                draft = (before + (" " if before and heard else "") + heard) if heard else before
                first = landing
                continue
            if landing and not text.startswith("\x00"):     # leaving the first screen: Pal's own header, the work goes on below it
                ui.show_header(console, boot.cwd, __version__, boot.brand)
            session = session_now()
            if session is None:
                code = 1
                break
            box = state["box"]
            history.add(text)
            text = text.strip()
            if text == "/voice":                          # the mic, in the box
                listen_now = True
                continue
            dropped = dropped_files(text)
            if dropped:                                   # files dropped on the window: attach them to the next message
                attach(session, dropped, console)
                console.print()
                continue
            try:
                if text.startswith("/"):
                    state["session"], go = handle(text, args, session, console, core_factory, box)
                    if not go:
                        break
                elif text and getattr(args, "bench_screen", False):     # pal bench: the task both ways, then the comparison
                    host = args.host or backend.DEFAULT_HOST
                    ui.show_user(console, text)
                    try:
                        result = bench_run(console, session.cwd, text, session.provider, session.model, host, core_factory=core_factory)
                        show_bench(console, result, session.provider, session.model, session.cwd)
                        keep_bench_files(console, result, session.cwd)
                    except PalError as e:
                        ui.say(console, Text(f"The bench stopped: {e}", style=theme.RED_HI))
                elif text:
                    flow_cli.run_turn(box["wf"], text, console)
            except KeyboardInterrupt:                     # Esc or Ctrl+C at an awkward moment: back to the box, never out of Pal
                ui.say(console, Text("Stopped.", style=theme.MUTED))
                console.print()
    finally:
        ui.unpin()
        if state["box"] is not None:
            state["box"]["wf"].close()
        if state["session"] is not None:
            state["session"].close()
    console.print(Text("Bye.", style=theme.MUTED))
    return code


def ask_once(args, core_factory: Callable | None = None) -> int:
    console = ui.make_console()
    args.mode = "build" if args.build else "plan" if args.plan else "ask"
    session = start(args, core_factory)
    if session is None:
        return 1
    question = " ".join(args.question)
    if not sys.stdin.isatty():
        try:
            piped = sys.stdin.read()
        except (OSError, ValueError):
            piped = ""
        if piped.strip():
            question = f"{question}\n\n```\n{piped[:150_000]}\n```"
    try:
        result = session.ask(question)
    except Exception as e:  # noqa: BLE001
        print(f"pal: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
        return 1
    try:
        print(result.text)
        used = result.prompt_tokens + result.completion_tokens
        print(f"\n[{used:,} tokens, {result.seconds:.1f}s]", file=sys.stderr)
        if result.changes and sys.stdin.isatty() and sys.stdout.isatty():
            ui.show_changes(console, result.changes)
            if ui.confirm(console, f"Write {len(result.changes)} file(s) to {ui.short_path(session.cwd)}?"):
                for path, status in session.apply(result.changes):
                    print(f"{status}: {path}")
    finally:
        session.close()
    return 0


TOOLS = ("doctor", "config", "tools", "models", "agent", "chat", "serve")     # the older commands (pal.cli): checks, settings, agents


def main(argv: Sequence[str] | None = None, core_factory: Callable | None = None) -> int:
    _utf8()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in TOOLS:
        from pal.cli import main as tools_main
        return tools_main(argv)
    args = build_parser().parse_args(argv)
    if args.cmd == "web":
        return web(args)
    if args.cmd == "setup":
        return setup(args)
    if args.cmd == "bench":
        return bench(args, core_factory)
    if args.cmd == "ask":
        return ask_once(args, core_factory)
    if not (sys.stdin.isatty() and sys.stdout.isatty()) and core_factory is None:
        print('pal cli needs a terminal. For a single answer use:  pal ask "your question"   For the browser:  pal web', file=sys.stderr)
        return 2
    return interactive(args, core_factory)


if __name__ == "__main__":
    sys.exit(main())
