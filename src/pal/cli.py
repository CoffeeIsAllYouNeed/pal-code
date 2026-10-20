"""Command line: `pal serve`, `pal doctor`, `pal config` (also `python -m pal ...`).

Works the same on Linux, macOS and Windows: pure Python, no shell scripts.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import yaml

from pal import __version__
from pal.agents import AGENTS
from pal.config import Settings, load_settings
from pal.errors import ConfigError, PalError
from pal.logging_setup import configure_logging
from pal.paths import pal_home
from pal.sdk.agent import agent_key

_LOOPBACK = ("127.0.0.1", "localhost", "::1")


@dataclass
class Check:
    name: str
    status: str          # "ok" | "warn" | "fail" | "info"
    detail: str = ""


def _services_check(settings: Settings, env) -> Check:
    """Which third-party tools are switched on, and which have their key."""
    from pal.chat.services import selected
    try:
        specs = selected(settings.tools.services)
    except ConfigError as e:
        return Check("service tools", "fail", str(e))
    if not specs:
        return Check("service tools", "info", "off (name them in tools.services, or all: google_search, wolfram_alpha, currency_convert, "
                                              "words_api, moon_phase, and the image and audio tools)")
    missing: dict[str, list[str]] = {}
    for s in specs:
        if s.key_env and not env.get(s.key_env):
            missing.setdefault(s.key_env, []).append(s.name)
    ready = [s.name for s in specs if not s.key_env or env.get(s.key_env)]
    detail = f"{len(ready)} of {len(specs)} ready" + (f" ({', '.join(ready)})" if ready else "")
    if missing:
        detail += "; missing: " + "; ".join(f"{k} for {', '.join(v)}" for k, v in missing.items())
    return Check("service tools", "warn" if missing else "ok", detail)


def _installed_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(name)
    except PackageNotFoundError:
        return None


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def run_doctor(settings: Settings | None, env=None, which: Callable[[str], str | None] = shutil.which,
               has_module: Callable[[str], bool] = _has_module, http_get=None,
               version_of: Callable[[str], str | None] = _installed_version) -> list[Check]:
    """Environment checks. Everything the OS or network is asked is injectable, so this is testable."""
    env = os.environ if env is None else env
    out: list[Check] = []
    v = sys.version_info
    out.append(Check("python", "ok" if v >= (3, 10) else "fail", f"{v.major}.{v.minor}.{v.micro} (needs 3.10+)"))
    out.append(Check("platform", "info", f"{sys.platform}, pal-code {__version__}"))
    if settings is None:
        return out

    home = pal_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write-test"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        out.append(Check("data folder", "ok", f"{home} is writable"))
    except OSError as e:
        out.append(Check("data folder", "fail", f"{home}: {e}"))

    out.append(Check("pandoc", "ok" if which("pandoc") else "warn",
                     "found" if which("pandoc") else "not found: HTML/DOCX/ODT/RTF/EPUB/LaTeX attachments will be "
                     "rejected (PDF, PPTX, XLSX and text still work). Install: https://pandoc.org/installing.html"))
    for label, module, why in (("pypdf", "pypdf", "PDF attachments"), ("python-pptx", "pptx", "PowerPoint attachments"),
                               ("openpyxl", "openpyxl", "Excel attachments"), ("graphify", "graphify", "code-project graphs"),
                               ("waitress", "waitress", "production server (Flask dev server is the fallback)")):
        ok = has_module(module)
        out.append(Check(label, "ok" if ok else "warn", "installed" if ok else f"missing: no {why}. pip install pal-code[full]"))
    for label, module, why in (("chromadb", "chromadb", "documents are kept in memory instead of a disk vector store"),
                               ("sentence-transformers", "sentence_transformers", "keyword matching is used instead of embeddings")):
        out.append(Check(label, "ok" if has_module(module) else "info", "installed" if has_module(module) else f"optional: {why}"))

    llm = settings.local_llm
    if llm.get("backend") == "ollama":
        host = (llm.get("host") or "http://localhost:11434").rstrip("/")
        try:
            get = http_get or _http_get_json
            names = [m["name"] for m in get(host + "/api/tags").get("models", [])]
            have = any(n == llm["model"] or n.startswith(llm["model"] + ":") for n in names)
            out.append(Check("local LLM", "ok" if have else "warn",
                             f"{llm['model']} ready" if have else f"Ollama is up but {llm['model']} is not pulled: ollama pull {llm['model']}"))
        except Exception as e:  # noqa: BLE001  (any failure = unreachable)
            out.append(Check("local LLM", "warn", f"Ollama not reachable at {host} ({e.__class__.__name__}): "
                             "routing and input savers need it; every request will fail until it runs"))
    else:
        out.append(Check("local LLM", "info", f"{llm.get('backend')} (not checked)"))

    from pal.chat.providers import API_PROVIDERS
    have_keys = [p for p, var in API_PROVIDERS.items() if env.get(var)]
    out.append(Check("API keys", "ok" if have_keys else "info",
                     ", ".join(have_keys) if have_keys else "none in the environment (you can paste one in the UI)"))

    from pal.llm_wrapper.modes import describe_modes
    for mode in describe_modes(env):
        with_keys = ", ".join(p["name"] for p in mode["providers"] if p["key_set"])
        want = " or ".join(p["key_env"] for p in mode["providers"][:2]) + (" ..." if len(mode["providers"]) > 2 else "")
        out.append(Check(f"mode: {mode['title'].lower()}", "ok" if mode["ready"] else "info",
                         f"ready ({with_keys})" if mode["ready"] else f"not set up: set {want}"))

    from pal.llm_wrapper.components.litellm_wrapper import PINNED
    ll_version = version_of("litellm")
    if ll_version is None:
        out.append(Check("litellm", "warn", f"not installed: cloud APIs need it. Run python run.py, or pip install --require-hashes "
                                            f"-r requirements/litellm.lock (pinned to {PINNED})"))
    elif ll_version != PINNED:
        out.append(Check("litellm", "warn", f"version {ll_version} is installed but this release is pinned to {PINNED}. Reinstall "
                                            "from requirements/litellm.lock (the pin was reviewed; other versions were not)"))
    else:
        out.append(Check("litellm", "ok", f"{ll_version} (pinned; the SDK only, never the proxy server)"))
    fb = settings.fallback
    if not fb.enabled:
        out.append(Check("fallback", "info", "off (set fallback.enabled and fallback.model to answer with a second model when "
                                             "your own provider is rate-limited or down)"))
    elif not fb.model:
        out.append(Check("fallback", "warn", "enabled but fallback.model is empty, so it does nothing"))
    else:
        var = API_PROVIDERS.get(fb.provider)
        if var and not env.get(var):
            out.append(Check("fallback", "warn", f"{fb.provider}/{fb.model} is set but {var} is not in the environment"))
        else:
            out.append(Check("fallback", "ok", f"{fb.provider}/{fb.model} answers when your own provider is rate-limited or down"))

    out.append(_services_check(settings, env))

    for cls in AGENTS.values():
        spec = cls.spec
        if spec is None:
            continue
        ok = bool(agent_key(spec, env))
        out.append(Check(f"agent: {spec.id}", "ok" if ok else "info",
                         "key found" if ok else f"set {spec.key_env} in .env to use the {spec.name}"))

    from pal.os_kernel import system_info
    info = system_info()
    mem = (info.get("memory") or {}).get("total_mb")
    out.append(Check("OS kernel", "info", f"{info['distro']}, kernel {info['kernel']}, {info['cpu_count']} CPUs"
                     + (f", {mem} MB RAM" if mem else "") + " (non-LLM queries go here)"))

    from pal.orchestrator.components.cgroup_manager import CgroupManager
    enforced = CgroupManager.supported()
    out.append(Check("agent resource limits", "info", "cgroup v2 available" if enforced else
                     "not enforced on this OS (only matters for the multi-agent orchestrator, not the chat)"))
    return out


def _http_get_json(url: str) -> dict:
    import requests
    return requests.get(url, timeout=3).json()


def _print_checks(checks: list[Check], stream=None) -> None:
    stream = stream or sys.stdout
    mark = {"ok": "ok  ", "warn": "warn", "fail": "FAIL", "info": "info"}
    for c in checks:
        print(f"[{mark[c.status]}] {c.name}: {c.detail}", file=stream)


def load_dotenv(path: str | os.PathLike | None = None, env=None) -> list[str]:
    """Read KEY=VALUE lines from a .env file into the environment. Never overrides a variable that is already
    set. Returns the names it set (values are never printed or logged)."""
    from pathlib import Path
    env = os.environ if env is None else env
    p = Path(path) if path else Path.cwd() / ".env"
    if not p.is_file():
        return []
    loaded = []
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and value and key not in env:
            env[key] = value
            loaded.append(key)
    return loaded


def _serve(settings: Settings, open_browser: bool = False, page: str = "", graph=None, documents: list | None = None, project_dir=None,
           background=None) -> int:
    from pal.chat import ChatService
    from pal.web import create_app

    s = settings.server
    configure_logging(s.log_level)
    service = ChatService(settings, graph=graph) if graph is not None else ChatService(settings)
    if documents:
        service.preload_docs = list(documents)
    app = create_app(service, studio={"project_dir": project_dir}) if project_dir else create_app(service)
    if background is not None:                                   # pal web: the folder's setup runs while the page is already usable
        import threading as _bg
        _bg.Thread(target=background, args=(service,), daemon=True, name="pal-web-setup").start()
    if s.host not in _LOOPBACK:
        print(f"WARNING: listening on {s.host}. Pal has no login: anyone who can reach this port can use "
              "your API keys' quota and read stored memory. Put it behind an authenticating reverse proxy.",
              file=sys.stderr)
    shown = "localhost" if s.host in ("0.0.0.0", "::") else s.host  # noqa: S104
    url = f"http://{shown}:{s.port}" + ("/" + page.strip("/") if page else "")
    print(f"Pal {__version__} on {url}  (Ctrl+C to stop)")
    if background is None:                                       # pal serve: say what is set up; pal web says nothing more and starts at once
        from pal.chat.providers import API_PROVIDERS
        keys = [p for p, var in API_PROVIDERS.items() if os.environ.get(var)]
        print("API key found for: " + ", ".join(keys) if keys else
              "No API key in the environment: paste one in the page (API key menu), or set e.g. GEMINI_API_KEY.")
        status = service.local_status()
        if status["ok"] is False:
            print(f"WARNING: local model: {status['detail']}. Input reduction and local answers are off until it runs; "
                  "the app still works and sends everything to the API model.", file=sys.stderr)
        else:
            print(f"Local model: {status['detail']}")
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        try:
            from waitress import serve
        except ImportError:
            print("waitress is not installed: using Flask's development server (pip install waitress)",
                  file=sys.stderr)
            app.run(host=s.host, port=s.port, threaded=True)
        else:
            serve(app, host=s.host, port=s.port, threads=s.threads, ident="pal")
    finally:
        service.close()
    return 0


class _UploadFile:
    """Just enough of an uploaded file for the attachment reader."""

    def __init__(self, path):
        from pathlib import Path
        self._path = Path(path)
        self.filename = self._path.name

    def read(self) -> bytes:
        try:
            return self._path.read_bytes()
        except OSError as e:
            raise ConfigError(f"cannot read {self._path}: {e.strerror or e}") from e


def _agent(settings: Settings, name: str, message: str, files: list[str]) -> int:
    """One agent turn from the terminal. The same pipeline, token savers included."""
    from pal.chat import ChatService
    from pal.chat.uploads import read_upload

    service = ChatService(settings)
    try:
        attachments = [read_upload(service, _UploadFile(p), "cli") for p in files]
        res = service.send("cli", "", "", None, message, attachments, agent=name)
    except PalError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        service.close()
    print(res.reply)
    print(f"\n[{res.input_tokens} in / {res.output_tokens} out tokens, {res.latency.get('total', 0)} s]", file=sys.stderr)
    return 0


_CHAT_HELP = ("/file PATH   attach a file or source zip to your next message\n/agent NAME   talk to an agent (coding, docs); "
              "/agent none for plain chat\n/reset        start a new conversation\n/stats        what has been saved so far\n"
              "/help         this list\n/quit         leave")


def render_models(models: list, free_only: bool = False) -> str:
    """OpenRouter's models as a plain table: the name to pass as --model, the context size, the price, and whether it is free."""
    if not models:
        return "No models match.\n"
    width = max(len(m.id) for m in models)
    rows = [f"{'MODEL (pass as --model)':<{width}}  {'CONTEXT':>8}  PRICE (USD per million tokens in / out)"]
    for m in models:
        ctx = f"{m.context_length:,}" if m.context_length else "?"
        price = "free" if m.free else (f"{m.prompt_per_million:g} / {m.completion_per_million:g}" if m.prompt_per_million is not None and m.completion_per_million is not None else "?")
        rows.append(f"{m.id:<{width}}  {ctx:>8}  {price}")
    rows.append(f"\n{len(models)} model(s). Use one with: pal chat --mode explore --model <MODEL>")
    return "\n".join(rows) + "\n"


def _models(args, fetch=None) -> int:
    """pal models [--free] [--search WORDS] [--limit N] [--json]: browse OpenRouter's models (needs no key, needs the network)."""
    from pal.llm_wrapper.components.openrouter_wrapper import fetch_models, search_models
    try:
        found = search_models(fetch_models(fetch), args.search or "", args.free, args.limit)
    except Exception as e:  # noqa: BLE001
        print(f"could not read OpenRouter's model list ({type(e).__name__}: {e}). Check your network, or type a model name yourself.", file=sys.stderr)
        return 1
    if args.json:
        import json
        from dataclasses import asdict
        print(json.dumps([asdict(m) for m in found], indent=2))
    else:
        print(render_models(found, args.free), end="")
    return 0


def _pick_provider(settings: Settings, provider: str | None, model: str | None, env, mode: str | None = None) -> tuple[str, str]:
    """Which API provider and model plain chat uses: what you asked for, else the first provider that has a key. `mode` (explore or own_key)
    limits the choice to that mode's providers."""
    from pal.chat.providers import API_PROVIDERS
    from pal.llm_wrapper.modes import MODES, mode_for
    if mode and provider and mode_for(provider) != mode:
        raise ConfigError(f"{provider} is not an {MODES[mode].title.lower()} provider (mode {mode}: {', '.join(MODES[mode].providers)})")
    if not provider:
        allowed = MODES[mode].providers if mode else tuple(API_PROVIDERS)
        provider = next((p for p in allowed if env.get(API_PROVIDERS[p])), "")
        if not provider and mode == "explore":
            raise ConfigError("Explore mode needs an OpenRouter key: set OPENROUTER_API_KEY (pal models lists what it can reach)")
        if not provider and mode:
            raise ConfigError("own-key mode needs a key for one provider: set for example GEMINI_API_KEY, or pass --provider")
    if not provider:
        raise ConfigError("no API key found: set one (for example GEMINI_API_KEY) or pass --provider")
    if not model:
        if provider != "gemini":
            raise ConfigError(f"choose a model: --model <name> (for {provider})")
        model = settings.agents.model
    return provider, model


def _chat(settings: Settings, agent: str | None, provider: str | None, model: str | None, files: list[str],
          input_fn: Callable[[str], str] | None = None, env=None, mode: str | None = None) -> int:
    """An interactive terminal chat: the same pipeline as the web page (token savers, scheduler, agents)."""
    from pal.chat import ChatService
    from pal.chat.uploads import read_upload

    env = os.environ if env is None else env
    read = input_fn if input_fn is not None else input      # looked up now, so it can be replaced (and tested)
    service = ChatService(settings)
    session, pending = "terminal", []
    try:
        provider, model = _pick_provider(settings, provider, model, env, mode) if not agent else (provider or "", model or "")
        print(f"Pal chat ({'agent: ' + agent if agent else provider + '/' + model}). /help lists commands.", file=sys.stderr)
        for p in files:
            pending.append(read_upload(service, _UploadFile(p), session))
        while True:
            try:
                line = read("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, arg = line.partition(" ")
                if cmd in ("/quit", "/exit"):
                    return 0
                if cmd == "/help":
                    print(_CHAT_HELP)
                elif cmd == "/reset":
                    service.reset(session)
                    pending.clear()
                    print("new conversation")
                elif cmd == "/agent":
                    name = arg.strip()
                    if name in ("", "none"):
                        agent = None
                        provider, model = _pick_provider(settings, provider, model, env, mode)
                    elif name in AGENTS:
                        agent = name
                    else:
                        print(f"unknown agent {name!r}; choose one of: {', '.join(AGENTS)}")
                        continue
                    print(f"now talking to: {agent or 'plain chat'}")
                elif cmd == "/file":
                    try:
                        pending.append(read_upload(service, _UploadFile(arg.strip()), session))
                        print(f"attached {pending[-1].get('name', arg.strip())}; it goes with your next message")
                    except PalError as e:
                        print(f"error: {e}")
                elif cmd == "/stats":
                    s = service.metrics.chat_summary(session)
                    print(f"turns {s.get('turns', 0)} | tokens saved (estimate) {s.get('tokens_saved_estimate', {})} | "
                          f"cost ${s.get('cost_usd', 0)} | saved about ${s.get('saved_cost_usd', 0)}")
                else:
                    print("unknown command; /help lists them")
                continue
            try:
                res = service.send(session, provider or "", model or "", None, line, pending, agent=agent)
            except PalError as e:
                print(f"error: {e}")
                continue
            pending = []
            print(res.reply)
            note = [res.handled_by, f"{res.input_tokens} in / {res.output_tokens} out"]
            if res.fallback:
                note.append(f"answered by fallback {res.fallback['provider']}/{res.fallback['model']}")
            if res.cost_usd:
                note.append(f"${res.cost_usd:.4f}")
            print("[" + " | ".join(note) + "]", file=sys.stderr)
    except PalError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        service.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pal", description="Pal chat layer")
    p.add_argument("--version", action="version", version=f"pal-code {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("serve", "run the web chat"), ("doctor", "check this machine"),
                        ("config", "print the effective settings"), ("agent", "run the coding or documentation agent"),
                        ("chat", "chat in the terminal"), ("tools", "list every tool by name"),
                        ("models", "browse the models OpenRouter can reach (Explore mode)")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", help="path to a settings YAML file (or set PAL_CONFIG)")
        if name == "agent":
            sp.add_argument("name", choices=list(AGENTS))
            sp.add_argument("message")
            sp.add_argument("--file", action="append", default=[], help="attach a file or source zip (repeatable)")
        if name == "models":
            sp.add_argument("--search", help="only models whose name contains these words")
            sp.add_argument("--free", action="store_true", help="only free models")
            sp.add_argument("--limit", type=int, default=30, help="at most this many (default 30; 0 for all)")
            sp.add_argument("--json", action="store_true")
        if name == "chat":
            sp.add_argument("--agent", choices=list(AGENTS))
            sp.add_argument("--mode", choices=["explore", "own_key"], help="explore: one OpenRouter key, many models. own_key: your key for one provider")
            sp.add_argument("--provider")
            sp.add_argument("--model")
            sp.add_argument("--file", action="append", default=[], help="attach a file or source zip (repeatable)")
        if name == "serve":
            sp.add_argument("--host")
            sp.add_argument("--port", type=int)
            sp.add_argument("--open", action="store_true", help="open the page in your browser")
            sp.add_argument("--page", choices=["", "studio"], default="", help="which page to open: the chat (default) or the Studio")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    try:
        overrides: dict = {}
        if args.cmd == "serve":
            server = {k: v for k, v in (("host", args.host), ("port", args.port)) if v is not None}
            overrides = {"server": server} if server else {}
        settings = load_settings(args.config, overrides=overrides)
    except ConfigError as e:
        if args.cmd == "doctor":
            _print_checks(run_doctor(None) + [Check("config", "fail", str(e))])
            return 1
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "config":
        print(yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True), end="")
        return 0
    if args.cmd == "doctor":
        checks = run_doctor(settings)
        _print_checks(checks)
        failed = [c for c in checks if c.status == "fail"]
        print(f"\n{len(failed)} problem(s)." if failed else "\nReady. Run: pal cli  (terminal)  or  pal web  (browser)")
        return 1 if failed else 0
    if args.cmd == "agent":
        return _agent(settings, args.name, args.message, args.file)
    if args.cmd == "tools":
        from pal.tool_catalog import render
        print(render(settings.tools, os.environ), end="")
        return 0
    if args.cmd == "models":
        return _models(args)
    if args.cmd == "chat":
        return _chat(settings, args.agent, args.provider, args.model, args.file, mode=args.mode)
    return _serve(settings, open_browser=args.open, page=args.page)


if __name__ == "__main__":
    raise SystemExit(main())
