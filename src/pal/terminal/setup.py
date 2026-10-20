"""`pal setup`: look at this computer's GPU, pick the local models that suit it, download them with Ollama, and remember the choice.

Runs by itself the first time `pal cli` or `pal web` starts in a terminal. Two models are chosen:
- the main model, which builds and plans when you use a local model;
- the helper, the small model Pal runs for its own jobs on every request (routing, trimming what you type, checking answers), even when the
  main model is an API model.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from pal.llm_wrapper.modes import PROVIDER_KEYS
from pal.terminal import backend, theme
from pal.terminal.hardware import Hardware, detect
from pal.terminal.local_models import BY_TAG, FIT_LABEL, budget_gb, by_tier, fit, recommend, suitable, tier_name

INSTALL_OLLAMA = {"Windows": "download the installer from https://ollama.com/download",
                  "Darwin": "download the app from https://ollama.com/download (or: brew install ollama)",
                  "Linux": "run: curl -fsSL https://ollama.com/install.sh | sh"}


def _say(console: Console, renderable, indent: int = 2) -> None:
    console.print(Padding(renderable, (0, 0, 0, indent)))


def ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def pull(tag: str) -> bool:
    """`ollama pull <tag>`, showing Ollama's own progress. True when it worked."""
    exe = shutil.which("ollama")
    if not exe:
        return False
    try:
        return subprocess.call([exe, "pull", tag]) == 0
    except OSError:
        return False


def show_table(console: Console, hw: Hardware) -> None:
    """Every model in the catalogue, grouped by the GPU it needs, with how each fits this computer."""
    _say(console, Text.assemble(("Local models by GPU memory", f"bold {theme.WHITE}"), (f"   this computer: {tier_name(hw)}", theme.MUTED)))
    console.print()
    for tier, models in by_tier():
        head = f"{tier} GB GPU" if tier else "CPU"
        mine = max((t for t, _ in by_tier() if t <= hw.vram_gb + 0.5), default=None) if hw.vram_gb else 0
        mark = "  \u25C0 your computer" if tier == mine else ""
        _say(console, Text.assemble((head, f"bold {theme.RED_HI}"), (mark, theme.WHITE)))
        grid = Table.grid(padding=(0, 2))
        for width in (22, 8, 10, 26):
            grid.add_column(width=width, no_wrap=True, overflow="ellipsis")
        grid.add_column(no_wrap=True, overflow="ellipsis")
        for m in models:
            f = fit(m.need_gb, hw)
            grid.add_row(Text(m.tag, style=theme.WHITE), Text(f"{m.size_gb:g} GB", style=theme.INK), Text(m.use + (" *" if m.thinks else ""), style=theme.MUTED),
                         Text(FIT_LABEL[f], style=f"bold {theme.RED_HI}" if f == "gpu" else theme.INK if f == "partial" else theme.MUTED),
                         Text(m.note, style=theme.MUTED))
        _say(console, grid, 4)
        console.print()
    _say(console, Text("Sizes are Ollama's 4-bit downloads (approximate). * prints its reasoning first, so Pal does not pick it by itself.", style=theme.MUTED))


def default_pick(console: Console, question: str, options: list[tuple[str, str]], recommended: int = 0, other: bool = False,
                 header: str = "Setup") -> str | None:
    """A choice as a card in a terminal; without one (pipes, tests), the recommended choice."""
    from pal.terminal import decide
    if not decide._interactive():
        return options[recommended or 0][0] if options else None
    q = decide.Question(question, [decide.Option(label, about) for label, about in options], header, recommended, other=other)
    answer = decide.ask(console, q)
    decide.show_answer(console, q, answer)
    return answer


LOCAL_MAIN = "A local model on this computer"
API_MAIN = "An API model"


def run_setup(console: Console, hw: Hardware | None = None, host: str = backend.DEFAULT_HOST, assume_yes: bool = False,
              ask: Callable[[str], bool] | None = None, puller: Callable[[str], bool] = pull, ollama: Callable[[], bool] = ollama_installed, dry: bool = False,
              pick: Callable[..., str | None] | None = None, main_kind: str | None = None, api_chooser: Callable | None = None) -> dict:
    """Scan, choose, download (if allowed), save. Returns what was saved. `dry` only shows the recommendation (for a pretend computer).

    The main model (it writes and plans) can be local or an API model. With an API model only the helper is downloaded. With a local one,
    the models that suit this GPU are offered (the best first) and the one chosen is downloaded with the helper."""
    hw = hw or detect()
    rec = recommend(hw)
    pick = pick or (lambda question, options, recommended=0, other=False: default_pick(console, question, options, recommended, other))
    _say(console, Text("Pal setup", style=f"bold {theme.WHITE}"))
    _say(console, Text(f"This computer: {hw.summary()}", style=theme.INK))
    for note in hw.notes:
        _say(console, Text(note, style=theme.RED_HI))
    console.print()
    if rec is None:
        explain_api_only(console)
        if not dry:
            backend.save_prefs(helper_model="", setup_done="1")
        return {"provider": "", "model": "", "helper_model": "", "downloaded": False, "api_only": True}
    grid = Table.grid(padding=(0, 3))
    for role, m, why in (("Best main model", rec.main, "builds and plans when you use a local model"), ("Helper", rec.helper, "Pal's own small jobs on every request: routing, trimming, checking")):
        grid.add_row(Text(role, style=theme.MUTED), Text(m.tag, style=f"bold {theme.WHITE}"), Text(f"{m.size_gb:g} GB", style=theme.INK), Text(why, style=theme.MUTED))
    _say(console, Text("For this computer", style=f"bold {theme.RED_HI}"))
    _say(console, grid, 4)
    _say(console, Text(f"Why: {rec.reason}.", style=theme.MUTED), 4)
    console.print()
    if dry:
        if rec.others:
            _say(console, Text("Also fits: " + ", ".join(m.tag for m in rec.others) + "   (every model: pal setup --list)", style=theme.MUTED), 4)
        return {"provider": backend.LOCAL, "model": rec.main.tag, "helper_model": rec.helper.tag, "downloaded": False}

    kind = main_kind or pick("Which main model should Pal use? (it writes and plans; the helper is always local)",
                             [(LOCAL_MAIN, f"runs on your GPU, free and private: {rec.main.tag} fits best"),
                              (API_MAIN, "OpenAI, Anthropic, Gemini, OpenRouter or any other: needs a key, nothing big to download")], 0)
    api = kind in (API_MAIN, "api")

    if not ollama():
        how = INSTALL_OLLAMA.get(hw.system, INSTALL_OLLAMA["Linux"])
        _say(console, Text.assemble(("Ollama runs local models (the helper too) and is not installed yet. ", theme.WHITE), (f"To install it, {how}.", theme.INK)))
        _say(console, Text("Then run  pal setup  again. An API model works meanwhile:  pal cli --provider openrouter --model <name>", style=theme.MUTED))
        backend.save_prefs(helper_model=rec.helper.tag, **({} if api else {"provider": backend.LOCAL, "model": rec.main.tag}))
        return {"provider": "" if api else backend.LOCAL, "model": "" if api else rec.main.tag, "helper_model": rec.helper.tag, "downloaded": False}

    info = backend.local_info(host)
    installed = backend.chat_models(info) if info.get("ok") else []
    main_tag = ""
    saved: dict = {"helper_model": rec.helper.tag}
    if api:
        chosen = (api_chooser or (lambda c, system: ask_api(c, system)))(console, hw.system)
        if chosen:
            saved.update(provider=chosen[0], model=chosen[1])
        else:
            _say(console, Text("No API model chosen yet: pick one in a session with  /model <provider>:<model>", style=theme.MUTED))
    else:
        options = suitable(hw, installed)
        labels = [(tag, about) for tag, about, _ in options]
        main_tag = pick("Which local model should be the main one? (the best for your GPU is first; type another name to use it)", labels, 0, True) or rec.main.tag
        main_tag = main_tag.strip().removeprefix("ollama pull ").strip()
        saved.update(provider=backend.LOCAL, model=main_tag)

    missing = [tag for tag in (main_tag, rec.helper.tag) if tag and tag not in installed]
    if not missing:
        _say(console, Text("Everything needed is already installed.", style=theme.INK))
        backend.save_prefs(**saved, setup_done="1")
        return {**{"provider": "", "model": ""}, **saved, "downloaded": True}
    sizes = {m.tag: m.size_gb for m in (BY_TAG.get(x) for x in missing) if m}
    total = sum(sizes.values())
    question = f"Download {', '.join(missing)} now" + (f" (about {total:g} GB)" if total else "") + "?"
    agreed = assume_yes or (ask or (lambda q: _confirm(console, q)))(question)
    if not agreed:
        _say(console, Text("Skipped. Download later with  pal setup, or pull any model yourself:  ollama pull <name>", style=theme.MUTED))
        backend.save_prefs(**saved, setup_done="1")
        return {**{"provider": "", "model": ""}, **saved, "downloaded": False}
    ok = True
    for tag in missing:
        _say(console, Text(f"Downloading {tag} ...", style=theme.WHITE))
        ok = puller(tag) and ok
    _say(console, Text("Ready." if ok else "A download did not finish. Run  pal setup  again to retry.", style=f"bold {theme.RED_HI}" if ok else theme.RED_HI))
    backend.save_prefs(**saved, setup_done="1" if ok else "")
    return {**{"provider": "", "model": ""}, **saved, "downloaded": ok}


KEY_HOWTO = {"Windows": 'setx {var} "your-key"   (then open a new terminal)', "Darwin": 'add  export {var}="your-key"  to ~/.zshrc',
             "Linux": 'add  export {var}="your-key"  to ~/.bashrc'}


def explain_api_only(console: Console) -> None:
    keyed = [p for p, var in PROVIDER_KEYS.items() if os.environ.get(var)]
    _say(console, Text("No GPU on this computer, so Pal uses an API model.", style=f"bold {theme.RED_HI}"))
    _say(console, Text("Local models and Pal's local helper jobs (routing, trimming what you type, checking answers) are off here. The savers that need no "
                       "model still work: only the relevant history, files and code are sent, and files are not sent twice.", style=theme.INK), 4)
    console.print()
    if keyed:
        _say(console, Text(f"API key found for: {', '.join(keyed)}.", style=theme.WHITE))
    _say(console, Text("Start with:  pal cli --provider openrouter --model <name>      browse models:  pal models --free", style=theme.INK))
    _say(console, Text("or in a session:  /key openrouter <key>  then  /model openrouter:<name>", style=theme.MUTED))
    _say(console, Text("Keys are never saved by Pal. To keep one, set it in your environment, for example OPENROUTER_API_KEY.", style=theme.MUTED))


def ask_api(console: Console, system: str = "", ask: Callable[[str], str] | None = None, secret: Callable[[str], str] | None = None) -> tuple[str, str] | None:
    """For a computer without a GPU: ask which API provider and model to use, and the key when it is not in the environment.
    The key is put in this process's environment only (never written anywhere). Returns (provider, model), or None if the person stops."""
    import getpass
    ask = ask or (lambda q: console.input(Text("  " + q + " ", style=theme.WHITE)))
    secret = secret or (lambda q: getpass.getpass("  " + q + " "))
    names = list(PROVIDER_KEYS)
    keyed = [p for p in names if os.environ.get(PROVIDER_KEYS[p])]
    default = keyed[0] if keyed else "openrouter"
    try:
        provider = (ask(f"API provider ({', '.join(names)}) [{default}]:").strip().lower() or default)
        if provider not in PROVIDER_KEYS:
            _say(console, Text(f"Unknown provider {provider!r}.", style=theme.RED_HI))
            return None
        model = ask("Model name (for example the one your provider lists; pal models --free shows OpenRouter's):").strip()
        if not model:
            return None
        var = PROVIDER_KEYS[provider]
        if not os.environ.get(var):
            key = secret(f"Paste your {provider} API key (it is not shown and not saved):").strip()
            if not key:
                return None
            os.environ[var] = key
            how = KEY_HOWTO.get(system or "Linux", KEY_HOWTO["Linux"]).format(var=var)
            _say(console, Text(f"Using it for this session. To keep it for next time: {how}", style=theme.MUTED))
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None
    backend.save_prefs(provider=provider, model=model)
    return provider, model


def _confirm(console: Console, question: str) -> bool:
    try:
        answer = console.input(Text.assemble(("  " + question + " ", theme.WHITE), ("[Y/n] ", theme.MUTED))).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    return answer in ("", "y", "yes")


def needs_setup() -> bool:
    return not backend.load_prefs().get("setup_done") and not os.environ.get("PAL_SKIP_SETUP")


def budget_note(hw: Hardware) -> str:
    return f"about {budget_gb(hw):g} GB for models"


# -- getting local models ready when a session starts -------------------------------------------------------------------------------


OLLAMA_START_WAIT_S = 20


@dataclass
class LocalStatus:
    state: str              # "ready", "off" (no GPU), "missing" (Ollama not installed), "stopped" (could not start), "remote"
    detail: str = ""
    info: dict | None = None


def _is_local_host(host: str) -> bool:
    return any(h in host for h in ("localhost", "127.0.0.1", "[::1]"))


def install_command(system: str) -> list[str] | None:
    """The command that installs Ollama on this computer, or None when it has to be downloaded by hand."""
    if system == "Linux" and shutil.which("curl") and shutil.which("sh"):
        return ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
    if system == "Darwin" and shutil.which("brew"):
        return ["brew", "install", "ollama"]
    if system == "Windows" and shutil.which("winget"):
        return ["winget", "install", "-e", "--id", "Ollama.Ollama", "--accept-source-agreements", "--accept-package-agreements"]
    return None


def install_ollama(console: Console, system: str, run: Callable[[list[str]], int] = subprocess.call) -> bool:
    cmd = install_command(system)
    if cmd is None:
        _say(console, Text(f"Install Ollama by hand: {INSTALL_OLLAMA.get(system, INSTALL_OLLAMA['Linux'])}. Then run pal again.", style=theme.INK))
        return False
    _say(console, Text("Installing Ollama: " + " ".join(cmd), style=theme.MUTED))
    try:
        ok = run(cmd) == 0
    except OSError:
        ok = False
    if system == "Windows" and not ollama_installed():          # winget puts it here; this terminal's PATH is from before
        exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama"
        if (exe / "ollama.exe").is_file():
            os.environ["PATH"] = f"{exe}{os.pathsep}{os.environ.get('PATH', '')}"
    return ok and ollama_installed()


def start_ollama(host: str, wait_s: float = OLLAMA_START_WAIT_S, info: Callable[[str], dict] | None = None,
                 popen: Callable | None = None, sleep: Callable[[float], None] | None = None) -> bool:
    """Start `ollama serve` in the background (it keeps running after Pal exits, like the Ollama app) and wait until it answers."""
    import time
    info = info or backend.local_info
    sleep = sleep or time.sleep
    exe = shutil.which("ollama")
    if not exe:
        return False
    from pal.paths import pal_home
    log = pal_home() / "ollama.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"stdout": log.open("ab"), "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        (popen or subprocess.Popen)([exe, "serve"], **kwargs)
    except OSError:
        return False
    waited = 0.0
    while waited < wait_s:
        if info(host).get("ok"):
            return True
        sleep(0.5)
        waited += 0.5
    return bool(info(host).get("ok"))


def ensure_local(console: Console, hw: Hardware, host: str = backend.DEFAULT_HOST, interactive: bool = True,
                 ask: Callable[[str], bool] | None = None, installer: Callable | None = None, starter: Callable | None = None,
                 info: Callable[[str], dict] | None = None, setup: Callable | None = None, ollama: Callable[[], bool] | None = None,
                 choose: bool = True) -> LocalStatus:
    """Make local models usable before the session starts: install Ollama (asking first), start it, and download the models chosen
    for this GPU when they are missing (asking first). Never fails: the status says what is not ready and the session goes on."""
    from pal.terminal.local_models import api_only
    info = info or backend.local_info
    ollama = ollama or ollama_installed
    ask = ask or (lambda q: _confirm(console, q))
    if api_only(hw):
        return LocalStatus("off", "no GPU: API models only, local helper jobs off")
    if not _is_local_host(host):
        state = info(host)
        return LocalStatus("ready" if state.get("ok") else "remote", f"Ollama at {host}" + ("" if state.get("ok") else " is not answering"), state)
    if not ollama():
        if interactive and ask("Local models run with Ollama, which is not installed. Install it now?"):
            (installer or install_ollama)(console, hw.system)
        if not ollama():
            return LocalStatus("missing", f"Ollama is not installed: {INSTALL_OLLAMA.get(hw.system, INSTALL_OLLAMA['Linux'])}")
    state = info(host)
    if not state.get("ok"):
        if not (starter or start_ollama)(host):
            return LocalStatus("stopped", "Ollama did not start: see ~/.pal/ollama.log, or start it yourself with  ollama serve")
        state = info(host)
    prefs = backend.load_prefs()
    have = set(backend.chat_models(state))
    rec = recommend(hw)
    local_main = prefs.get("provider", backend.LOCAL) == backend.LOCAL           # an API main model needs no big local download
    wanted = [m for m in ((prefs.get("model") or (rec.main.tag if rec else "")) if local_main else "",
                          prefs.get("helper_model") or (rec.helper.tag if rec else "")) if m]
    if choose and interactive and (needs_setup() or any(m not in have for m in wanted)):     # (pal cli asks for its models itself)
        (setup or run_setup)(console, hw, host=host, ask=ask)
        state = info(host)
        prefs = backend.load_prefs()
    installed = backend.chat_models(state)
    if not installed:
        return LocalStatus("ready", "Ollama running, no models yet: pal setup downloads the ones for this GPU", state)
    main = prefs.get("model") if prefs.get("provider") == backend.LOCAL and prefs.get("model") in installed else ""
    helper = prefs.get("helper_model") if prefs.get("helper_model") in installed else ""
    detail = "Ollama running" + (f" · {main}" if main else f" · {len(installed)} model{'s' if len(installed) != 1 else ''}") + (f" · helper {helper}" if helper else "")
    return LocalStatus("ready", detail, state)


# -- choosing the models when a session starts ----------------------------------------------------------------------------------------

API_OPTION = "An external API model"
NO_HELPER = "No helper"


def helper_choices(hw: Hardware, installed: list[str], exclude: str = "") -> list[tuple[str, str]]:
    """Small local models that can be Pal's helper on this computer, installed ones first, each marked installed or with its download size."""
    from pal.terminal.local_models import CATALOGUE, lookup
    have = {(lookup(n).tag if lookup(n) else n) for n in installed}
    rows = [m for m in CATALOGUE if m.helper and not m.thinks and m.tag != exclude and fit(m.need_gb, hw) != "too big"]
    rows.sort(key=lambda m: (m.tag not in have, -m.rank))
    return [(m.tag, ("installed · " if m.tag in have else f"{m.size_gb:g} GB download · ") + (m.note or "small and fast")) for m in rows]


def main_choices(hw: Hardware, installed: list[str]) -> list[tuple[str, str]]:
    """Local models for the main role: the ones that suit this GPU (installed ones marked), then any other installed model that fits."""
    from pal.terminal.local_models import installed_fit
    from pal.terminal.local_models import lookup
    helpers = {tag for tag, _ in helper_choices(hw, [])}
    real = {}                                       # catalogue name -> the name it is installed under (llama3.1:8b -> llama3.1:8b-instruct-q4_K_M)
    for n in installed:
        known = lookup(n)
        real.setdefault(known.tag if known else n, n)
    out = [(real.get(tag, tag), about) for tag, about, _ in suitable(hw, [n for n in installed if n not in helpers])]
    for row in installed_fit([{"name": n} for n in installed], hw):
        if row["name"] not in helpers and row["fit"] == "gpu" and all(row["name"] != t for t, _ in out):
            out.append((row["name"], "installed"))
    out.sort(key=lambda o: not o[1].startswith("installed"))          # what is already on the computer first, then the best downloads
    return out[:7]


def _get(console: Console, tag: str, installed: list[str], confirm: Callable[[str], bool], puller: Callable[[str], bool]) -> bool:
    """Download a model that is not installed yet, after asking. True when it is ready."""
    if tag in installed:
        return True
    size = BY_TAG[tag].size_gb if tag in BY_TAG else 0
    if not confirm(f"{tag} is not installed. Download it now" + (f" (about {size:g} GB)" if size else "") + "?"):
        return False
    _say(console, Text(f"Downloading {tag} ...", style=theme.WHITE))
    if puller(tag):
        return True
    _say(console, Text(f"The download of {tag} did not finish.", style=theme.RED_HI))
    return False


def choose_models(console: Console, hw: Hardware, host: str = backend.DEFAULT_HOST, pick: Callable[..., str | None] | None = None,
                  puller: Callable[[str], bool] = pull, api_chooser: Callable | None = None, confirm: Callable[[str], bool] | None = None,
                  info: dict | None = None, helper: bool = True) -> bool:
    """Ask which models to use for this session; nothing is chosen in advance. The main model is an external API model (any provider, the key
    is never saved) or a local model that suits this GPU (installed ones are marked; others are downloaded after asking). Then the helper: a small
    local model for Pal's own jobs, or none. The choice is saved. False when the person chose nothing."""
    pick = pick or (lambda question, options, recommended=None, other=False, header="Setup":
                    default_pick(console, question, options, recommended, other, header))
    confirm = confirm or (lambda q: _confirm(console, q))
    info = info if info is not None else backend.local_info(host)
    ready = bool(info.get("ok"))
    installed = backend.chat_models(info) if ready else []
    options = [(API_OPTION, "OpenAI, Anthropic, Gemini, OpenRouter or any LiteLLM model: needs your key (never saved)")]
    if ready:
        options += main_choices(hw, installed)
    else:
        _say(console, Text("Local models are not available right now (Ollama is not running), so only API models are offered.", style=theme.MUTED))
    answer = pick(f"Which main model should Pal use? (it writes and plans) · {hw.summary()}", options, None, ready, "Main model")
    if not answer:
        return False
    answer = answer.strip().removeprefix("ollama pull ").strip()
    if answer == API_OPTION:
        chosen = (api_chooser or (lambda c, system: ask_api(c, system)))(console, hw.system)
        if not chosen:
            return False
    else:
        if not _get(console, answer, installed, confirm, puller):
            return False
        backend.save_prefs(provider=backend.LOCAL, model=answer)
    if not ready:
        backend.save_prefs(helper_model="", setup_done="1")
        return True
    if not helper:                                     # Pal chooses the helper itself (auto_helper)
        backend.save_prefs(setup_done="1")
        return True
    main = "" if answer == API_OPTION else answer
    helpers = helper_choices(hw, installed, exclude=main) + [(NO_HELPER, "the main model also does Pal's small jobs (uses more of its tokens)")]
    helper = pick("Which small local helper should do Pal's own jobs? (routing easy questions, trimming what you type)", helpers, None, True, "Helper")
    helper = (helper or NO_HELPER).strip().removeprefix("ollama pull ").strip()
    if helper == NO_HELPER or not _get(console, helper, installed, confirm, puller):
        helper = ""
    backend.save_prefs(helper_model=helper, setup_done="1")
    return True


AGENTS = [("Ask", "ask", "answers questions about your code and anything else; never changes files"),
          ("Plan", "plan", "reads your folder and writes a step-by-step plan; never changes files"),
          ("Build", "build", "writes and changes files in this folder, after you review them")]


def choose_agent(console: Console, pick: Callable[..., str | None] | None = None, current: str = "") -> str | None:
    """Which agent to work with: Ask, Plan or Build, nothing chosen in advance. Saved for next time. Returns the mode, or None."""
    pick = pick or (lambda question, options, recommended=None, other=False, header="Setup":
                    default_pick(console, question, options, recommended, other, header))
    question = "Which agent do you want to use? (tab switches later, or /agent)" + (f" · now: {current.capitalize()}" if current else "")
    answer = pick(question, [(label, about) for label, _, about in AGENTS], None, False, "Agent")
    mode = next((m for label, m, _ in AGENTS if label == answer), None)
    if mode:
        backend.save_prefs(mode=mode)
    return mode


def auto_helper(console: Console, hw: Hardware, host: str = backend.DEFAULT_HOST, puller: Callable[[str], bool] = pull,
                info: dict | None = None) -> str:
    """Pal chooses its own helper: the best small model for this computer. Already installed: used as it is. Not installed: downloaded now
    (Ollama shows its progress). Nothing is asked. Returns the helper's name, or '' when there is none (no GPU, Ollama not running, failed)."""
    from pal.terminal.local_models import api_only, pick_helper
    if api_only(hw):
        return ""
    info = info if info is not None else backend.local_info(host)
    if not info.get("ok"):
        return ""
    best = pick_helper(hw)
    if best.tag not in backend.chat_models(info):
        _say(console, Text(f"Getting Pal's helper model ready: {best.tag} (about {best.size_gb:g} GB, once). It does Pal's small jobs, like trimming what you type.",
                           style=theme.MUTED))
        if not puller(best.tag):
            _say(console, Text(f"The helper {best.tag} could not be downloaded: Pal works without it for now and tries again next time.", style=theme.MUTED))
            backend.save_prefs(helper_model="")
            return ""
    backend.save_prefs(helper_model=best.tag)
    return best.tag
