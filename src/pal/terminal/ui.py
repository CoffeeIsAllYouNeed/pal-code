"""The Pal screen: a centred column in the terminal, a prompt box with a red bar, and a status line under it.

The box is a small prompt_toolkit application that disappears when you press Enter; what you said and what Pal answered are then printed with rich,
so they stay in the terminal's own scroll-back and can be selected and copied. Colours are black, red and white.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, VSplit, Window, WindowAlign
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.styles import Style
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from pal.terminal import theme
from pal.terminal.logo import BANNER_ROWS, banner, logo_lines, ptk_style
from pal.terminal.security import SEVERITIES, Finding, clean_terminal
from pal.terminal.session import Change, PalSession, Turn

COMMANDS: list[tuple[str, str]] = [
    ("/help", "show the commands"), ("/agent", "choose the agent: Ask, Plan or Build (or press tab)"), ("/mode", "switch agent: build, plan or ask (or press tab)"), ("/model", "show or switch the model"),
    ("/files", "list the files Pal can see in this folder"), ("/security", "check this folder for security problems (add 'ai' for a model review)"),
    ("/budget", "cap the tokens this session may use: /budget 50000, /budget off"), ("/key", "use an API key for this session: /key openrouter sk-..."),
    ("/add", "attach a file from anywhere (PDF, Word, code ...); dropping a file on the window works too"), ("/tokens", "tokens saved, and where (last, turns, help)"),
    ("/project", "this folder's setup: project map, code graph, documents as markdown, local model (/project rebuild)"),
    ("/exit", "leave Pal"),
]
PLACEHOLDER = 'Build anything\u2026  "Make me a Flappy Bird game\u2026"'
PLACEHOLDERS = {"build": PLACEHOLDER,
                "plan": 'Plan anything\u2026  "Plan a todo app with login\u2026"',
                "ask": 'Ask anything\u2026  "How does the cart total work?"',
                "bench": 'Bench a task\u2026  "Make me a Flappy Bird game\u2026"  (run plain and with Pal, then compared)'}
RICH_THEME = Theme({"markdown.code": f"{theme.RED_HI} on {theme.PANEL}", "markdown.code_block": f"{theme.INK} on {theme.PANEL}", "markdown.h1": f"bold {theme.RED_HI}",
                    "markdown.h2": f"bold {theme.WHITE}", "markdown.h3": f"bold {theme.WHITE}", "markdown.link": f"underline {theme.WHITE}", "markdown.item.bullet": theme.RED, "markdown.item.number": theme.RED_HI})
STYLE = Style.from_dict({                       # the chatbox: Pal's red and white
    "screen": f"bg:{theme.BG}", "placeholder": theme.MUTED, "prompt": "#B8323A bold", "frame.border": "#8C8C8C",
    "mode.build": "#B8323A bold", "mode.plan": "#B8323A bold", "mode.ask": "#B8323A bold",
    "dd": "#FFFFFF", "dd.on": "#FFFFFF bold reverse", "dd.star": "#FFFFFF bold", "dd.desc": "#BDBDBD", "model": theme.WHITE,
    "muted": theme.MUTED, "hint": theme.MUTED, "hintkey": f"{theme.WHITE} bold", "ok": f"{theme.RED_HI} bold", "cmd": f"{theme.WHITE} bold",
    "cmddesc": theme.MUTED, "warn": theme.RED_HI, "link": "underline", "ask": f"{theme.RED_HI} bold", "askq": f"{theme.WHITE} bold",
    "mic": "#FFFFFF", "wave": f"{theme.RED_HI} bold", "pick": f"{theme.RED_HI} bold",
})


def metrics(cols: int | None = None) -> tuple[int, int]:
    """(width of the conversation column, spaces on its left): the full width of the terminal, inside a small margin, like the input box."""
    cols = cols or shutil.get_terminal_size((100, 30)).columns
    return max(40, min(cols - 6, 140)), 3


def make_console() -> Console:
    return Console(theme=RICH_THEME, highlight=False, soft_wrap=False)


def short_path(p: Path) -> str:
    home = Path.home()
    try:
        return "~" + os.sep + str(p.relative_to(home)) if p != home else "~"
    except ValueError:
        return str(p)


def say(console: Console, renderable: RenderableType) -> None:
    """Print inside the centred column."""
    width, pad = metrics(console.width)
    console.print(Padding(renderable, (0, 0, 0, pad)), width=pad + width)


_PINNED = {"rows": 0}          # how many lines at the top hold Pal's header (0: none)


def clear_screen(console: Console) -> None:
    """Really clear the window, scrollback too: Windows needs `cls` (an escape code only moves the cursor in some consoles)."""
    import os
    import sys
    if os.name == "nt":
        os.system("cls")                                         # noqa: S605,S607  (a fixed command, no input)
    else:
        sys.stdout.write("\x1b[3J\x1b[2J\x1b[H")
        sys.stdout.flush()


def pin(height: int | None = None) -> None:
    """Keep the header still: only the lines under it scroll (a scroll region). Called again before each input, so a resized window
    keeps it."""
    import shutil
    import sys
    rows = _PINNED["rows"] if height is None else height
    if not rows:
        return
    total = shutil.get_terminal_size((120, 40)).lines
    if total <= rows + 6:                                         # too small to keep a header: everything scrolls
        sys.stdout.write("\x1b[r")
    else:
        sys.stdout.write(f"\x1b[{rows + 1};{total}r\x1b[{total};1H")
    sys.stdout.flush()


def unpin() -> None:
    import sys
    if _PINNED["rows"]:
        sys.stdout.write("\x1b[r")
        sys.stdout.flush()
        _PINNED["rows"] = 0


def show_header(console: Console, cwd, version: str = "", brand: str = "cli") -> None:
    """After the first screen: the window is cleared and Pal's own header is printed and pinned at the top, so the work carries on inside
    Pal, under its name, and never under the commands typed in the terminal before it."""
    from pal.terminal.logo import banner, rich_style
    clear_screen(console)
    lines = banner("word", brand)
    width = max(sum(len(c) for c, _ in line) for line in lines)
    pad = max(0, (console.width - width) // 2)
    for line in lines:
        t = Text(" " * pad)
        for chars, style in line:
            t.append(chars, style=rich_style(style) or None)
        console.print(t, overflow="crop", no_wrap=True)
    console.print(Text(f"{short_path(cwd)}  \u00B7  pal {version}".center(console.width), style=theme.MUTED), overflow="crop", no_wrap=True)
    console.print()


def show_user(console: Console, text: str) -> None:
    out = Text()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        out.append("▌ ", style=theme.RED)
        out.append(line, style=f"bold {theme.WHITE}")
        out.append("\n" if i < len(lines) - 1 else "")
    say(console, out)
    console.print()


def show_reply(console: Console, turn: Turn, session: PalSession) -> None:
    for w in turn.warnings:
        say(console, Text(clean_terminal(w), style=theme.MUTED))
    if turn.text:
        grid = Table.grid(padding=0)
        grid.add_column(width=2)
        grid.add_column(ratio=1)
        grid.add_row(Text("✻", style=theme.RED), Markdown(clean_terminal(turn.text), code_theme="ansi_dark"))
        say(console, grid)
    for path, why in turn.refused:
        say(console, Text(f"  refused {path}: {why}", style=theme.RED_HI))
    used = turn.prompt_tokens + turn.completion_tokens
    saved = turn.plain_estimate - used
    line = Text("  \u00BB ", style=theme.MUTED)                # "»": in every font (an arrow like ↳ is not)
    line.append(f"{used:,} used" + (f" (+{turn.helper_tokens:,} helper)" if turn.helper_tokens else ""), style=theme.MUTED)
    line.append(" \u00B7 ", style=theme.MUTED)
    if saved > 0:
        line.append(f"saved ~{saved:,}", style=f"bold {theme.RED_HI}")
    else:
        line.append("no saving here", style=theme.MUTED)
    if session.saved > 0 and session.turns > 1:
        line.append(f" \u00B7 total ~{session.saved:,} ({session.saved_pct:g}%)", style=theme.WHITE)
    line.append(f" \u00B7 {turn.seconds:.1f}s" + ("  \u00B7  /tokens" if session.turns <= 2 else ""), style=theme.MUTED)
    say(console, line)
    console.print()


def bar(console: Console, label: str, ok: bool = True) -> None:
    """A heading in a coloured bar, for what a build did: " 🏗  Built " (red), or " 🏗  Build incomplete " (grey)."""
    style = f"bold {theme.WHITE} on {theme.RED}" if ok else f"bold {theme.WHITE} on #444444"
    say(console, Text(f" {label} ", style=style))               # no emoji in the bar: some Windows fonts draw it as a box


def file_row(console: Console, paths: list[str], mark: str = "\u2705") -> None:
    """📄 a.py ✅ | 📄 b.css ✅ ... wrapped to the window."""
    width, _ = metrics(console.width)
    lines: list[Text] = []
    cur = Text()
    for path in paths:
        block = f"\U0001F4C4 {path} {mark}"
        if cur.plain and len(cur.plain) + 3 + len(block) > width:
            lines.append(cur)
            cur = Text()
        if cur.plain:
            cur.append(" | ", style=theme.MUTED)
        cur.append(block, style=theme.WHITE)
    if cur.plain:
        lines.append(cur)
    for line in lines:
        say(console, line)


def show_changes(console: Console, changes: list[Change], limit: int = 60) -> None:
    """What a build made, like a build report: the bar, then every file with its mark (no code: /diff shows it)."""
    if not changes:
        return
    bar(console, "Built")
    file_row(console, [c.path for c in changes])
    found = [f for c in changes for f in c.findings]
    if found:
        say(console, Text("  Security check on these changes", style=f"bold {theme.WHITE}"))
        show_findings(console, found, indent=4)
    console.print()


SEV_STYLE = {"critical": f"bold {theme.WHITE} on {theme.RED}", "high": f"bold {theme.RED_HI}", "medium": theme.WHITE, "low": theme.MUTED}


def _flow_commands() -> list[tuple[str, str]]:
    from pal.terminal.flow_cli import FLOW_COMMANDS
    return FLOW_COMMANDS


def show_findings(console: Console, findings: list[Finding], indent: int = 0, limit: int = 40) -> None:
    pad = " " * indent
    for f in sorted(findings, key=lambda x: SEVERITIES.index(x.severity))[:limit]:
        where = f"{f.path}:{f.line}" if f.line else f.path
        say(console, Text.assemble((pad, ""), (f" {f.severity.upper()} ", SEV_STYLE[f.severity]), ("  ", ""), (f.title, f"bold {theme.WHITE}"), (f"  {where}", theme.MUTED)))
        say(console, Padding(Text(f.why, style=theme.INK), (0, 0, 0, indent + 11)))
    if len(findings) > limit:
        say(console, Text(f"{pad}… and {len(findings) - limit} more", style=theme.MUTED))


def select(console: Console, message: str, options: list[str], default: int = 0) -> str | None:
    """A question with choices, picked with the arrows and Enter (or the number):

        ? What do you want to do?
        > Debug and retry once
          Roll back the changes

    Without a terminal (pipes, tests) the default choice is taken. None when the person leaves (Esc, Ctrl+C)."""
    import sys
    if not options:
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return options[default]
    from prompt_toolkit.key_binding import KeyBindings as _KB
    state = {"sel": default, "out": None}
    pad = " " * (metrics(console.width)[1] + 2)

    def render():
        rows = [("class:ask", f"{pad}? "), ("class:askq", message + "\n")]
        for i, opt in enumerate(options):
            on = i == state["sel"]
            rows += [("class:ok" if on else "class:cmd", f"{pad}{'> ' if on else '  '}{opt}\n")]
        return FormattedText(rows[:-1] + [(rows[-1][0], rows[-1][1].rstrip("\n"))])
    kb = _KB()

    @kb.add("up")
    def _(e):
        state["sel"] = (state["sel"] - 1) % len(options)

    @kb.add("down")
    def _(e):
        state["sel"] = (state["sel"] + 1) % len(options)

    @kb.add("enter")
    def _(e):
        state["out"] = options[state["sel"]]
        e.app.exit()

    for n in range(1, min(9, len(options)) + 1):
        @kb.add(str(n))
        def _(e, _n=n):
            state["out"] = options[_n - 1]
            e.app.exit()

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(e):
        e.app.exit()
    app: Application = Application(layout=Layout(HSplit([Window(FormattedTextControl(render), height=len(options) + 1)])), key_bindings=kb,
                                   style=STYLE, full_screen=False, erase_when_done=True)
    try:
        app.run()
    except (EOFError, KeyboardInterrupt):
        return None
    if state["out"] is not None:
        say(console, Text.assemble(("? ", f"bold {theme.RED_HI}"), (message + " ", f"bold {theme.WHITE}"), (state["out"], theme.RED_HI)))
    return state["out"]


def confirm(console: Console, question: str) -> bool:
    """A yes/no question as a choice: No is the default (Enter on its own keeps things as they are)."""
    return select(console, question, ["Yes", "No"], default=1) == "Yes"


def show_help(console: Console) -> None:
    grid = Table.grid(padding=(0, 2))
    for name, desc in all_commands():
        grid.add_row(Text(name, style=f"bold {theme.WHITE}"), Text(desc, style=theme.MUTED))
    say(console, Group(grid, Text(""), Text("tab switches agent (Build / Plan / Ask) · @file adds a file · \\ then Enter (or Alt+Enter) for a new line · Ctrl+D exits", style=theme.MUTED)))
    console.print()


LABELS = {
    "input": "Trimmed filler from your message",
    "history": "Left out old turns this request did not need",
    "compression": "Shortened old replies (names, numbers and code signatures kept)",
    "documents": "Sent only the relevant parts of indexed files",
    "code": "Sent only the relevant code from the project map",
    "files": "Did not resend @files the model already had",
    "carried": "Kept earlier @files out of the conversation history",
}
SHORT = {"input": "message trimmed", "history": "old turns left out", "compression": "old replies shortened", "documents": "only relevant file parts",
         "code": "only relevant code", "files": "@files not resent", "carried": "@files kept out of history"}
HELP_TOKENS = """How Pal counts tokens
  used     what the main model really read and wrote (its own counts when it reports them, else characters / 4)
  helper   Pal's small local jobs (trimming, checking): an extra cost, shown on its own
  plain    an estimate of what a plain chat would have used for the same request: it resends the whole
           conversation and pastes every @file again each time
  saved    plain minus used. Every saving here is an estimate, not a measurement.

Where savings come from
""" + "\n".join(f"  {v}" for v in LABELS.values()) + """
  Other    the part of the gap no stage reports by itself
  Pal's own additions (negative)   the answer rules and notices Pal adds to each request: why tiny chats can cost more

Commands:  /tokens   /tokens last   /tokens turns   /tokens help"""


def _bars(console: Console, saved_by: dict, gap: int) -> None:
    """One row per stage, with a bar scaled to the largest saving, plus what no stage explains."""
    rows = sorted(((LABELS.get(k, k), v) for k, v in saved_by.items() if v > 0), key=lambda r: -r[1])
    rest = gap - sum(v for _, v in rows)
    if rest > 0:
        rows.append(("Other (not itemized)", rest))
    top = max([v for _, v in rows] + [1])
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=24, no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(ratio=1)
    for label, v in rows:
        grid.add_row(Text("\u2588" * max(1, round(22 * v / top)), style=theme.RED), Text(f"{v:,}", style=f"bold {theme.WHITE}"), Text(label, style=theme.INK))
    if rest < 0:
        grid.add_row(Text(""), Text(f"{rest:,}", style=theme.MUTED), Text("Pal's own additions (answer rules and notices)", style=theme.MUTED))
    if not rows and rest >= 0:
        grid.add_row(Text(""), Text("0", style=theme.MUTED), Text("nothing saved yet", style=theme.MUTED))
    say(console, grid)


def _head(used: int, helper: int, plain: int) -> Table:
    saved = plain - used
    pct = round(100.0 * saved / plain, 1) if plain else 0.0
    grid = Table.grid(padding=(0, 3))
    for k, v, style in (("used", f"{used:,}" + (f"  (+{helper:,} helper)" if helper else ""), theme.WHITE), ("plain chat (est.)", f"{plain:,}", theme.INK),
                        ("saved", f"{saved:,}  ({pct:g}%)" if saved > 0 else f"{saved:,}  (Pal used more here)", f"bold {theme.RED_HI}" if saved > 0 else theme.MUTED)):
        grid.add_row(Text(k, style=theme.MUTED), Text(v, style=style))
    return grid


def show_tokens(console: Console, session: PalSession, arg: str = "") -> None:
    """/tokens: the session; /tokens last: the last request; /tokens turns: every request; /tokens help: how it is counted."""
    arg = arg.strip().lower()
    if arg == "help":
        say(console, Text(HELP_TOKENS, style=theme.INK))
    elif not session.records:
        say(console, Text("No requests yet. Ask Pal something, then type /tokens.", style=theme.MUTED))
    elif arg == "last":
        r = session.records[-1]
        say(console, Text.assemble(("Last request  ", f"bold {theme.WHITE}"), (r.text[:70] + ("\u2026" if len(r.text) > 70 else ""), theme.MUTED)))
        say(console, _head(r.used, r.helper, r.plain))
        console.print()
        _bars(console, r.saved_by, r.saved)
    elif arg == "turns":
        grid = Table.grid(padding=(0, 2))
        for width, just in ((3, "right"), (26, "left"), (8, "right"), (8, "right"), (8, "right"), (24, "left")):
            grid.add_column(width=width, justify=just, no_wrap=True, overflow="ellipsis")
        grid.add_row(*(Text(h, style=theme.MUTED) for h in ("#", "request", "used", "plain", "saved", "biggest saving")))
        for i, r in enumerate(session.records, 1):
            best = max(r.saved_by.items(), key=lambda kv: kv[1])[0] if r.saved_by else ""
            grid.add_row(Text(str(i), style=theme.MUTED), Text(r.text.replace("\n", " "), style=theme.INK), Text(f"{r.used:,}", style=theme.WHITE),
                         Text(f"{r.plain:,}", style=theme.INK), Text(f"{r.saved:,}", style=f"bold {theme.RED_HI}" if r.saved > 0 else theme.MUTED),
                         Text(SHORT.get(best, ""), style=theme.MUTED))
        say(console, grid)
    else:
        say(console, Text.assemble(("This session  ", f"bold {theme.WHITE}"), (f"{session.turns} request{'s' if session.turns != 1 else ''}, {session.seconds:.1f}s", theme.MUTED)))
        say(console, _head(session.pal_tokens, session.helper_tokens, session.plain_tokens))
        console.print()
        _bars(console, session.saved_by_total(), session.saved)
        console.print()
        say(console, Text("/tokens last  the last request \u00B7 /tokens turns  every request \u00B7 /tokens help  how it is counted", style=theme.MUTED))
    console.print()


def show_models(console: Console, session: PalSession, info: dict, hw, keyed: list[str]) -> None:
    """/model: the model in use, the installed local models with how each fits this computer, what setup recommends, and API providers with a key."""
    from pal.terminal.local_models import FIT_LABEL, installed_fit, recommend, tier_name
    say(console, Text.assemble(("Now  ", theme.MUTED), (model_label(session), f"bold {theme.WHITE}"),
                               (f"   helper: {session.helper_model}" if session.helper_model else
                                "   no local helper (no GPU: API only)" if not session.local else "   helper: the main model (run pal setup for a small local helper)", theme.MUTED)))
    say(console, Text(f"This computer: {hw.summary()}  ({tier_name(hw)})", style=theme.MUTED))
    console.print()
    rows = installed_fit(info.get("models", []), hw) if info.get("ok") else []
    if rows:
        grid = Table.grid(padding=(0, 2))
        for r in sorted(rows, key=lambda r: -r["rank"]):
            grid.add_row(Text(r["name"], style=f"bold {theme.WHITE}" if r["name"] == session.model else theme.INK), Text(f"{r['need_gb']:g} GB", style=theme.MUTED),
                         Text(FIT_LABEL[r["fit"]], style=f"bold {theme.RED_HI}" if r["fit"] == "gpu" else theme.MUTED))
        say(console, Text("Installed", style=f"bold {theme.WHITE}"))
        say(console, Padding(grid, (0, 0, 0, 2)))
    else:
        say(console, Text("No local models found (is Ollama running?).", style=theme.MUTED))
    rec = recommend(hw)
    names = {r["name"] for r in rows}
    if rec is None:
        say(console, Text("This computer has no GPU, so Pal uses API models only.", style=theme.INK))
    elif rec.main.tag not in names:
        say(console, Text(f"Best for this computer: {rec.main.tag} ({rec.main.size_gb:g} GB). Get it with  ollama pull {rec.main.tag}  or  pal setup", style=theme.INK))
    say(console, Text(f"API providers with a key: {', '.join(keyed) or 'none (add one with /key <provider> <key>)'}", style=theme.MUTED))
    say(console, Text("Switch with  /model <name>  or  /model openrouter:<name>   \u00B7   every local model by GPU:  pal setup --list", style=theme.MUTED))
    console.print()


def all_commands() -> list[tuple[str, str]]:
    from pal.terminal.commands import COMMANDS as MORE
    seen: set[str] = set()
    out = []
    for name, desc in COMMANDS[:-1] + MORE + _flow_commands() + COMMANDS[-1:]:
        if name not in seen:
            seen.add(name)
            out.append((name, desc))
    return out


CMD_PREFIX = "/"                                   # what starts a command in the box ("\\" in pal cli's REPL)
MENU_SOURCE: Callable[[], list[tuple[str, str]]] | None = None     # the commands the box suggests (default: the ones below)


def matching(text: str) -> list[tuple[str, str]]:
    items = MENU_SOURCE() if MENU_SOURCE is not None else all_commands()
    if CMD_PREFIX == "\\":
        return [(n, d) for n, d in items if text.startswith(CMD_PREFIX) and n.startswith(text)][:10]
    return [(n, d) for n, d in items if text.startswith(CMD_PREFIX) and " " not in text and n.startswith(text)][:10]


def model_label(session: PalSession) -> str:
    return session.model + ("  local" if session.provider == "local" else f"  {session.provider}")


def brand_block() -> list[tuple[str, str]]:
    """The logo with the wordmark beside it, centred on the logo's height (kept for wide screens and the tests)."""
    art, word = logo_lines(), theme.wordmark()
    top = (len(art) - len(word)) // 2
    width = sum(len(c) for c, _ in word[0])
    frags: list[tuple[str, str]] = []
    for i, line in enumerate(art):
        frags += [(ptk_style(st), ch) for ch, st in line]
        frags.append(("", "    "))
        w = word[i - top] if 0 <= i - top < len(word) else [(" " * width, "")]
        frags += [(f"fg:{colour}" if colour else "", chars) for chars, colour in w]
        if i < len(art) - 1:
            frags.append(("", "\n"))
    return frags


LOGO_ROWS = 21          # terminal rows for the full logo above the box (logo 11 + box 5 + hints + footer + gaps)
SMALL_LOGO_ROWS = 18    # rows for the smaller logo; below that the "pal code" wordmark, which always shows
BOX_WIDTH = 100         # the first screen's input box is at most this wide, centred under the logo


def dot_logo(size: str = "full", kind: str = "cli") -> list[tuple[str, str]]:
    """The start screen's mark (the logo and the "pal code" wordmark) as prompt_toolkit fragments."""
    frags: list[tuple[str, str]] = []
    lines = banner(size, kind)
    for i, runs in enumerate(lines):
        frags += [(ptk_style(style), chars) for chars, style in runs]
        if i < len(lines) - 1:
            frags.append(("", "\n"))
    return frags


def left_line(session: PalSession) -> list[tuple[str, str]]:
    """How many tokens are left: in the model's context window, and of the session's cap when /budget set one."""
    from pal.context_window import short
    try:
        left, window = session.tokens_left()
    except Exception:  # noqa: BLE001  (a session without a model yet)
        return []
    out = [("class:model", f"{short(left)}"), ("class:hint", f" of {short(window)} tokens left")]
    cap = session.budget_left() if hasattr(session, "budget_left") else None
    if cap is not None:
        out += [("class:hint", "  \u00B7  budget "), ("class:warn" if cap < session.budget * 0.1 else "class:model", f"{cap:,}"), ("class:hint", " left")]
    return out


def clickable_agent_line(session: PalSession, click) -> list:
    """The agent and the model under the input box; clicking either opens its dropdown."""
    frags = agent_line(session)
    out = []
    for frag in frags:
        if frag[0] == "class:model":
            out.append(("class:model class:link", frag[1] + " \u25BE", click("\x00model\x00")))
        elif frag[0].startswith("class:mode.") and getattr(session, "brand", "") == "bench":
            out.append(("class:mode.build", "Bench"))                       # the bench always builds the task: no agent to choose
        elif frag[0].startswith("class:mode."):
            out.append((frag[0] + " class:link", frag[1] + " \u25BE", click("\x00agent\x00")))
        else:
            out.append(frag)
    return out


def agent_line(session: PalSession) -> list[tuple[str, str]]:
    """The agent and the model (no provider in brackets, no plan names); a note only when files are waiting to be applied."""
    _provider, model = session.model_for() if hasattr(session, "model_for") else (session.provider, session.model)
    note = getattr(session, "status_note", None)
    pending = note() if callable(note) else ""
    return [("class:mode." + session.mode, session.mode.capitalize()), ("class:hint", "  \u00B7  "), ("class:model", model),
            ("class:hint", "  \u00B7  no GPU: API only" if not getattr(session, "local", True) else ""),
            ("class:ok", f"  \u00B7  {pending}" if pending else "")]


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def project_line(session: PalSession) -> list[tuple[str, str]]:
    """One line about the prepared folder, for the first screen: files, the map, documents, the graph, the local model."""
    project = getattr(session, "project", None)
    if project is None:
        return []
    r = project.report
    bits = [f"{r.code_files} files"]
    if r.map_tokens:
        bits.append(f"map {_k(r.map_tokens)} tokens")
    if r.converted:
        bits.append(f"{len(r.converted)} doc{'s' if len(r.converted) != 1 else ''} → markdown")
    graph = r.graph.replace("ready: ", "graph ") if r.graph.startswith("ready") else ("graph building…" if r.graph == "building" else "")
    if graph:
        bits.append(graph)
    local = getattr(session, "local_status", "")
    out = [("class:hint", "this folder: "), ("class:ok", "  ·  ".join(bits))]
    if local and not local.startswith(("Ollama running", "no GPU")):    # the models are on the line above: only a problem is worth showing here
        out += [("class:hint", "  ·  "), ("class:warn", local)]
    return out


def show_project(console: Console, session: PalSession) -> None:
    """/project, and the summary printed while the folder is set up."""
    project = getattr(session, "project", None)
    if project is None:
        say(console, Text("This session has no prepared folder.", style=theme.MUTED))
        return
    r = project.report

    def row(ok: bool | None, name: str, text: str, note: str = "") -> None:
        mark = ("✓", theme.RED_HI) if ok else ("…", theme.MUTED) if ok is None else ("✗", theme.MUTED)
        say(console, Text.assemble((f"{mark[0]} ", mark[1]), (f"{name:<13}", f"bold {theme.WHITE}"), (text, theme.INK), (f"   {note}" if note else "", theme.MUTED)))
    local = getattr(session, "local_status", "")
    if local:
        row(not local.startswith(("Ollama is not", "Ollama did not", "no GPU")) or None, "Local LLM", local)
    row(bool(r.code_files), "Project map", f"{r.code_files} files, {_k(r.map_tokens)} tokens" if r.code_files else "no source files here",
        f"(the code itself is about {_k(r.source_tokens)})" if r.source_tokens else "")
    g = r.graph
    row(True if g.startswith("ready") else None if g == "building" else False, "Code graph",
        g.replace("ready: ", "") + (" (graphify, no model involved)" if g.startswith("ready") else ""),
        "built in the background; the session can start now" if g == "building" else "pip install graphifyy" if g == "needs graphify" else "")
    conv = r.converted
    text_docs = [d for d in r.docs if not d.converted and not d.error]
    failed = [d for d in r.docs if d.error]
    if r.docs:
        row(True, "Documents", (f"{len(conv)} converted to markdown" if conv else "") + (", " if conv and text_docs else "")
            + (f"{len(text_docs)} text file{'s' if len(text_docs) != 1 else ''}" if text_docs else ""), "indexed: only matching passages are sent")
        for d in conv[:6]:
            say(console, Text(f"                {d.path} → {short_path(Path(d.markdown))}", style=theme.MUTED))
        for d in failed[:4]:
            say(console, Text(f"                {d.path}: {d.error}", style=theme.MUTED))
    else:
        row(False, "Documents", "none in this folder", "PDF, Word, PowerPoint and Excel files are converted to markdown")
    if r.tests:
        row(True, "Tests", r.tests)


def read_input(session: PalSession, console: Console, save_mode: Callable[..., None] | None = None, landing: bool = False, version: str = "",
               history=None, initial: str = "", on_shift_tab: Callable[[], str] | None = None,
               models: Callable[[], list[tuple[str, str, str]] | None] | None = None, open_models: bool = False,
               start_listening: bool = False) -> str | None:
    """Show the input box and return what was typed, or None when the person leaves (Ctrl+Q, or Ctrl+D on an empty box).

    `landing` draws the whole first screen full-screen: the folder at the top left, the dot-matrix logo, what Pal is set to, a full-width input
    box and the key hints. Otherwise a status line, the box and the hints appear under the conversation and disappear on Enter."""
    from prompt_toolkit.application import get_app
    from prompt_toolkit.widgets import Frame
    buf = Buffer(multiline=True)
    if initial:
        buf.text = initial
        buf.cursor_position = len(initial)
    empty = Condition(lambda: not buf.text)
    result: dict[str, str | None] = {"text": None}
    from pal.terminal.history import Browser
    browse = Browser(history) if history is not None else None
    ui_state: dict = {"sel": 0, "search": None, "flash": "", "recalled": False, "setting": False}   # menu, Ctrl+R, a note, text from history
    menu_open = Condition(lambda: bool(matching(buf.text)) and not ui_state["recalled"])

    def size():
        try:
            return get_app().output.get_size()
        except Exception:  # noqa: BLE001  (no app yet)
            return None

    def status_left():
        files = len(session.pending)
        if ui_state["search"] is not None:
            return FormattedText([("class:ok", "reverse search: "), ("class:model", ui_state["search"] or "…"), ("class:hint", "   ctrl+r older · enter use · esc esc clear")])
        if ui_state["flash"]:
            return FormattedText([("class:ok", ui_state["flash"])])
        return FormattedText(agent_line(session) + ([("class:ok", f"   {files} file{'s' if files != 1 else ''} attached")] if files else []))

    def status_right():
        if mic["state"] == "listening":
            return FormattedText([("class:warn", "listening "), ("class:wave", wave()), ("class:hint", "  enter to stop")])
        if mic["state"] == "working":
            return FormattedText([("class:hint", "writing it down\u2026")])
        return FormattedText([])

    def menu():
        rows: list[tuple[str, str]] = []
        for i, (n, d) in enumerate(matching(buf.text)):
            on = i == min(ui_state["sel"], len(matching(buf.text)) - 1)
            rows += [("class:ok" if on else "class:cmd", f"{'> ' if on else '  '}{n:<14}"), ("class:cmddesc", d + "\n")]
        if rows:
            rows[-1] = (rows[-1][0], rows[-1][1].rstrip("\n"))
        return FormattedText(rows)

    def keys():
        if ui_state["flash"] and landing:                               # a short note (the mic, for example) where the hints are
            return FormattedText([("class:warn", ui_state["flash"])])
        out: list[tuple[str, str]] = []
        for k, label in (("tab", "agent"), (CMD_PREFIX, "commands")):          # the rest are listed by ? and /shortcuts
            out += [("class:hintkey", k), ("class:hint", f" {label}   ")]
        out[-1] = (out[-1][0], out[-1][1].rstrip())
        return FormattedText(out)

    kb = KeyBindings()

    def pick() -> str | None:
        m = matching(buf.text)
        return m[min(ui_state["sel"], len(m) - 1)][0] if m else None

    @kb.add("enter")
    def _(event):
        ui_state["search"] = None
        chosen = pick() if not ui_state["recalled"] else None
        if chosen and buf.text != chosen and buf.text.strip() != chosen:      # Enter on the command menu takes the highlighted command
            buf.text = chosen + " "
            buf.cursor_position = len(buf.text)
            ui_state["sel"] = 0
            return
        if buf.text.endswith("\\") and buf.text.strip() != "\\":
            buf.delete_before_cursor(1)
            buf.insert_text("\n")
            return
        if buf.text.strip():
            result["text"] = buf.text
            event.app.exit()

    @kb.add("escape", "enter")
    @kb.add("c-j")
    def _(event):
        buf.insert_text("\n")

    @kb.add("tab")
    def _(event):
        chosen = pick()
        if chosen:
            buf.text = chosen + " "
            buf.cursor_position = len(buf.text)
            ui_state["sel"] = 0
        else:
            session.toggle_mode()
            if save_mode:
                save_mode(mode=session.mode)

    @kb.add("up", filter=menu_open)
    def _(event):
        ui_state["sel"] = max(0, ui_state["sel"] - 1)

    @kb.add("down", filter=menu_open)
    def _(event):
        ui_state["sel"] = min(len(matching(buf.text)) - 1, ui_state["sel"] + 1)

    def set_text(text: str) -> None:
        ui_state["setting"] = True
        buf.text = text
        buf.cursor_position = len(text)
        ui_state.update(setting=False, recalled=True)            # a recalled line does not open the command menu

    @kb.add("up", filter=~menu_open)
    def _(event):
        if buf.document.cursor_position_row > 0 or browse is None:     # inside a multi-line message: move up a line
            buf.cursor_up()
            return
        older = browse.older(buf.text)
        if older is not None:
            set_text(older)

    @kb.add("down", filter=~menu_open)
    def _(event):
        if buf.document.cursor_position_row < buf.document.line_count - 1 or browse is None:
            buf.cursor_down()
            return
        newer = browse.newer()
        if newer is not None:
            set_text(newer)

    @kb.add("escape", "escape")
    def _(event):
        buf.reset()
        ui_state.update(search=None, sel=0)

    @kb.add("c-r")
    def _(event):
        if browse is None:
            return
        if ui_state["search"] is None:
            ui_state["search"] = buf.text
            browse.pos = len(browse.h.items)
        found = browse.find(ui_state["search"])
        if found is not None:
            set_text(found)
        else:
            ui_state["flash"] = f"no older match for {ui_state['search']!r}"

    @kb.add("s-tab")
    def _(event):
        if on_shift_tab:
            ui_state["flash"] = on_shift_tab()

    @kb.add("c-l")
    def _(event):
        event.app.renderer.clear()

    @kb.add("?", filter=empty)
    def _(event):
        result["text"] = "/shortcuts" if CMD_PREFIX == "/" else "\\help"
        event.app.exit()

    # -- the mic, in the box: tap it to start, again (or Enter) to stop; the words are added to the box -----------------
    import threading as _threading
    mic: dict = {"state": "", "stop": None, "app": None, "levels": [0.0] * 16, "drawn": 0.0}   # state: "" | "listening" | "working"
    BARS = "._\u2584\u2588"                                         # only characters every font has (the logo uses the same blocks)

    def wave() -> str:
        """The voice graph: the last moments of sound as bars, like a level meter."""
        return "".join(BARS[min(len(BARS) - 1, int(v * (len(BARS) - 1) + 0.5))] for v in mic["levels"])

    def heard(level: float) -> None:
        import time as _time
        mic["levels"] = (mic["levels"] + [level])[-16:]
        now = _time.monotonic()
        if now - mic["drawn"] > 0.06:                                   # about 15 frames a second
            mic["drawn"] = now
            refresh()
    listening = Condition(lambda: mic["state"] == "listening")

    def refresh() -> None:
        if mic["app"] is not None:
            mic["app"].invalidate()

    def mic_toggle() -> None:
        if mic["state"] == "listening":
            mic["stop"].set()
            return
        if mic["state"] == "working":
            return
        from pal.terminal import voice
        if not voice.can_record():
            ui_state["flash"] = "The mic is not available here (no microphone found). Windows voice typing, Win+H, types into the box too."
            refresh()
            return
        stop = _threading.Event()
        mic.update(state="listening", stop=stop, levels=[0.0] * 16)

        def run() -> None:
            try:
                audio = voice.record(stop, on_level=heard)
                mic["state"] = "working"
                refresh()
                words = voice.transcribe(audio).text
            except Exception as e:  # noqa: BLE001  (no microphone, no speech-to-text: say so in the box, never crash)
                words = ""
                ui_state["flash"] = str(e)[:140]
            mic["state"] = ""
            if words:
                gap = " " if buf.text and not buf.text.endswith((" ", "\n")) else ""
                buf.insert_text(gap + words)
            refresh()
        _threading.Thread(target=run, daemon=True, name="pal-mic").start()
        refresh()

    def mic_label():
        state = mic["state"]
        if state == "listening":
            return [("class:mic", " \u25A0 stop ", click_mic), ("class:hint", "  ")]
        if state == "working":
            return [("class:mic", " \U0001F3A4 ", click_mic), ("class:hint", " writing it down\u2026  ")]
        return [("class:mic", " \U0001F3A4 ", click_mic), ("class:hint", "  ")]

    def click_mic(event):
        from prompt_toolkit.mouse_events import MouseEventType
        if event.event_type != MouseEventType.MOUSE_UP:
            return NotImplemented
        mic_toggle()
        return None

    @kb.add("enter", filter=listening, eager=True)
    def _(event):
        mic_toggle()                                                   # Enter while listening: stop and write it down (nothing is sent)

    buf.on_text_changed += lambda _b: None if ui_state["setting"] else ui_state.update(sel=0, flash="", recalled=False)

    @kb.add("c-p")
    def _(event):
        if not buf.text:
            buf.insert_text(CMD_PREFIX)

    @kb.add("c-c")
    def _(event):
        if buf.text:
            buf.reset()
        else:
            event.app.exit()

    @kb.add("c-q")
    def _(event):
        event.app.exit()

    @kb.add("c-d")
    def _(event):
        if not buf.text:
            event.app.exit()

    # -- the model dropdown, in the box: click the model, Ctrl+O or /model; arrows and Enter (or a click) choose, Esc closes ------------------
    first_kind = open_models if isinstance(open_models, str) and open_models else "model"      # open_models: True / "model" / "agent"
    drop = {"open": bool(open_models) and (first_kind == "agent" or bool(models)), "sel": 0, "kind": first_kind}
    models_open = Condition(lambda: drop["open"])
    AGENT_ITEMS = [("Build", "writes and edits files", "build"),
                   ("Plan", "plans the steps, no edits", "plan"),
                   ("Ask", "answers questions, no edits", "ask")]

    def choices() -> list[tuple[str, str, str]] | None:
        if drop["kind"] == "agent":
            return AGENT_ITEMS
        return models() if models else None

    def choose_item(index: int) -> None:
        items = choices() or []
        if not 0 <= index < len(items):
            return
        if drop["kind"] == "agent":                                     # the agent changes right here, the box stays open
            session.mode = items[index][2]
            if save_mode:
                save_mode(mode=session.mode)
            drop["open"] = False
            return
        result["text"] = "\x00pick\x00" + items[index][2] + "\x00" + buf.text
        get_app().exit()

    def open_drop(kind: str = "model") -> None:
        if kind == "agent" or models:
            current = session.mode if kind == "agent" else ""
            sel = next((i for i, it in enumerate(AGENT_ITEMS) if it[2] == current), 0) if kind == "agent" else 0
            drop.update(open=True, sel=sel, kind=kind)

    def dropdown():
        items = choices()
        if items is None:
            return FormattedText([("class:hint", "   looking for models\u2026")])
        rows: list = []
        current = (session.mode.capitalize() if drop["kind"] == "agent" else
                   (session.model_for()[1] if hasattr(session, "model_for") else ""))
        width = min(30, max((len(it[0]) for it in items), default=10) + 2)
        for i, (label, about, _spec) in enumerate(items):
            on = i == drop["sel"]

            def clicked(event, _i=i):
                from prompt_toolkit.mouse_events import MouseEventType
                if event.event_type != MouseEventType.MOUSE_UP:
                    return NotImplemented
                choose_item(_i)
                return None
            mark = "\u2713 " if label == current else "  "
            arrow = ">" if on else " "                                   # plain ">": every terminal font has it
            star = "recommended" in about
            rows += [("class:dd.on" if on else "class:dd", f" {arrow} {mark}{label:<{width}}", clicked),
                     ("class:dd.star" if star else "class:dd.desc", f"{about}\n", clicked)]
        rows.append(("class:hint", "   \u2191\u2193 choose  \u00B7  enter select  \u00B7  esc close"))
        return FormattedText(rows)

    models_box = ConditionalContainer(Window(FormattedTextControl(dropdown), height=Dimension(min=1, max=12)), filter=models_open)

    @kb.add("c-o")
    def _(event):
        if drop["open"]:
            drop["open"] = False
        else:
            open_drop()

    @kb.add("up", filter=models_open, eager=True)
    def _(event):
        drop["sel"] = max(0, drop["sel"] - 1)

    @kb.add("down", filter=models_open, eager=True)
    def _(event):
        drop["sel"] = min(len(choices() or [1]) - 1, drop["sel"] + 1)

    @kb.add("enter", filter=models_open, eager=True)
    def _(event):
        choose_item(drop["sel"])

    @kb.add("escape", filter=models_open, eager=True)
    def _(event):
        drop["open"] = False

    input_win = Window(BufferControl(buf, input_processors=[ConditionalProcessor(AfterInput(lambda: PLACEHOLDERS["bench"] if getattr(session, "brand", "") == "bench"
                                                                                            else PLACEHOLDERS.get(session.mode, PLACEHOLDER), style="class:placeholder"),
                                                                                 filter=empty & ~listening),
                                                            ConditionalProcessor(AfterInput(lambda: [("class:warn", "Listening\u2026 "), ("class:wave", wave()),
                                                                                                     ("class:hint", "   enter or \u25A0 to stop")]),
                                                                                 filter=empty & listening)]),
                       height=lambda: Dimension.exact(min(8, max(2 if landing else 1, buf.text.count("\n") + (2 if landing else 1)))), wrap_lines=True)
    prompt = Window(FormattedTextControl([("class:prompt", " > ")]), width=3, dont_extend_width=True)
    box = Frame(VSplit([prompt, input_win, Window(width=1)]), style="class:frame")
    hints = Window(FormattedTextControl(lambda: FormattedText(list(keys()) + [("", "  ")])), height=1, align=WindowAlign.RIGHT)  # under the box, right
    menu_box = ConditionalContainer(Window(FormattedTextControl(menu), height=Dimension(min=1, max=12)), filter=menu_open)

    def margins(container):
        return VSplit([Window(width=1), container, Window(width=1)])
    if not landing:
        # After the first screen: no box and nothing stretched to the window's edge, so a resized window never leaves broken lines behind.
        # A line with the agent, the model and the keys, then a single prompt line.
        def status_line():
            return FormattedText([("", "  ")] + list(status_left()) + [("class:hint", "   \u00B7   ")] + list(keys()) + list(status_right()))
        status = Window(FormattedTextControl(status_line), height=1, wrap_lines=False)
        line = VSplit([Window(width=1), prompt, input_win])
        root = HSplit([Window(height=1), status, line, VSplit([Window(width=2), models_box]), VSplit([Window(width=2), menu_box])])
    else:
        from prompt_toolkit.mouse_events import MouseEventType

        def click(action: str):
            """A mouse click on a part of the first screen: leave the box with an action the caller carries out (the typed text is kept)."""
            def handler(event):
                if event.event_type != MouseEventType.MOUSE_UP:
                    return NotImplemented
                if action in ("\x00model\x00", "\x00agent\x00"):       # the model or the agent: its dropdown opens under the box
                    kind = "agent" if action == "\x00agent\x00" else "model"
                    if drop["open"] and drop["kind"] == kind:
                        drop["open"] = False
                    else:
                        open_drop(kind)
                    return None
                result["text"] = action + buf.text
                get_app().exit()
                return None
            return handler

        rows = lambda: size().rows if size() else 30                             # noqa: E731
        cols = lambda: size().columns if size() else 120                         # noqa: E731
        brand = getattr(session, "brand", "cli")
        marks = {"full": dot_logo("full", brand), "small": dot_logo("small", brand), "word": dot_logo("word", brand)}
        wide = {name: max(sum(len(c) for c, _ in line) for line in banner(name, brand)) for name in marks}

        def which() -> str:
            """The biggest mark that fits with the box under it; in a small window still the "pal code" wordmark (never nothing)."""
            r, c = rows(), cols()
            if r >= LOGO_ROWS and c >= wide["full"] + 4:
                return "full"
            if r >= SMALL_LOGO_ROWS and c >= wide["small"] + 4:
                return "small"
            return "word"

        roomy = Condition(lambda: rows() >= 16)                                  # tiny windows drop the blank lines and the footer, not the logo

        def gap():
            return ConditionalContainer(Window(height=1), filter=roomy)

        def mark(name: str):
            return ConditionalContainer(HSplit([Window(FormattedTextControl(FormattedText(marks[name])), height=BANNER_ROWS[name], align=WindowAlign.CENTER),
                                                gap()]), filter=Condition(lambda: which() == name))

        box = Frame(HSplit([VSplit([prompt, input_win, Window(width=1)]),                                  # the agent and model inside the box
                            VSplit([Window(FormattedTextControl(lambda: FormattedText([("", "   ")] + clickable_agent_line(session, click)))),
                                    Window(FormattedTextControl(lambda: FormattedText(mic_label())),
                                           dont_extend_width=True)],          # no right alignment: it would shift where clicks land
                                   height=1)]),
                    style="class:frame")

        def centred(container):
            """The box at most BOX_WIDTH wide, in the middle, like the mark above it: the two stay one group at any window size."""
            return VSplit([Window(width=Dimension(weight=1)), HSplit([container], width=Dimension(max=BOX_WIDTH, weight=8)), Window(width=Dimension(weight=1))])

        footer = VSplit([Window(FormattedTextControl(lambda: FormattedText([("class:hint", "  " + short_path(session.cwd))]))),          # the folder, bottom left
                         Window(FormattedTextControl(lambda: FormattedText([("class:hint", f"pal {version}  ")])), align=WindowAlign.RIGHT)],  # the version, bottom right
                        height=1)
        root = HSplit([Window(height=Dimension(weight=1)),
                       mark("full"), mark("small"), mark("word"),
                       centred(box),
                       centred(models_box),
                       centred(hints),
                       centred(menu_box),
                       Window(height=Dimension(weight=1)),
                       ConditionalContainer(footer, filter=Condition(lambda: rows() >= 14))], style="class:screen")
    app: Application = Application(layout=Layout(root, focused_element=input_win), key_bindings=kb, style=STYLE, full_screen=landing, erase_when_done=True, mouse_support=landing,
                               refresh_interval=1.0 if landing else None)   # the graph finishes in the background
    mic["app"] = app
    if start_listening:
        mic_toggle()
    try:
        app.run()
    finally:
        if mic["stop"] is not None:
            mic["stop"].set()                                          # leaving the box always stops the mic
    return result["text"]
