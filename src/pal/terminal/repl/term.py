"""The terminal primitives everything else is drawn with: Pal's colours as escape codes, tables, command hints, errors, the bar spinner,
single-key yes/no questions, arrow-key choice lists and relative times.

Colours: Pal is red and white. Accents (commands, names, keys) are red, headings and questions bright white, labels white on red.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import unicodedata

# -- colours (SGR parameters) ------------------------------------------------------------------------------------------------------
RED = "38;2;229;9;20"
RED_HI = "38;2;255;77;87"
WHITE = "97"
INK = "37"
MUTED = "38;2;138;138;138"
BLACK = "30"
BG_RED = "48;2;229;9;20"
BG_DARK_RED = "48;2;120;6;12"
BG_WHITE = "107"
BG_GREY = "100"
BG_BLACK = "40"
BOLD = "1"
PROMPT_FG = "38;5;251"
# Pal's red and white (the Netflix look): every accent is red, headings and questions bright white
DULL_RED = "38;2;184;50;58"
BG_DULL_RED = "48;2;184;50;58"
HI_GREEN, HI_MAGENTA, HI_RED, HI_YELLOW, HI_CYAN, HI_BLUE = RED_HI, WHITE, RED_HI, WHITE, RED_HI, RED
BG_GREEN, BG_BLUE, BG_MAGENTA, BG_CYAN, BG_RED_STD, BG_HI_BLACK = BG_DULL_RED, BG_RED, BG_DARK_RED, BG_RED, BG_GREY, BG_GREY
PINK = "38;2;229;9;20"            # the loader: Pal red

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def colours_on() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("PAL_FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def c(params: str, text: str) -> str:
    """text in the given SGR parameters, reset after."""
    if not text or not colours_on():
        return text
    return f"\x1b[{params}m{text}\x1b[0m"


def accent(text: str) -> str:
    return c(f"{HI_CYAN};{BOLD}", text)


def strong(text: str) -> str:
    return c(f"{BOLD};{WHITE}", text)


def muted(text: str) -> str:
    return c(MUTED, text)


def red(text: str) -> str:
    return c(HI_RED, text)


def label(text: str) -> str:
    """White on red: section labels, banners."""
    return c(f"{WHITE};{BG_BLUE};{BOLD}", text)


def label_white(text: str) -> str:
    """Black on white."""
    return c(f"{BG_MAGENTA};{WHITE};{BOLD}", text)


def chip(text: str) -> str:
    """A command shown as a red chip."""
    return c(f"{BOLD};{WHITE};{BG_CYAN};{WHITE}", text)


def strip(text: str) -> str:
    return _ANSI.sub("", text)


# -- output -----------------------------------------------------------------------------------------------------------------------
def enable_vt() -> None:
    """Let the Windows console draw escape codes (Windows 10 and later)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32  # type: ignore[attr-defined]
        for handle in (-11, -12):
            h = k.GetStdHandle(handle)
            mode = ctypes.c_ulong()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:  # noqa: BLE001
        pass


def out(text: str = "", end: str = "\n") -> None:
    try:
        sys.stdout.write(text + end)
    except UnicodeEncodeError:
        sys.stdout.write((text + end).encode("ascii", "replace").decode("ascii"))
    sys.stdout.flush()


def err(text: str = "", end: str = "\n") -> None:
    sys.stdout.flush()
    try:
        sys.stderr.write(text + end)
    except UnicodeEncodeError:
        sys.stderr.write((text + end).encode("ascii", "replace").decode("ascii"))
    sys.stderr.flush()


def term_width() -> int:
    raw = os.environ.get("PAL_COLUMNS")
    if raw and raw.isdigit():
        return int(raw)
    try:
        return os.get_terminal_size().columns
    except OSError:
        return int(os.environ.get("COLUMNS", "80") or 80)


def term_height() -> int:
    try:
        return os.get_terminal_size().lines
    except OSError:
        return int(os.environ.get("LINES", "24") or 24)


# -- display width -------------------------------------------------------------------------------------------------------------------
def _char_width(ch: str) -> int:
    o = ord(ch)
    if o in (0x200D, 0xFE0F, 0xFE0E) or unicodedata.combining(ch):
        return 0
    if 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF and unicodedata.east_asian_width(ch) == "W":
        return 2
    try:
        from wcwidth import wcwidth
        w = wcwidth(ch)
        return max(0, w)
    except Exception:  # noqa: BLE001
        return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def width(text: str) -> int:
    """Cells the text takes on screen (escape codes left out; emoji are two cells, joiners and variation selectors none)."""
    text = strip(text)
    total, skip_next = 0, False
    for ch in text:
        if skip_next:                       # the second half of a ZWJ pair is drawn inside the first
            skip_next = False
            if _char_width(ch) == 2:
                continue
        if ch == "‍":
            skip_next = True
            continue
        total += _char_width(ch)
    return total


def pad(text: str, n: int) -> str:
    return text + " " * max(0, n - width(text))


# -- tables ----------------------------------------------------------------------------------------------------------------------------
_NUMBER = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$")


def _title(h: str) -> str:
    h = h.replace("_", " ")
    h = re.sub(r"(?<![\d ])\.|\.(?![\d ])", " ", h)
    return h.strip().upper()


def _wrap(cell: str, limit: int) -> list[str]:
    if width(cell) <= limit:
        return [cell]
    lines, line = [], ""
    for word in cell.split(" "):
        cand = (line + " " + word) if line else word
        if width(cand) > limit and line:
            lines.append(line)
            line = word
        else:
            line = cand
    if line:
        lines.append(line)
    return lines or [""]


def table(rows: list[list[str]], header: list[str] | None = None, align: list[str | None] | None = None, autowrap: bool = False,
          colwidth: int = 30, rowline: bool = False, format_header: bool = True, align_all: str | None = None) -> str:
    """A bordered table: +---+ lines, | cells |, a header row in capitals (centred), numbers right-aligned. Returns it with a newline at the end."""
    head = [(_title(h) if format_header else h) for h in header] if header else None
    ncols = max([len(head or [])] + [len(r) for r in rows]) if (rows or head) else 0
    split_rows = [[_wrap(r[i] if i < len(r) else "", colwidth) if autowrap else (r[i] if i < len(r) else "").split("\n") for i in range(ncols)] for r in rows]
    head_lines = [_wrap(h, colwidth) if autowrap else [h] for h in head] if head else None
    widths = [0] * ncols
    for i in range(ncols):
        cells = [line for r in split_rows for line in r[i]] + (head_lines[i] if head_lines else [])
        widths[i] = max([width(x) for x in cells] + [0])
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out_lines = [border]
    if head_lines:
        for k in range(max(len(x) for x in head_lines)):
            cells = []
            for i, w in enumerate(widths):
                h = head_lines[i][k] if k < len(head_lines[i]) else ""
                gap = w - width(h)
                cells.append(" " + " " * (gap // 2) + h + " " * (gap - gap // 2) + " ")
            out_lines.append("|" + "|".join(cells) + "|")
        out_lines.append(border)
    for ri, r in enumerate(split_rows):
        for k in range(max([len(x) for x in r] + [1])):
            cells = []
            for i, w in enumerate(widths):
                cell = r[i][k] if k < len(r[i]) else ""
                how = align_all or (align[i] if align and i < len(align) and align[i] else None)
                if how is None:
                    how = "R" if _NUMBER.match(cell.strip()) else "L"
                cells.append(" " + (" " * (w - width(cell)) + cell if how == "R" else pad(cell, w)) + " ")
            out_lines.append("|" + "|".join(cells) + "|")
        if rowline and ri < len(split_rows) - 1:
            out_lines.append(border)
    out_lines.append(border)
    return "\n".join(out_lines) + "\n"


# -- the commands Pal knows (for hints, help and suggestions) ---------------------------------------------------------------------
# (command, alias, description)
COMMANDS: list[tuple[str, str, str]] = [
    ("new", "", "start a new plan"),
    ("new --full", "", "start a new plan with auto-mode 'full'"),
    ("new --semi", "", "start a new plan with auto-mode 'semi'"),
    ("new --plus", "", "start a new plan with auto-mode 'plus'"),
    ("new --basic", "", "start a new plan with auto-mode 'basic'"),
    ("new --none", "", "start a new plan with auto-mode 'none'"),
    ("plans", "pl", "list plans"),
    ("cd", "", "set current plan by name or index"),
    ("current", "cu", "show current plan"),
    ("rename", "", "rename the current plan"),
    ("delete-plan", "dp", "delete plan by name or index"),
    ("config", "", "show current plan config"),
    ("set-config", "", "update current plan config"),
    ("config default", "", "show the default config for new plans"),
    ("set-config default", "", "update the default config for new plans"),
    ("set-auto", "", "update auto-mode (autonomy level) for current plan"),
    ("set-auto none", "", "set auto-mode to 'none'"),
    ("set-auto basic", "", "set auto-mode to 'basic'"),
    ("set-auto plus", "", "set auto-mode to 'plus'"),
    ("set-auto semi", "", "set auto-mode to 'semi'"),
    ("set-auto full", "", "set auto-mode to 'full'"),
    ("set-auto default", "", "set the default auto-mode for new plans"),
    ("load", "l", "load files/dirs/notes or pipe data into context"),
    ("ls", "", "list everything in context"),
    ("rm", "", "remove context by index, range, name, or glob"),
    ("clear", "", "remove all context"),
    ("update", "u", "update outdated context"),
    ("show", "", "show current context by name or index"),
    ("diff --ui", "", "review pending changes in a browser UI"),
    ("diff", "", "review pending changes in 'git diff' format"),
    ("summary", "", "show the latest summary of the current plan"),
    ("apply", "ap", "apply pending changes to project files"),
    ("reject", "rj", "reject pending changes to one or more project files"),
    ("log", "", "show log of plan updates"),
    ("rewind", "rw", "rewind to a previous state"),
    ("continue", "c", "continue the plan"),
    ("debug", "db", "repeatedly run a command and auto-apply fixes until it succeeds"),
    ("build", "b", "build any pending changes"),
    ("convo", "", "show plan conversation"),
    ("branches", "br", "list plan branches"),
    ("checkout", "co", "checkout or create a branch"),
    ("delete-branch", "dlb", "delete a branch by name or index"),
    ("plans --archived", "", "list archived plans"),
    ("archive", "arc", "archive a plan"),
    ("unarchive", "unarc", "unarchive a plan"),
    ("models", "", "show current plan model settings"),
    ("models available", "", "show all available models"),
    ("set-model", "", "update current plan model settings"),
    ("set-model default", "", "update the default model settings for new plans"),
    ("providers", "", "show all available model providers"),
    ("ps", "", "list active and recently finished plan streams"),
    ("stop", "", "stop an active plan stream"),
    ("connect", "conn", "connect to an active plan stream"),
    ("tokens", "", "show the tokens used and saved in this session"),
    ("budget", "", "cap the tokens this session may use"),
    ("endpoint", "", "use any OpenAI-compatible endpoint"),
    ("voice", "v", "speak your prompt (the mic, on this computer)"),
    ("key", "", "set an API key for this session (never saved)"),
    ("security", "", "check the project for common security problems"),
]
ALIASES = {a: cmd for cmd, a, _ in COMMANDS if a}
DESCS = {cmd: d for cmd, _, d in COMMANDS}
_ALIAS_OF = {cmd: a for cmd, a, _ in COMMANDS if a}


def cmd_label(cmd: str) -> str:
    a = _ALIAS_OF.get(cmd)
    return f"\\{cmd}" + (f" (\\{a})" if a else "")


def print_cmds(*cmds: str, prefix: str = "") -> None:
    """The suggested next commands, one per line:  \\cmd 👉 what it does."""
    if os.environ.get("PAL_DISABLE_SUGGESTIONS"):
        return
    for cmd in cmds:
        if cmd not in DESCS:
            continue
        err(prefix + chip(f" {cmd_label(cmd)} ") + " \U0001F449 " + DESCS[cmd])


def show_cmd(cmd: str) -> str:
    return chip(f" \\{cmd} ")


# -- errors ------------------------------------------------------------------------------------------------------------------------------
class Exit(Exception):  # noqa: N818  (a command ended early: the REPL carries on)
    def __init__(self, code: int = 0):
        super().__init__(code)
        self.code = code


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def simple_error(msg: str) -> None:
    err(c(f"{HI_RED};{BOLD}", "\U0001F6A8 " + _cap(msg)))


def fail(msg: str) -> None:
    """Print the error (a chain split on ': ', one level per line) and end the command."""
    spinner.stop()
    msg = msg.replace("status code:", "status code").replace(", body:", ":")
    parts = msg.split(": ")
    if len(parts) == 1:
        err(c(f"{HI_RED};{BOLD}", "\U0001F6A8 " + msg))
        raise Exit(1)
    shown: list[str] = []
    seen: set[str] = set()
    for i, part in enumerate(parts):
        if part.lower() in seen:
            continue
        seen.add(part.lower())
        if i == 0:
            shown.append(c(f"{HI_RED};{BOLD}", "\U0001F6A8 " + _cap(part)))
        elif len(parts[i - 1]) < 10:
            shown.append(": " + part)
        else:
            shown.append("\n" + "  " * i + "→ " + _cap(part))
    err(c(f"{HI_RED};{BOLD}", "".join(shown)))
    raise Exit(1)


def usage_error(msg: str) -> None:
    """A command used the wrong way: one line, 'Error: ...'."""
    spinner.stop()
    simple_error("Error: " + msg)
    raise Exit(1)


def no_plan() -> None:
    out("\U0001F937‍♂️ No current plan")
    out()
    print_cmds("new", "cd")
    raise Exit(1)


# -- the bar spinner, shown while something quick runs ---------------------------------------------------------------------------------
_FRAMES = ["[          ]", "[=         ]", "[==        ]", "[===       ]", "[====      ]", "[=====     ]", "[======    ]", "[=======   ]",
           "[========  ]", "[========= ]", "[==========]"]


class Spinner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.msg = ""
        self.started = 0.0
        self.active = False

    def start(self, msg: str = "") -> None:
        if self.active and msg == self.msg:
            return
        self._halt()
        self.msg, self.started, self.active = msg, time.monotonic(), True
        if not sys.stdout.isatty():
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pal-spinner")
        sys.stdout.write("\x1b[?25l")
        self._thread.start()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            sys.stdout.write("\r\x1b[K\r" + (self.msg + " ") + c(INK, _FRAMES[i % len(_FRAMES)]))
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.1)

    def _halt(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(0.5)
            self._thread = None
            sys.stdout.write("\r\x1b[K\x1b[?25h")
            sys.stdout.flush()

    def stop(self) -> None:
        if self.active:
            least = 0.7 if self.msg else 0.35
            left = least - (time.monotonic() - self.started)
            if left > 0:
                time.sleep(left)
        self._halt()
        self.active = False
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[2K\r")
            sys.stdout.flush()

    def resume(self) -> None:
        if not self.active:
            self.start(self.msg)


spinner = Spinner()


# -- single keys ---------------------------------------------------------------------------------------------------------------------------
def get_key() -> str:
    """One key, without Enter: a character, or 'down', 'up', 'enter', 'esc', 'ctrl-c'."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            return "ctrl-c"
        return line[:1] if line.strip() else "enter"
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            return {"P": "down", "H": "up", "K": "left", "M": "right"}.get(code, "")
        return {"\r": "enter", "\n": "enter", "\x03": "ctrl-c", "\x1b": "esc"}.get(ch, ch)
    import select as _select
    import termios
    import tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = os.read(fd, 1)
        need = 3 if first[0] >= 0xF0 else 2 if first[0] >= 0xE0 else 1 if first[0] >= 0xC0 else 0
        ch = (first + (os.read(fd, need) if need else b"")).decode("utf-8", "replace")
        if ch == "\x1b":
            seq = ""
            while _select.select([fd], [], [], 0.05)[0]:
                seq += os.read(fd, 1).decode("utf-8", "replace")
                if len(seq) >= 2 and seq[-1].isalpha():
                    break
            return {"[B": "down", "OB": "down", "[A": "up", "OA": "up"}.get(seq, "esc" if not seq else "")
        if ord(ch) >= 0xC0 or ch == "�":           # the rest of a multi-byte character
            return ch
        return {"\r": "enter", "\n": "enter", "\x03": "ctrl-c"}.get(ch, ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def confirm(msg: str) -> bool:
    """msg (y)es | (n)o> and one key. Ctrl+C is no."""
    while True:
        out(c(f"{HI_MAGENTA};{BOLD}", msg + " (y)es | (n)o"), end="")
        out(c(f"{HI_MAGENTA};{BOLD}", "> "), end="")
        k = get_key()
        if k == "ctrl-c":
            return False
        out(k if len(k) == 1 else "")
        if k in ("y", "Y"):
            return True
        if k in ("n", "N"):
            return False
        out()
        out(c(f"{HI_RED};{BOLD}", "Invalid input.\nEnter 'y' for yes or 'n' for no.\n"))


def confirm_cancel(msg: str) -> tuple[bool, bool]:
    """(yes, cancelled)."""
    while True:
        out(c(f"{HI_MAGENTA};{BOLD}", msg + " (y)es | (n)o | (c)ancel"), end="")
        out(c(f"{HI_MAGENTA};{BOLD}", "> "), end="")
        k = get_key()
        if k == "ctrl-c":
            return False, True
        out(k if len(k) == 1 else "")
        if k in ("y", "Y"):
            return True, False
        if k in ("n", "N"):
            return False, False
        if k in ("c", "C"):
            return False, True
        out()
        out(c(f"{HI_RED};{BOLD}", "Invalid input.\nEnter 'y' for yes, 'n' for no, or 'c' for cancel.\n"))


class Interrupted(Exception):  # noqa: N818
    pass


def _list_app(message: str, options: list[str], multi: bool):
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, Window

    st = {"sel": 0, "filter": "", "top": 0, "picked": set(), "result": None}

    def shown() -> list[str]:
        f = st["filter"].lower()
        return [o for o in options if f in o.lower()] if f else list(options)

    def render():
        opts = shown()
        st["sel"] = min(st["sel"], max(0, len(opts) - 1))
        if st["sel"] < st["top"]:
            st["top"] = st["sel"]
        if st["sel"] >= st["top"] + 7:
            st["top"] = st["sel"] - 6
        head = c(f"{BOLD};{HI_GREEN}", "? ") + c(f"{HI_MAGENTA};{BOLD}", message)
        if multi:
            head += "  " + c(MUTED, "[Use arrows to move, space to select, <right> to all, <left> to none, type to filter]")
        if st["filter"]:
            head += " " + st["filter"]
        lines = [head]
        for i, o in enumerate(opts[st["top"]:st["top"] + 7]):
            idx = st["top"] + i
            cur = c(f"{RED_HI};1", ">") if idx == st["sel"] else " "
            if multi:
                box = c(RED_HI, " [x] ") if o in st["picked"] else c(BOLD, " [ ] ")
                lines.append(f"{cur}{box} {o}")
            else:
                lines.append((cur + " " + c(f"{RED_HI};1", o)) if idx == st["sel"] else "  " + o)
        return ANSI("\n".join(lines))

    kb = KeyBindings()

    @kb.add("up")
    def _(e):
        n = len(shown())
        st["sel"] = (st["sel"] - 1) % n if n else 0
        if n and st["sel"] == n - 1:
            st["top"] = max(0, n - 7)

    @kb.add("down")
    @kb.add("tab")
    def _(e):
        n = len(shown())
        st["sel"] = (st["sel"] + 1) % n if n else 0
        if st["sel"] == 0:
            st["top"] = 0

    @kb.add("enter")
    def _(e):
        opts = shown()
        if multi:
            st["result"] = [o for o in options if o in st["picked"]]
        elif opts:
            st["result"] = opts[st["sel"]]
        else:
            return
        e.app.exit()

    @kb.add("c-c")
    def _(e):
        st["result"] = Interrupted
        e.app.exit()

    @kb.add("escape", eager=True)
    def _(e):
        st["filter"] = ""

    @kb.add("backspace")
    def _(e):
        st["filter"] = st["filter"][:-1]

    if multi:
        @kb.add("space")
        def _(e):
            opts = shown()
            if opts:
                o = opts[st["sel"]]
                st["picked"] ^= {o}

        @kb.add("right")
        def _(e):
            st["picked"] = set(shown()) | st["picked"]

        @kb.add("left")
        def _(e):
            st["picked"] = set()

    @kb.add("<any>")
    def _(e):
        ch = e.data
        if ch and ch.isprintable() and not (multi and ch == " "):
            st["filter"] += ch
            st["sel"], st["top"] = 0, 0

    app = Application(layout=Layout(HSplit([Window(FormattedTextControl(render), wrap_lines=True)])), key_bindings=kb, full_screen=False,
                      erase_when_done=True, mouse_support=False)
    return app, st


def select(message: str, options: list[str]) -> str:
    """? message, the options with > on the chosen one (arrows, typing filters, Enter). Ctrl+C raises Interrupted."""
    if not options:
        raise Interrupted()
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return options[0]
    app, st = _list_app(message, options, multi=False)
    app.run()
    if st["result"] is Interrupted or st["result"] is None:
        raise Interrupted()
    out(c(f"{BOLD};{HI_GREEN}", "? ") + c(f"{HI_MAGENTA};{BOLD}", message) + " " + c(RED_HI, st["result"]))
    return st["result"]


def multi_select(message: str, options: list[str]) -> list[str]:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return []
    app, st = _list_app(message, options, multi=True)
    app.run()
    if st["result"] is Interrupted or st["result"] is None:
        raise Interrupted()
    out(c(f"{BOLD};{HI_GREEN}", "? ") + c(f"{HI_MAGENTA};{BOLD}", message) + " " + c(RED_HI, ", ".join(st["result"])))
    return st["result"]


def ask_text(message: str) -> str:
    """A required line of text."""
    from prompt_toolkit import prompt as _prompt
    from prompt_toolkit.formatted_text import ANSI
    while True:
        if not sys.stdin.isatty():
            value = sys.stdin.readline().strip()
            if not value:
                raise Interrupted()
            return value
        try:
            value = _prompt(ANSI(c(f"{BOLD};{HI_GREEN}", "? ") + c(f"{HI_MAGENTA};{BOLD}", message) + " ")).strip()
        except (KeyboardInterrupt, EOFError):
            raise Interrupted() from None
        if value:
            return value
        out(c(f"{BOLD};{HI_RED}", "\U0001F6A8 This input is required"))


# -- time ------------------------------------------------------------------------------------------------------------------------------------
def rel_time(t: float, now: float | None = None) -> str:
    now = time.time() if now is None else now
    d = now - t
    word = "ago"
    if d < 0:
        d, word = -d, "from now"
    if d < 1:
        return "just now"
    steps = [(2, "1s"), (60, None), (120, "1m"), (3600, None), (7200, "1h"), (86400, None), (172800, "1d"), (604800, None), (1209600, "1w"),
             (2592000, None)]
    units = {60: (1, "s"), 3600: (60, "m"), 86400: (3600, "h"), 604800: (86400, "d"), 2592000: (604800, "w")}
    for limit, fixed in steps:
        if d < limit:
            if fixed:
                return f"{fixed} {word}"
            div, u = units[limit]
            return f"{int(d // div)}{u} {word}"
    lt = time.localtime(t)
    return time.strftime("%b ", lt) + str(lt.tm_mday) + time.strftime(" %Y", lt)


def log_time(t: float) -> str:
    """Mon Jan 2, 2006 | 3:04:05pm UTC"""
    g = time.gmtime(t)
    hour = g.tm_hour % 12 or 12
    return time.strftime("%a %b ", g) + f"{g.tm_mday}, {g.tm_year} | {hour}:{g.tm_min:02d}:{g.tm_sec:02d}{'am' if g.tm_hour < 12 else 'pm'} UTC"
