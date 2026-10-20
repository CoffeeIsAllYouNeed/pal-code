"""The live view while Pal writes: the prompt, the reply as it streams, the files being built, and the keys to stop, background or scroll.

It takes the whole window while it runs (and gives it back after), then prints the prompt, the reply and the files built where they stay:

     💬 User prompt 👇

    make a flappy bird game

     🤖 Pal reply 👇

    I'll create the game in two files ...
    ────────────────────────────────────────────
     🏗  Building plan                (↓) collapse
    📄 index.html ✅ | 📄 game.js 412 🪙
    ────────────────────────────────────────────
     (s)top • (b)ackground • (j/k) scroll • (d/u) page • (g/G) start/end
"""
from __future__ import annotations

import io
import re
import sys
import threading
import time

from pal import steps
from pal.terminal.repl import term
from pal.terminal.repl.term import BOLD, PROMPT_FG, WHITE, c, out, width
from pal.tokens import estimate_tokens

POINTS = ["∙∙∙", "●∙∙", "∙●∙", "∙∙●"]
MINIDOT = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
BORDER = "38;5;238"
HELP_FG = "38;2;138;138;138"
APPLY = "_apply.sh"                      # the commands to run, shown as "🚀 commands"

_HEADER = re.compile(r"^#{1,6}[ \t]*FILE:[ \t]*`?([^\n`]+?)`?[ \t]*$", re.M | re.I)
_COMMANDS = re.compile(r"^#{1,6}[ \t]*COMMANDS[ \t]*$", re.M | re.I)
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)


def _blocks(text: str) -> list[tuple[str, int, int, bool, str]]:
    """The file and command blocks in a reply: (path, start, end, finished, content)."""
    marks = [(m.start(), m.end(), m.group(1).strip()) for m in _HEADER.finditer(text)] + [(m.start(), m.end(), APPLY) for m in _COMMANDS.finditer(text)]
    marks.sort()
    found = []
    for i, (start, head_end, path) in enumerate(marks):
        limit = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[head_end:limit]
        opened = re.search(r"^[ \t]*(```|~~~)[^\n]*$", body, re.M)
        finished, end = False, limit
        if opened:
            closing = re.search(r"^[ \t]*" + re.escape(opened.group(1)) + r"[ \t]*$", body[opened.end():], re.M)
            if closing:
                finished = True
                end = head_end + opened.end() + closing.end()
            content = body[opened.end():(opened.end() + closing.start()) if closing else len(body)]
        else:
            content = ""
        found.append((path, start, end, finished or i + 1 < len(marks), content))
    return found


def reply_text(raw: str) -> str:
    """What the reply view shows: the model's words, without the files' code (the build view shows the files) or its thinking."""
    text = _THINK.sub("", raw)
    kept, pos = [], 0
    for _path, start, end, _fin, _content in _blocks(text):
        kept.append(text[pos:start])
        pos = end
    kept.append(text[pos:])
    text = "".join(kept)
    text = re.sub(r"^NOTE:[ \t]*", "", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class Stream:
    """One reply being written: fed by the model's pieces and Pal's steps, read by the view."""

    def __init__(self, prompt: str, build_only: bool = False, can_bg: bool = True, plan: str = "", branch: str = ""):
        self.prompt, self.build_only, self.can_bg = prompt, build_only, can_bg
        self.plan, self.branch = plan, branch
        self.id = f"{int(time.time() * 1000) % 100000:05d}"
        self.started = time.time()
        self.ended = 0.0
        self.segments: list[str] = [""]
        self.next_segment = False
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.finished = self.stopped = self.background = False
        self.error: BaseException | None = None
        self.turn = None
        self.status = "active"
        self.mode = "build"

    # -- fed from the worker -----------------------------------------------------------------------------------------------------
    def feed(self, piece: str) -> None:
        if not piece:
            return
        with self.lock:
            if self.next_segment and self.segments[-1]:
                self.segments.append("")
            self.next_segment = False
            self.segments[-1] += piece

    def on_step(self, event: str, key: str, text: str) -> None:
        if event == "begin" and key != "model":
            with self.lock:
                if self.segments[-1]:
                    self.next_segment = True
        elif event == "begin" and key == "model":
            with self.lock:
                if self.segments[-1]:
                    self.next_segment = True

    # -- read by the view ----------------------------------------------------------------------------------------------------------
    def snapshot(self) -> tuple[list[str], bool]:
        with self.lock:
            return list(self.segments), self.next_segment

    def reply(self) -> str:
        segs, _ = self.snapshot()
        parts = [reply_text(s) for s in segs]
        return "\n\n\U0001F447\n\n".join(p for p in parts if p)

    def raw_reply(self) -> str:
        return "\n\n".join(self.snapshot()[0])

    def files(self) -> dict[str, dict]:
        """path -> {"tokens", "finished"} over every segment: a file written again starts over."""
        state: dict[str, dict] = {}
        for seg in self.snapshot()[0]:
            for path, _s, _e, fin, content in _blocks(_THINK.sub("", seg)):
                state[path] = {"tokens": estimate_tokens(content) if content else 0, "finished": fin}
        if self.finished:
            for v in state.values():
                v["finished"] = True
        return state

    @property
    def starting(self) -> bool:
        segs, nxt = self.snapshot()
        return not any(s.strip() for s in segs)

    @property
    def processing(self) -> bool:
        segs, nxt = self.snapshot()
        return not self.done.is_set() and (not any(s.strip() for s in segs) or nxt)


# -- drawing -------------------------------------------------------------------------------------------------------------------------
_MD_CACHE: dict[tuple[str, int], str] = {}


def markdown(text: str, w: int) -> str:
    key = (text, w)
    if key in _MD_CACHE:
        return _MD_CACHE[key]
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.theme import Theme
    buf = io.StringIO()
    con = Console(file=buf, width=max(20, min(w, 80) - 2), force_terminal=term.colours_on(), color_system="truecolor", highlight=False,
                  theme=Theme({"markdown.h1": "bold #E50914", "markdown.h2": "bold #FFFFFF", "markdown.h3": "bold #FFFFFF", "markdown.h4": "bold #FFFFFF",
                               "markdown.code": "bold #FF4D57 on #262626", "markdown.item.bullet": "bold #FF4D57",
                               "markdown.item.number": "bold #FF4D57", "markdown.link": "underline #FF4D57", "markdown.hr": "#444444"}))
    con.print(Markdown(text, code_theme="monokai"))
    lines = buf.getvalue().rstrip("\n").split("\n")
    rendered = "\n".join(("  " + ln).rstrip() for ln in lines).strip()
    if len(_MD_CACHE) > 64:
        _MD_CACHE.clear()
    _MD_CACHE[key] = rendered
    return rendered


def _hex(sgr: str) -> str:
    r, g, b = sgr.split(";")[-3:]
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _wrap_plain(text: str, w: int) -> str:
    import textwrap
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=max(10, w), break_long_words=True, replace_whitespace=False) or [""]
    return "\n".join(lines)


def main_display(st: Stream, w: int) -> str:
    s = ""
    if st.prompt:
        s += c(f"48;2;184;50;58;{BOLD};{WHITE}", " \U0001F4AC User prompt \U0001F447 ")
        s += "\n\n" + c(PROMPT_FG, _wrap_plain(st.prompt.strip(), min(w - 2, 80))) + "\n"
    reply = st.reply()
    if reply:
        s += "\n" + c(f"{term.BG_WHITE};{BOLD};{term.BLACK}", " \U0001F916 Pal reply \U0001F447 ")
        s += "\n\n" + markdown(reply, w)
    else:
        s += "\n"
    return s


def _build_header(st: Stream, w: int, static: bool, collapsed: bool) -> str:
    did = not (st.stopped or st.error is not None)
    if static:
        text, bg = ("Built plan " if did else "Build incomplete "), (term.BG_GREEN if did else term.BG_RED_STD)
    else:
        text, bg = "Building plan ", term.BG_GREEN
    head = c(f"{bg};{WHITE};{BOLD}", " \U0001F3D7  ") + c(f"{bg};{WHITE}", text)
    hint = "" if static else ("(↑) expand" if collapsed else "(↓) collapse")
    gap = w - width(head) - width(hint) - 1
    if gap > 0:
        head += " " * gap + hint
    return head


def _order(files: dict) -> list[str]:
    return sorted(p for p in files if p != APPLY) + ([APPLY] if APPLY in files else [])


def build_rows(st: Stream, w: int, static: bool, frame: int) -> list[str]:
    files = st.files()
    did = not (st.stopped or st.error is not None)
    blocks = []
    for path in _order(files):
        icon, name = ("\U0001F680", "commands") if path == APPLY else ("\U0001F4C4", path)
        f = files[path]
        block = f"{icon} {name}"
        if f["finished"] or (did and static):
            block += " ✅"
        elif f["tokens"] > 0:
            block += f" {f['tokens']} \U0001FA99"
        else:
            block += " " + MINIDOT[frame % len(MINIDOT)]
        if width(block) > w:
            room = w - 1
            block = block[0] + "⋯" if room < 4 else block[:room // 2] + "⋯" + block[-(room // 2):]
        blocks.append(block)
    rows: list[list[str]] = []
    line_w = 0
    for block in blocks:
        cand = (" | " if rows and rows[-1] else "") + block
        if not rows or line_w + width(cand) > w:
            rows.append([])
            line_w, cand = 0, block
        rows[-1].append(cand)
        line_w += width(cand)
    return ["".join(r) for r in rows]


def build_view(st: Stream, w: int, static: bool, collapsed: bool, frame: int) -> list[str]:
    files = st.files()
    if not files:
        return []
    top = [] if st.build_only else [c(BORDER, "─" * w)]
    if collapsed and not static:
        total = len([p for p in files if p != APPLY])
        busy = len([p for p, f in files.items() if p != APPLY and not f["finished"]])
        spin = MINIDOT[frame % len(MINIDOT)]
        summary = ""
        if total:
            summary = f" \U0001F4C4 {total} {'files' if total > 1 else 'file'}"
        if busy:
            summary += f" • \U0001F4DD editing {busy} {spin}"
        if APPLY in files:
            if total:
                summary += " •"
            summary += " \U0001F680 wrote commands" if files[APPLY]["finished"] else f" \U0001F680 editing commands {spin}"
        return top + [_build_header(st, w, False, True), summary]
    return top + [_build_header(st, w, static, collapsed)] + build_rows(st, w, static, frame)


def help_bar(st: Stream, w: int) -> list[str]:
    text = " (s)top" + (" • (b)ackground" if st.can_bg else "")
    if not st.build_only:
        text += " • (j/k) scroll • (d/u) page • (g/G) start/end"
    import textwrap
    lines = textwrap.wrap(text, width=max(10, w), drop_whitespace=False) if term.width(text) > w else [text]
    return [c(BORDER, "─" * w)] + [c(HELP_FG, term.pad(ln, w)) for ln in lines]


def static_build(st: Stream, w: int) -> str:
    return "\n".join(build_view(st, w, True, False, 0))


# -- the live view --------------------------------------------------------------------------------------------------------------------
class View:
    def __init__(self, st: Stream):
        self.st = st
        self.frame = 0
        self.y = 0
        self.at_bottom = True
        self.collapsed = False
        self.user_toggled = False
        self.result = ""

    def compose(self, w: int, h: int) -> str:
        st = self.st
        files = st.files()
        if not self.user_toggled:
            self.collapsed = len(build_view(st, w, False, False, self.frame)) - 1 > 3 if files else False
        helps = help_bar(st, w)
        builds = build_view(st, w, False, self.collapsed, self.frame) if files else []
        proc = ["", " " + c(term.PINK, POINTS[self.frame % len(POINTS)])] if st.processing else []
        views: list[str] = []
        if not st.build_only:
            content = main_display(st, w).split("\n")
            avail = max(0, h - (len(helps) + len(proc) + len(builds)))
            vh = min(avail, len(content))
            most = max(0, len(content) - vh)
            if self.at_bottom:
                self.y = most
            self.y = max(0, min(self.y, most))
            self.page = max(1, vh)
            self.scrollable = len(content) > vh
            for ln in content[self.y:self.y + vh]:
                views.append(" " + ln)
        views += proc + builds + helps
        if len(views) > h:
            views = views[len(views) - h:]
        return "\n".join(views)

    def run(self) -> str:
        """Until the reply is done ("finished"), or the person stops ("stopped") or backgrounds it ("background")."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.formatted_text import ANSI
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, Window

        st = self.st
        holder: dict = {}

        def text():
            app = holder["app"]
            size = app.output.get_size()
            return ANSI(self.compose(size.columns, size.rows))

        kb = KeyBindings()

        def end(e, how: str) -> None:
            self.result = how
            e.app.exit()

        @kb.add("s")
        @kb.add("c-c")
        def _(e):
            if st.done.is_set() and st.finished:                       # it had already finished
                end(e, "finished")
                return
            steps.request_stop()
            st.stopped = True
            end(e, "stopped")

        @kb.add("b")
        def _(e):
            if st.can_bg:
                st.background = True
                end(e, "background")

        def move(n: int):
            def f(e):
                if not getattr(self, "scrollable", False):
                    return
                most = max(0, len(main_display(st, e.app.output.get_size().columns).split("\n")) - self.page)
                self.y = max(0, min(most, self.y + n))
                self.at_bottom = n > 0 and self.y >= most
            return f

        kb.add("j")(move(1))
        kb.add("k")(move(-1))

        @kb.add("d")
        @kb.add("pagedown")
        def _(e):
            move(getattr(self, "page", 1))(e)

        @kb.add("u")
        @kb.add("pageup")
        def _(e):
            move(-getattr(self, "page", 1))(e)

        @kb.add("g")
        @kb.add("home")
        def _(e):
            if getattr(self, "scrollable", False):
                self.y, self.at_bottom = 0, False

        @kb.add("G")
        @kb.add("end")
        def _(e):
            if getattr(self, "scrollable", False):
                self.at_bottom = True

        @kb.add("up")
        def _(e):
            if st.files():
                self.collapsed, self.user_toggled = False, True

        @kb.add("down")
        def _(e):
            if st.files():
                self.collapsed, self.user_toggled = True, True

        app: Application = Application(layout=Layout(HSplit([Window(FormattedTextControl(text), wrap_lines=False)])), key_bindings=kb, full_screen=True,
                          mouse_support=False)
        holder["app"] = app

        def tick() -> None:
            while not st.done.wait(0.1):
                self.frame += 1
                app.invalidate()
            self.frame += 1
            if not self.result:
                self.result = "finished"
                for _ in range(200):                                     # the app may not be running yet
                    loop = app.loop
                    if app.is_running and loop is not None:
                        try:
                            loop.call_soon_threadsafe(lambda: app.exit() if app.is_running else None)
                        except Exception:  # noqa: BLE001
                            pass
                        return
                    time.sleep(0.02)

        t = threading.Thread(target=tick, daemon=True, name="pal-stream-tick")
        t.start()
        if st.done.is_set():
            return "finished"
        app.run()
        return self.result or "finished"


# -- running one request -----------------------------------------------------------------------------------------------------------------
ACTIVE: list[Stream] = []           # replies running now or recently (\ps, \connect, \stop)


def active() -> Stream | None:
    for s in ACTIVE:
        if not s.done.is_set():
            return s
    return None


def start(wf, send: str, shown: str, mode: str, can_bg: bool = True, build_only: bool = False) -> Stream:
    """Start the request in a thread; the stream is returned at once."""
    st = Stream(shown, build_only=build_only, can_bg=can_bg, plan=wf.plan.name, branch=wf.plan.branch)
    st.mode = mode
    ACTIVE.append(st)
    del ACTIVE[:-20]
    s = wf.session

    def work() -> None:
        before = s.mode
        s.mode = mode
        steps.clear_stop()
        try:
            with steps.stream_to(st.feed), steps.listen(st.on_step):
                st.turn = wf.tell(send)
            st.finished = True
            st.status = "finished"
        except steps.Stopped:
            st.stopped = True
            st.status = "stopped"
        except BaseException as e:  # noqa: BLE001  (shown after the view closes)
            st.error = e
            st.status = "error"
        finally:
            s.mode = before
            st.ended = time.time()
            st.done.set()

    threading.Thread(target=work, daemon=True, name="pal-reply").start()
    return st


def show(st: Stream) -> str:
    """The live view (in a terminal), then what stays on screen. Returns "finished", "stopped", "background" or "error"."""
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    how = "finished"
    if interactive:
        try:
            how = View(st).run()
        except Exception:  # noqa: BLE001  (no full-screen terminal: wait and print)
            st.done.wait()
    else:
        st.done.wait()
    if how == "finished":
        st.done.wait()
    elif how == "stopped":
        st.done.wait(10)                                                 # the model call ends at its next piece
    w = term.term_width()
    out()
    if not st.build_only:
        out(main_display(st, w))
    if st.files():
        out(static_build(st, w))
    if st.error is not None and how == "finished":
        out()
        return "error"
    if how == "stopped" or (st.stopped and how == "finished"):
        out()
        out(c(f"{term.BG_BLACK};{BOLD};{term.HI_RED}", " \U0001F6D1 Stopped early "))
        out()
        term.print_cmds("log", "rewind")
        return "stopped"
    if how == "background":
        out()
        out(c(f"{term.BG_BLACK};{BOLD};{term.HI_GREEN}", " ✅ Plan is active in the background "))
        out()
        term.print_cmds("ps", "connect", "stop")
        return "background"
    return "finished"
