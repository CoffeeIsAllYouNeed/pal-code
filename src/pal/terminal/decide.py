"""Decisions: one question with a few choices, asked the way Claude asks, in Pal's colours.

    ┌ Database ──────────────────────────────────────────────────────────────
    │ Which database should the app use?
    │
    │ > 1. SQLite (Recommended)
    │      one file, no server: right for a small app
    │   2. PostgreSQL
    │      a server; better for many users at once
    │   3. Something else…   (type it)
    └ ↑↓ choose · enter select · 1-9 jump · esc skip

Used when the Ask agent needs a choice from the person to answer well (Build and Plan never stop to ask). Without a terminal (pipes, tests) it falls back to a numbered list and a typed answer.
"""
from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from rich.console import Console
from rich.text import Text

from pal.terminal import theme

OTHER = "Something else…"
MAX_QUESTIONS = 3
MAX_OPTIONS = 5


@dataclass
class Option:
    label: str
    description: str = ""


@dataclass
class Question:
    question: str
    options: list[Option] = field(default_factory=list)
    header: str = "Decision"
    recommended: int | None = None
    other: bool = True                      # a last "Something else…" choice that takes typed text

    def labels(self) -> list[str]:
        return [o.label + (" (Recommended)" if i == self.recommended else "") for i, o in enumerate(self.options)]


# -- what the agents send --------------------------------------------------------------------------------------------------------
DECIDE = ("Questions back (Ask agent only): when a good answer depends on a choice only the person can make (which of several meanings they intend, "
          "their goal, a trade-off they must pick), do not guess: ask. End your reply with a line `### QUESTION` and a ```json block: "
          "{\"header\": \"a 1-2 word topic\", \"question\": \"...?\", \"options\": [{\"label\": \"1-5 words\", \"description\": \"one line on what it "
          "leads to\"}], \"recommended\": 0} with 2 to 4 options, the one you recommend first (or a JSON list of up to 3 such questions). "
          "Ask only when it really matters; otherwise answer directly.")

_BLOCK = re.compile(r"^#+\s*QUESTIONS?\s*$\s*```(?:json)?\s*\n(.*?)```", re.M | re.S | re.I)


def parse_questions(text: str) -> tuple[list[Question], str]:
    """The questions in a reply's `### QUESTION` block, and the reply without the block. A block that is not valid is left in the text."""
    m = _BLOCK.search(text or "")
    if not m:
        return [], text
    try:
        raw = json.loads(m.group(1))
    except ValueError:
        return [], text
    items = raw if isinstance(raw, list) else [raw]
    out: list[Question] = []
    for item in items[:MAX_QUESTIONS]:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        opts = []
        for o in item.get("options") or []:
            if isinstance(o, str) and o.strip():
                opts.append(Option(o.strip()[:80]))
            elif isinstance(o, dict) and str(o.get("label", "")).strip():
                opts.append(Option(str(o["label"]).strip()[:80], str(o.get("description", "")).strip()[:200]))
        if len(opts) < 2:
            continue
        rec = item.get("recommended")
        out.append(Question(str(item["question"]).strip()[:300], opts[:MAX_OPTIONS], str(item.get("header") or "Decision").strip()[:24],
                            rec if isinstance(rec, int) and 0 <= rec < len(opts[:MAX_OPTIONS]) else None))
    if not out:
        return [], text
    return out, (text[:m.start()] + text[m.end():]).strip()


def answers_message(questions: list[Question], answers: list[str]) -> str:
    """What goes back to the agent once the person has decided."""
    lines = [f"- {q.question} → {a}" for q, a in zip(questions, answers)]
    return "My decisions:\n" + "\n".join(lines) + "\nGo ahead with these."


# -- asking -------------------------------------------------------------------------------------------------------------------------
def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def ask(console: Console, q: Question, reader: Callable[[str], str] | None = None) -> str | None:
    """The chosen label (or the typed answer for "Something else…"), or None when the person skips (Esc, Ctrl+C)."""
    if reader is not None or not _interactive():
        return _ask_plain(console, q, reader or (lambda prompt: console.input(prompt)))
    return _ask_card(q)


def _ask_plain(console: Console, q: Question, reader: Callable[[str], str]) -> str | None:
    console.print(Text(f"   {q.header}: {q.question}", style=f"bold {theme.WHITE}"))
    labels = q.labels() + ([OTHER] if q.other else [])
    for i, label in enumerate(labels, 1):
        desc = q.options[i - 1].description if i <= len(q.options) else "type your own answer"
        console.print(Text.assemble((f"     {i}. ", theme.RED_HI), (label, theme.WHITE), (f"   {desc}" if desc else "", theme.MUTED)))
    try:
        raw = reader("   choose a number (Enter: " + (str((q.recommended or 0) + 1)) + "): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return q.options[q.recommended or 0].label
    if raw.isdigit() and 1 <= int(raw) <= len(labels):
        n = int(raw) - 1
        if n < len(q.options):
            return q.options[n].label
        try:
            typed = reader("   your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return typed or None
    return raw                                            # typed an answer directly


def _ask_card(q: Question) -> str | None:
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.containers import ConditionalContainer
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.styles import Style

    labels = q.labels() + ([OTHER] if q.other else [])
    n_other = len(q.options) if q.other else -1
    sel = {"i": q.recommended or 0}
    out: dict[str, str | None] = {"result": None}
    typed = Buffer(multiline=False)
    style = Style.from_dict({"bar": theme.RED, "head": f"bold {theme.RED_HI}", "q": f"bold {theme.WHITE}", "sel": f"bold {theme.WHITE}",
                             "pointer": f"bold {theme.RED}", "opt": theme.INK, "desc": theme.MUTED, "hint": theme.MUTED, "key": f"bold {theme.WHITE}"})

    def body():
        out = [("class:bar", "   ┌ "), ("class:head", q.header), ("class:bar", " " + "─" * max(4, 60 - len(q.header)) + "\n"),
               ("class:bar", "   │ "), ("class:q", q.question + "\n"), ("class:bar", "   │\n")]
        for i, label in enumerate(labels):
            on = i == sel["i"]
            out += [("class:bar", "   │ "), ("class:pointer", "> " if on else "  "), ("class:sel" if on else "class:opt", f"{i + 1}. {label}")]
            if i == n_other:
                out.append(("class:desc", "   (type it below)" if not on else "   type, then enter"))
            out.append(("", "\n"))
            desc = q.options[i].description if i < len(q.options) else ""
            if desc:
                out += [("class:bar", "   │ "), ("class:desc", f"     {desc}\n")]
        return FormattedText(out)

    def hints():
        return FormattedText([("class:bar", "   └ "), ("class:key", "↑↓"), ("class:hint", " choose · "), ("class:key", "enter"), ("class:hint", " select · "),
                              ("class:key", "1-9"), ("class:hint", " jump · "), ("class:key", "esc"), ("class:hint", " skip")])
    kb = KeyBindings()
    typing = Condition(lambda: sel["i"] == n_other)

    @kb.add("up")
    def _(e):
        sel["i"] = (sel["i"] - 1) % len(labels)

    @kb.add("down")
    @kb.add("tab")
    def _(e):
        sel["i"] = (sel["i"] + 1) % len(labels)

    for d in range(1, min(9, len(labels)) + 1):
        @kb.add(str(d), filter=~typing | Condition(lambda: not typed.text))
        def _(e, d=d):
            sel["i"] = d - 1
            if d - 1 != n_other:
                out["result"] = q.options[d - 1].label
                e.app.exit()

    @kb.add("enter")
    def _(e):
        if sel["i"] == n_other:
            if typed.text.strip():
                out["result"] = typed.text.strip()
                e.app.exit()
            return
        out["result"] = q.options[sel["i"]].label
        e.app.exit()

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(e):
        e.app.exit()
    typed_win = Window(BufferControl(typed), height=1)
    field_row = ConditionalContainer(VSplit([Window(FormattedTextControl([("class:bar", "   │   "), ("class:pointer", "› ")]), width=9, dont_extend_width=True),
                                            typed_win]), filter=typing)
    if n_other >= 0:                                     # typing anything goes to "Something else…"
        typed.on_text_changed += lambda _b: sel.update(i=n_other) if typed.text else None
    root = HSplit([Window(FormattedTextControl(body), dont_extend_height=True), field_row, Window(FormattedTextControl(hints), height=1)])
    app: Application = Application(layout=Layout(root, focused_element=typed_win), key_bindings=kb, style=style, full_screen=False, erase_when_done=True)
    try:
        app.run()
    except (EOFError, KeyboardInterrupt):
        return None
    return out["result"]


def show_answer(console: Console, q: Question, answer: str | None) -> None:
    """What stays on screen after the card disappears: the question and what was chosen."""
    console.print(Text.assemble(("   ◆ ", theme.RED), (f"{q.question}  ", theme.INK), ("→ " + answer if answer else "→ skipped", f"bold {theme.WHITE}" if answer else theme.MUTED)))


def ask_all(console: Console, questions: list[Question], reader: Callable[[str], str] | None = None) -> list[str] | None:
    """Every question in turn. None when the person skips one."""
    answers = []
    for q in questions:
        a = ask(console, q, reader)
        show_answer(console, q, a)
        if a is None:
            return None
        answers.append(a)
    return answers


def yes_no(console: Console, question: str, yes: str = "Yes", no: str = "No", yes_about: str = "", no_about: str = "", default: bool = False,
           header: str = "Decision", reader: Callable[[str], str] | None = None) -> bool:
    """A yes/no decision as a card. The default is the recommended choice."""
    q = Question(question, [Option(yes, yes_about), Option(no, no_about)], header, 0 if default else 1, other=False)
    if reader is None and not _interactive():
        return default
    a = ask(console, q, reader)
    show_answer(console, q, a)
    return a == yes
