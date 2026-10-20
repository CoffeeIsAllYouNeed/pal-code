"""Session commands in the style of Claude Code, for Pal: /init, /clear, /compact, /btw, /cost, /context, /memory, /add-dir, /permissions, /config,
/plan /build /ask, /copy, /export, /agents, /simplify, /status, /doctor, /resume, /shortcuts, /voice.

`handle_extra` returns (session, keep going) for a command it knows, or None so the caller tries the others.
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich.table import Table
from rich.text import Text

from pal import __version__
from pal.terminal import backend, theme, ui
from pal.terminal.project import NOTES_FILE
from pal.terminal.session import MODES, PalSession
from pal.tokens import estimate_tokens

COMMANDS: list[tuple[str, str]] = [
    ("/init", f"look through the project and start a {NOTES_FILE} (its instructions for Pal); you review it like any change"),
    ("/clear", "a fresh conversation (the plan, its context and pending changes stay). Alias: /reset"),
    ("/compact", "shorten the conversation to a summary to save tokens: /compact [what to keep]"),
    ("/btw", "a quick side question that is not added to the conversation: /btw what does HTTP 409 mean?"),
    ("/cost", "tokens used and saved this session, and the cost when the provider's price is known"),
    ("/context", "what goes to the model with each request, and how big each part is"),
    ("/memory", f"show the project's {NOTES_FILE}; /memory edit opens it in your editor"),
    ("/add-dir", "read files from another folder too: /add-dir ../shared-lib"),
    ("/permissions", "what Pal may do by itself (autonomy). Alias of /autonomy"),
    ("/config", "settings: model per agent, endpoint, autonomy, where things are kept. Alias: /settings"),
    ("/plan", "switch to the Plan agent, and optionally ask it: /plan add a login page"),
    ("/build", "switch to the Build agent, and optionally ask it"),
    ("/ask", "switch to the Ask agent, and optionally ask it"),
    ("/copy", "copy the last answer to the clipboard"),
    ("/export", "save the conversation as markdown: /export [file]"),
    ("/agents", "the three agents and the model each one uses"),
    ("/simplify", "review the files changed since the last commit for reuse, quality and speed; fixes go to the sandbox"),
    ("/status", "version, folder, models, plan, autonomy, this computer"),
    ("/doctor", "check this computer and the install"),
    ("/resume", "pick up an earlier plan and its conversation: /resume [name or number]"),
    ("/voice", "talk instead of typing (or tap the mic on the first screen)"),
    ("/shortcuts", "every keyboard shortcut (also ?)"),
]

SHORTCUTS: list[tuple[str, str]] = [
    ("enter", "send"), ("alt+enter  or  \\ then enter", "a new line"), ("↑ / ↓", "earlier messages (inside a long message: move between lines)"),
    ("esc esc", "clear what you typed"), ("esc", "stop Pal while it works"), ("ctrl+c", "clear the box, stop Pal, or leave on an empty box"),
    ("ctrl+d / ctrl+q", "leave"), ("tab", "switch agent: Build, Plan, Ask (or complete a command)"),
    ("shift+tab", "cycle autonomy: none, basic, plus, semi (full only with /autonomy full)"), ("/  or  ctrl+p", "the command menu (↑↓ to pick, enter or tab)"),
    ("ctrl+r", "search earlier messages (again: older)"), ("ctrl+l", "clear the screen"),
    ("?", "these shortcuts (on an empty box)"), ("@path", "add a file to the message"), ("drop files on the window", "attach them"),
    ("ctrl+a / ctrl+e", "start / end of the line"), ("ctrl+u / ctrl+k", "delete to the start / end of the line"), ("alt+b / alt+f", "a word back / forward"),
]


def _say(console, text: str, style: str = theme.MUTED) -> None:
    ui.say(console, Text(text, style=style))


def copy_to_clipboard(text: str) -> str:
    """Copy with the system's tool, or the terminal's OSC 52 (works over SSH in most terminals). Returns how it was copied, or ''."""
    tools = [["pbcopy"]] if sys.platform == "darwin" else [["clip"]] if os.name == "nt" else [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"]]
    for cmd in tools:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode("utf-16le" if cmd[0] == "clip" else "utf-8"), check=True, timeout=5)
                return cmd[0]
            except (OSError, subprocess.SubprocessError):
                continue
    if sys.stdout.isatty():
        sys.stdout.write("\x1b]52;c;" + base64.b64encode(text.encode("utf-8")).decode("ascii") + "\x07")
        sys.stdout.flush()
        return "the terminal (OSC 52)"
    return ""


def conversation_markdown(wf) -> str:
    b = wf.plan.b
    out = [f"# Pal conversation: plan {wf.plan.name} ({wf.plan.branch})", "", f"_{time.strftime('%Y-%m-%d %H:%M')} · {wf.session.cwd}_", ""]
    for m in b.convo:
        who = "**You**" if m.get("role") == "you" else f"**Pal ({m.get('agent', '')})**"
        out += [f"{who}", "", m.get("text", ""), ""]
    if b.pending:
        out += ["## Pending changes", ""] + [f"- {p}" for p in sorted(b.pending)] + [""]
    return "\n".join(out)


def handle_extra(cmd: str, rest: str, args, session: PalSession, console, core_factory, box: dict | None):
    wf = box["wf"] if box else None
    host = getattr(args, "host", None) or backend.DEFAULT_HOST
    if cmd in ("/shortcuts", "/keys", "?"):
        grid = Table.grid(padding=(0, 3))
        for k, what in SHORTCUTS:
            grid.add_row(Text(k, style=f"bold {theme.WHITE}"), Text(what, style=theme.MUTED))
        ui.say(console, grid)
        console.print()
        return session, True

    if cmd in ("/plan", "/build", "/ask"):
        session.mode = cmd[1:]
        backend.save_prefs(mode=session.mode)
        _say(console, f"Agent: {session.mode.capitalize()}")
        console.print()
        if rest and wf is not None:
            from pal.terminal import flow_cli
            flow_cli.run_turn(wf, rest, console)
        return session, True

    if cmd in ("/clear", "/reset"):
        if wf is None or box is None:
            return session, True
        from pal.terminal.cli import _bind_status, _fresh
        from pal.terminal.workflow import Workflow
        fresh = _fresh(session, args, core_factory)
        wf.plan.b.convo = []
        wf.save()
        box["wf"] = Workflow(fresh, wf.store, plan=wf.plan.name)
        _bind_status(box["wf"])
        _say(console, "Fresh conversation. The plan, its context and pending changes are still here (/current).")
        console.print()
        return fresh, True

    if cmd == "/compact":
        if wf is None or box is None or not wf.plan.b.convo:
            _say(console, "Nothing to compact yet.")
            console.print()
            return session, True
        from pal.terminal import flow_cli
        from pal.terminal.cli import _bind_status, _fresh
        from pal.terminal.workflow import Workflow
        before = sum(estimate_tokens(m.get("text", "")) for m in wf.plan.b.convo)
        transcript = "\n".join(f"{m.get('role')}: {m.get('text', '')[:3000]}" for m in wf.plan.b.convo[-60:])
        focus = f" Keep in particular: {rest}." if rest else ""
        mode = session.mode
        session.mode = "ask"
        try:
            reply = flow_cli.safely(console, lambda: flow_cli.spin(session, console, "Pal is summarising…", lambda: session.client.ask(
                "Summarise this conversation so it can continue from the summary alone: decisions made, what was built, open questions, "
                f"names of files and functions that matter. Short bullet points, no preface.{focus}\n\n{transcript}")))
        finally:
            session.mode = mode
        if reply is None:
            return session, True
        summary = ui.clean_terminal(reply.text).strip()
        fresh = _fresh(session, args, core_factory)
        wf.plan.b.convo = [{"role": "pal", "agent": "summary", "text": "Summary of the conversation so far:\n" + summary, "t": time.time()}]
        wf.save()
        box["wf"] = Workflow(fresh, wf.store, plan=wf.plan.name)
        _bind_status(box["wf"])
        after = estimate_tokens(summary)
        _say(console, f"Compacted: the conversation is now a summary of about {after:,} tokens (it was about {before:,}).")
        console.print()
        return fresh, True

    if cmd == "/btw":
        if not rest:
            _say(console, "Use: /btw <a quick question>   (not added to the conversation)")
            console.print()
            return session, True
        from pal.terminal import flow_cli
        from pal.terminal.session import make_session
        prov, model = session.model_for("ask")
        try:
            core = (core_factory or backend.make_core)(prov, model, host)
        except Exception as e:  # noqa: BLE001
            _say(console, f"Could not start the side question: {e}", theme.RED_HI)
            return session, True
        side = make_session(core, model, prov, session.cwd, "ask", helper=session.helper_core, local=session.local, project=session.project)
        try:
            ui.show_user(console, "btw: " + rest)
            result = flow_cli.safely(console, lambda: flow_cli.spin(side, console, "Pal is answering on the side…", lambda: side.ask(rest)))
            if result is not None:
                ui.show_reply(console, result, side)
                session.pal_tokens += side.pal_tokens
                session.cost_usd += side.cost_usd
        finally:
            side.close()
        _say(console, "(not added to the conversation)")
        console.print()
        return session, True

    if cmd == "/cost":
        grid = Table.grid(padding=(0, 3))
        rows = [("used", f"{session.pal_tokens:,} tokens"), ("helper (local)", f"{session.helper_tokens:,} tokens"),
                ("a plain chat would have used", f"about {session.plain_tokens:,} tokens"),
                ("saved", f"about {max(0, session.saved):,} tokens ({session.saved_pct:g}%)"),
                ("cost", f"${session.cost_usd:.4f}" if session.cost_usd else "no price known (local models are free)"),
                ("requests", str(session.turns)), ("time with the model", f"{session.seconds:.0f}s")]
        for k, v in rows:
            grid.add_row(Text(k, style=theme.MUTED), Text(v, style=f"bold {theme.WHITE}"))
        ui.say(console, grid)
        _say(console, "Where the savings came from: /tokens")
        console.print()
        return session, True

    if cmd == "/context":
        parts: list[tuple[str, int, str]] = []
        if session.project is not None:
            r = session.project.report
            parts.append(("project map (indexed)", r.map_tokens, "only matching lines are sent"))
            docs = sum(d.tokens for d in r.docs if not d.error)
            if docs:
                parts.append((f"documents, {len(r.docs)} (indexed)", docs, "only matching passages are sent"))
            parts.append(("code graph", 0, r.graph))
            notes = session.project.notes()
            if notes:
                parts.append((NOTES_FILE, estimate_tokens(notes), "sent once per session"))
        if wf is not None:
            rows = wf.context_rows()
            if rows:
                parts.append((f"loaded context, {len(rows)} item(s)", sum(row[2] for row in rows), "sent when it changes"))
            if wf.plan.b.pending:
                parts.append((f"pending changes, {len(wf.plan.b.pending)} file(s)", sum(estimate_tokens(v) for v in wf.plan.b.pending.values()), "diffs are taken against these"))
        parts.append(("conversation so far", session.history_tokens, "recent turns sent, older ones only when relevant"))
        top = max((n for _, n, _ in parts), default=1) or 1
        grid = Table.grid(padding=(0, 2))
        for name, n, note in parts:
            bar = "█" * max(1, round(24 * n / top)) if n else ""
            grid.add_row(Text(name, style=theme.WHITE), Text(f"{n:,}" if n else "-", style=theme.INK, justify="right"),
                         Text(bar, style=theme.RED), Text(note, style=theme.MUTED))
        ui.say(console, grid)
        _say(console, "Tokens are estimates (characters / 4). /cost for the totals, /tokens for the savings.")
        console.print()
        return session, True

    if cmd == "/memory":
        f = session.cwd / NOTES_FILE
        if rest.strip() == "edit":
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "nano" if shutil.which("nano") else "vi")
            if not f.exists():
                f.write_text(f"# {session.cwd.name}\n\nInstructions for Pal in this project.\n", encoding="utf-8")
            subprocess.call([*editor.split(), str(f)])         # noqa: S603  (your own editor, on your own file)
            session.brief = session.project.brief() if session.project is not None else session.brief
            _say(console, f"{NOTES_FILE} saved. It goes with the first request of every session.")
        elif f.is_file():
            ui.say(console, Text(f.read_text(encoding="utf-8", errors="replace")[:4000], style=theme.INK))
            _say(console, f"/memory edit opens it in your editor ({ui.short_path(f)})")
        else:
            _say(console, f"No {NOTES_FILE} yet. /init starts one from the project, /memory edit writes one yourself.")
        console.print()
        return session, True

    if cmd == "/add-dir":
        if wf is None or not rest:
            _say(console, "Use: /add-dir <folder>   its files join the context (read-only; secrets stay hidden)")
            console.print()
            return session, True
        target = Path(rest.strip().strip("\"'")).expanduser().resolve()
        if not target.is_dir():
            _say(console, f"{rest}: not a folder", theme.RED_HI)
        else:
            added, problems = wf.load([str(target)])
            _say(console, f"Added {len(added)} file(s) from {ui.short_path(target)} to the context." + (f"  {'; '.join(problems)}" if problems else ""))
        console.print()
        return session, True

    if cmd == "/permissions":
        from pal.terminal import flow_cli
        flow_cli.handle_flow("/autonomy", rest, wf, console) if wf is not None else None
        return session, True

    if cmd in ("/config", "/settings"):
        grid = Table.grid(padding=(0, 3))
        rows = [("main model", f"{session.model} ({session.provider})")]
        rows += [(f"{m} agent", f"{session.model_for(m)[1]} ({session.model_for(m)[0]})" + ("" if m in session.role_models else "  (main)")) for m in MODES]
        rows += [("helper", session.helper_model or "none"), ("custom endpoint", backend.custom_base_url() or "none"),
                 ("autonomy", wf.plan.autonomy if wf else "-"), ("token cap", f"{session.budget:,}" if session.budget else "none"),
                 ("preferences", str(backend.prefs_path())), ("project data", str(wf.store.root) if wf else "-")]
        for k, v in rows:
            grid.add_row(Text(k, style=theme.MUTED), Text(v, style=theme.WHITE))
        ui.say(console, grid)
        _say(console, "Change: /model [agent] <provider>:<model>, /endpoint <url>, /autonomy <level>, /set <setting> on|off, /budget <tokens>. Keys are never saved.")
        console.print()
        return session, True

    if cmd == "/copy":
        text = session.last_reply
        if not text:
            _say(console, "Nothing to copy yet.")
        else:
            how = copy_to_clipboard(text)
            _say(console, f"Copied the last answer ({len(text):,} characters) with {how}." if how else "No clipboard here (install xclip or wl-copy).")
        console.print()
        return session, True

    if cmd == "/export":
        if wf is None:
            return session, True
        out = Path(rest.strip()).expanduser() if rest.strip() else session.cwd / f"pal-{wf.plan.name}-{time.strftime('%Y%m%d-%H%M')}.md"
        if not out.is_absolute():
            out = session.cwd / out
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(conversation_markdown(wf), encoding="utf-8")
            _say(console, f"Saved the conversation to {ui.short_path(out)}")
        except OSError as e:
            _say(console, f"Could not save: {e.strerror or e}", theme.RED_HI)
        console.print()
        return session, True

    if cmd == "/agents":
        from pal.terminal.session import AGENT_ABOUT
        grid = Table.grid(padding=(0, 3))
        for m in MODES:
            prov, model = session.model_for(m)
            grid.add_row(Text(m.capitalize(), style=f"bold {theme.RED_HI if m == session.mode else theme.WHITE}"), Text(f"{model} ({prov})", style=theme.WHITE),
                         Text(AGENT_ABOUT.get(m, ""), style=theme.MUTED))
        if session.helper_model:
            grid.add_row(Text("helper", style=theme.MUTED), Text(session.helper_model, style=theme.WHITE), Text("Pal's own small jobs: routing, trimming, checking", style=theme.MUTED))
        ui.say(console, grid)
        _say(console, "Give an agent its own model: /model plan <provider>:<model>   tab switches agent")
        console.print()
        return session, True

    if cmd == "/simplify":
        if wf is None:
            return session, True
        from pal.terminal import flow_cli
        from pal.terminal.workflow import is_git_repo
        files: list[str] = []
        if is_git_repo(session.cwd):
            r = subprocess.run(["git", "diff", "--name-only", "HEAD~1", "HEAD"], cwd=session.cwd, capture_output=True, text=True)     # noqa: S607
            r2 = subprocess.run(["git", "diff", "--name-only"], cwd=session.cwd, capture_output=True, text=True)      # noqa: S607
            files = sorted({x for x in (r.stdout + r2.stdout).split() if (session.cwd / x).is_file()})
        files = sorted(set(files) | set(wf.plan.b.pending))[:12]
        if not files:
            _say(console, "No recently changed files found (it looks at the last commit, uncommitted changes and the sandbox).")
            console.print()
            return session, True
        mode = session.mode
        session.mode = "build"
        try:
            flow_cli.run_turn(wf, "Review these recently changed files for reuse (duplicated code), quality (naming, structure, error handling) and efficiency, "
                                  "then fix what is worth fixing without changing behaviour: " + " ".join(f"@{f}" for f in files) + (f". Focus: {rest}" if rest else ""), console)
        finally:
            session.mode = mode
        return session, True

    if cmd == "/status":
        hw = backend.hardware()
        grid = Table.grid(padding=(0, 3))
        rows = [("pal", __version__), ("folder", str(session.cwd)), ("agent", session.mode), ("model", f"{session.model_for()[1]} ({session.model_for()[0]})"),
                ("local", session.local_status or ("off (no GPU)" if not session.local else "-")), ("this computer", hw.summary())]
        if wf is not None:
            rows += [("plan", f"{wf.plan.name} / {wf.plan.branch}"), ("autonomy", wf.plan.autonomy), ("pending", f"{len(wf.plan.b.pending)} file(s)")]
        rows += [("tokens", f"{session.pal_tokens:,} used, about {max(0, session.saved):,} saved")]
        for k, v in rows:
            grid.add_row(Text(k, style=theme.MUTED), Text(v, style=theme.WHITE))
        ui.say(console, grid)
        console.print()
        return session, True

    if cmd == "/doctor":
        from pal.cli import main as tools_main
        try:
            tools_main(["doctor"])
        except SystemExit:
            pass
        console.print()
        return session, True

    if cmd == "/resume":
        if wf is None:
            return session, True
        if not rest:
            names = wf.store.names()
            for i, name in enumerate(names, 1):
                b = wf.store.load(name).b
                last = (b.convo[-1]["text"][:60] + "…") if b.convo else "no conversation yet"
                ui.say(console, Text.assemble((f"{i:>3}  {name:<16}", f"bold {theme.WHITE}" if name == wf.plan.name else theme.INK), (last, theme.MUTED)))
            _say(console, "/resume <name or number> picks one up where it stopped")
            console.print()
            return session, True
        from pal.terminal.cli import handle
        return handle(f"/cd {rest}", args, session, console, core_factory, box)

    if cmd == "/init":
        if wf is None:
            return session, True
        from pal.terminal import flow_cli
        mode = session.mode
        session.mode = "build"
        exists = (session.cwd / NOTES_FILE).is_file()
        try:
            flow_cli.run_turn(wf, (f"{'Improve' if exists else 'Write'} {NOTES_FILE} for this project: what it is, how it is laid out, how to run and test it, the "
                                   "conventions the code follows, and anything a new developer (or Pal) must not get wrong. Use the project map, the documents "
                                   f"and the files you can see. Short sections, bullet points, under 120 lines. Reply with the complete {NOTES_FILE} only."), console)
        finally:
            session.mode = mode
        _say(console, f"Review it with /diff, write it with /apply. {NOTES_FILE} then goes with the first request of every session.")
        console.print()
        return session, True

    if cmd == "/voice":
        return None                                           # the loop handles it (it needs the input box)
    return None
