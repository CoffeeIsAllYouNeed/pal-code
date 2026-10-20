"""The workflow in the terminal: one request, building a plan step by step, applying with checks, and the workflow commands."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from pal.terminal import decide, theme, ui
from pal.terminal.security import worst
from pal.terminal.session import BudgetExceeded, Change, Turn
from pal.terminal.workflow import (
    AUTONOMY,
    DEBUG_TRIES,
    SETTING_ABOUT,
    SETTINGS,
    Workflow,
    git_commit,
    is_git_repo,
    run_shell,
)

FLOW_COMMANDS: list[tuple[str, str]] = [
    ("/new", "start a new plan (fresh conversation)"), ("/plans", "list this folder's plans"), ("/cd", "switch plan: /cd <name or number>"),
    ("/current", "the current plan: branch, steps, context, pending"), ("/rename", "rename the current plan"), ("/delete-plan", "delete a plan"),
    ("/load", "add files, folders or globs to the context (-n \"note\" adds a note)"), ("/ls", "list the context with token counts"),
    ("/rm", "remove context by number, range, name or glob"), ("/clear-context", "remove all context (alias /unload)"),
    ("/continue", "build the next step of the plan (all: build every step)"), ("/diff", "show the pending changes"),
    ("/review", "review the pending changes in your browser"), ("/apply", "apply pending changes, run the checks, commit"),
    ("/reject", "drop pending changes: /reject <file>  or  /reject all"), ("/log", "every version of the sandbox"),
    ("/rewind", "go back: /rewind [n]  or  /rewind <version>"), ("/convo", "the plan's conversation"),
    ("/branches", "list branches"), ("/checkout", "switch to (or create) a branch"), ("/delete-branch", "delete a branch"),
    ("/autonomy", "none, basic, plus, semi or full"), ("/set", "one setting: /set auto-exec on"), ("/run", "run a command in this folder (asks first)"),
    ("/debug", "run a command and let the Build agent fix it until it passes: /debug npm test"),
]


def spin(session, console: Console, label: str, fn: Callable):
    """Run fn() with a spinner on the step Pal is on, and a line for every step it finished (what it did and how long it took)."""
    from pal import steps
    from pal.terminal.keys import EscToStop
    with console.status(Text(label, style=theme.MUTED), spinner=theme.SPINNER, spinner_style=theme.RED) as status, EscToStop():
        done = threading.Event()
        t0 = time.monotonic()
        now = {"label": label.rstrip("\u2026. ")}

        def tick() -> None:
            while not done.wait(0.25):
                line = Text(f"{now['label']}\u2026  {time.monotonic() - t0:.0f}s", style=theme.MUTED)
                line.append("   esc to stop", style=theme.MUTED)
                status.update(line)

        def on_step(event: str, key: str, text: str) -> None:
            if event == "begin":
                now["label"] = text
            else:
                ui.say(console, Text.assemble(("  \u2713 ", theme.RED_HI), (text, theme.INK)))
                now["label"] = "Working"
        threading.Thread(target=tick, daemon=True).start()
        with steps.listen(on_step):
            try:
                return fn()
            finally:
                done.set()


def safely(console: Console, fn: Callable):
    """Run a model request; report a stop, a budget or an error in one line instead of a traceback. Returns the result or None."""
    try:
        return fn()
    except BudgetExceeded as e:
        ui.say(console, Text(f"Stopped: {e}", style=theme.RED_HI))
    except KeyboardInterrupt:
        ui.say(console, Text("Stopped.", style=theme.MUTED))
    except Exception as e:  # noqa: BLE001  (a model that cannot be reached, a missing key, a full context ...)
        ui.say(console, Text(f"Something went wrong: {type(e).__name__}: {str(e)[:300]}", style=theme.RED_HI))
    console.print()
    return None


DOING = {"build": "Pal is building…", "plan": "Pal is planning…", "ask": "Pal is thinking…"}


def run_turn(wf: Workflow, text: str, console: Console) -> None:
    """One request with the current agent, then what follows from it: pending changes to review, or a plan to build."""
    s = wf.session
    ui.show_user(console, text)
    result = safely(console, lambda: spin(s, console, DOING.get(s.mode, "Pal is working…"), lambda: wf.tell(text)))
    if result is None:
        return
    ui.show_reply(console, result, s)
    if s.mode == "ask" and result.questions:
        answers = decide.ask_all(console, result.questions)
        console.print()
        if answers is None:
            ui.say(console, Text("Skipped. Answer in your own words whenever you like.", style=theme.MUTED))
            console.print()
        else:
            run_turn(wf, decide.answers_message(result.questions, answers), console)
        return
    if s.mode == "build":
        after_build(wf, result, console)
    elif s.mode == "plan" and wf.plan.b.steps:
        show_steps(wf, console)
        if wf.plan.setting("auto-continue") and ui.confirm(console, f"Build the {len(wf.plan.b.steps)} steps now, into the sandbox?"):
            continue_flow(wf, console, every=True)
        else:
            ui.say(console, Text("/continue builds the next step  ·  /continue all builds every step", style=theme.MUTED))
            console.print()


def after_build(wf: Workflow, result: Turn, console: Console) -> None:
    if result.changes:
        ui.show_changes(console, result.changes)
    if wf.plan.b.pending and wf.plan.setting("auto-apply"):          # the default: written into the folder at once (checks, rollback, commit)
        apply_flow(wf, console, ask=False)
        return
    pending_line(wf, console)


def pending_line(wf: Workflow, console: Console) -> None:
    """Changes waiting in the sandbox: the files, then what can be done with them."""
    from rich import box as rich_box
    n, cmds = len(wf.plan.b.pending), wf.plan.b.commands
    if not n:
        return
    ui.say(console, Text(f"\U0001F9D0 {n} file{'s' if n != 1 else ''} {'have' if n != 1 else 'has'} pending changes", style=f"bold {theme.RED_HI}"))
    for path in sorted(wf.plan.b.pending):
        ui.say(console, Text(f"\u2022 \U0001F4C4 {path}", style=theme.WHITE))
    if cmds:
        console.print()
        ui.say(console, Text("\U0001F680 Commands pending", style=f"bold {theme.RED_HI}"))
        for c in cmds:
            ui.say(console, Text(f"\u2022 {c}", style=theme.INK))
    console.print()
    table = Table(box=rich_box.ASCII, show_header=False, padding=(0, 1))
    table.add_column(style=f"bold {theme.RED_HI}")
    table.add_column(style=theme.WHITE)
    table.add_column(style=theme.INK)
    for key, name, about in (("/diff", "diff", "Review the changes here"), ("/review", "review", "Review the changes in the browser"),
                             ("/apply", "apply", "Apply all pending changes"), ("/reject", "reject", "Reject some or all pending changes"),
                             ("/rewind", "rewind", "Go back to an earlier state")):
        table.add_row(key, name, about)
    ui.say(console, table)
    console.print()


def show_steps(wf: Workflow, console: Console) -> None:
    grid = Table.grid(padding=(0, 1))
    for i, step in enumerate(wf.plan.b.steps, 1):
        mark = "✓" if i <= wf.plan.b.done else "•"
        grid.add_row(Text(mark, style=theme.RED_HI if i <= wf.plan.b.done else theme.MUTED), Text(f"{i}.", style=theme.MUTED), Text(step, style=theme.INK))
    ui.say(console, Text(f"Plan saved: {len(wf.plan.b.steps)} steps ({wf.plan.b.done} built)", style=f"bold {theme.WHITE}"))
    ui.say(console, Padding(grid, (0, 0, 0, 2)))


def continue_flow(wf: Workflow, console: Console, every: bool = False) -> None:
    """Build the next step, or every remaining step, into the sandbox."""
    s = wf.session
    if wf.next_step() is None:
        ui.say(console, Text("Every step is built. Make a new plan with the Plan agent, or ask Build for more." if wf.plan.b.steps else
                             "There is no plan yet: switch to the Plan agent (tab) and describe the task.", style=theme.MUTED))
        console.print()
        return
    while True:
        nxt = wf.next_step()
        if nxt is None:
            break
        i, step = nxt
        ui.say(console, Text.assemble((f"Step {i} of {len(wf.plan.b.steps)}  ", f"bold {theme.WHITE}"), (step, theme.INK)))
        label = f"Pal is building step {i}…"
        result = safely(console, lambda label=label: spin(s, console, label, wf.build_step))
        if result is None:
            break
        if result.changes:
            ui.say(console, Text("  " + "  ".join(("+ " if c.is_new else "~ ") + c.path for c in result.changes), style=theme.INK))
        for path, why in result.refused:
            ui.say(console, Text(f"  refused {path}: {why}", style=theme.RED_HI))
        if not every:
            break
    console.print()
    pending_line(wf, console)
    if wf.plan.b.pending and wf.plan.setting("auto-apply"):
        apply_flow(wf, console, ask=False)


def show_diff(wf: Workflow, console: Console) -> None:
    diffs = wf.diffs()
    if not diffs:
        ui.say(console, Text("Nothing is pending.", style=theme.MUTED))
        console.print()
        return
    changes = [Change(path, "", "", new, lines, findings) for path, new, lines, findings in diffs]
    ui.show_changes(console, changes, limit=200)
    console.print()


def broken_pending(wf: Workflow) -> list:
    """Pending files that do not parse (a fix that never came, or a file edited by hand into the sandbox)."""
    from pal.terminal.syntax import check
    return [p for path, new in wf.plan.b.pending.items() if (p := check(path, new))]


def parses_or_confirmed(wf: Workflow, console: Console) -> bool:
    """True when every pending file parses, or the person accepts the ones that do not (the default is no)."""
    bad = broken_pending(wf)
    if not bad:
        return True
    for prob in bad:
        ui.say(console, Text(f"  ✗ {prob}", style=theme.RED_HI))
    if ui.confirm(console, f"{len(bad)} pending file{'s' if len(bad) != 1 else ''} do{'es' if len(bad) == 1 else ''} not parse. Apply anyway?"):
        return True
    ui.say(console, Text("  Nothing applied. Fix it with the Build agent, or drop it with /reject <file>.", style=theme.MUTED))
    console.print()
    return False


def apply_flow(wf: Workflow, console: Console, ask: bool = True, yes: bool = False) -> None:
    """Write the pending changes, run the checks the person approves, roll back on failure (and auto-debug), and commit."""
    s = wf.session
    pending = wf.plan.b.pending
    if not pending:
        ui.say(console, Text("Nothing is pending.", style=theme.MUTED))
        console.print()
        return
    if not parses_or_confirmed(wf, console):
        return
    level = worst([f for _, _, _, fs in wf.diffs() for f in fs])
    n = len(pending)
    if level == "critical" and not ui.confirm(console, f"A critical security problem was found (see /diff). Apply {n} file{'s' if n != 1 else ''} anyway?"):
        ui.say(console, Text("  Nothing applied.", style=theme.MUTED))
        return
    if level != "critical" and ask and not yes and not wf.plan.setting("auto-apply") and not ui.confirm(console, f"Apply {n} file{'s' if n != 1 else ''} to {ui.short_path(s.cwd)}?"):
        ui.say(console, Text("  Nothing applied. The changes stay in the sandbox.", style=theme.MUTED))
        return

    def run_command(cmd: str):
        if not wf.plan.setting("auto-exec") and not ui.confirm(console, f"Run  {cmd}  in {ui.short_path(s.cwd)}?"):
            return None, ""
        ui.say(console, Text(f"  $ {cmd}", style=theme.WHITE))
        ok, out = spin(s, console, "Running…", lambda: run_shell(cmd, s.cwd))
        tail = "\n".join(out.strip().splitlines()[-12:])
        if tail:
            ui.say(console, Padding(Text(ui.clean_terminal(tail), style=theme.MUTED), (0, 0, 0, 4)))
        ui.say(console, Text("  ✓ passed" if ok else "  ✗ failed", style=theme.RED_HI if not ok else f"bold {theme.WHITE}"))
        return ok, out

    choice = {"anyway": False}

    def debug(cmd: str, output: str) -> bool:
        """A command failed and every file was put back. Fix it with the Build agent: it sees the error and this computer's system, and may
        change the files, change the command, or add the install step a missing tool needs. Asks first, unless auto-debug is on."""
        import platform
        ui.say(console, Text.assemble(("\U0001F6A8 ", ""), ("Commands failed", f"bold {theme.RED_HI}")))
        if not wf.plan.setting("auto-debug"):
            pick = ui.select(console, "What do you want to do?", ["Debug and retry once", "Debug in full auto mode", "Roll back the changes", "Apply the changes anyway"])
            if pick == "Debug in full auto mode":
                for name in ("auto-debug", "auto-exec", "auto-apply"):
                    wf.plan.custom[name] = True
                wf.save()
                ui.say(console, Text("\u2705 Full auto mode on: Pal fixes, applies and runs the checks by itself (/set auto-debug off to stop).", style=theme.INK))
            elif pick == "Apply the changes anyway":
                choice["anyway"] = True
                return False
            elif pick != "Debug and retry once":
                return False
        before_cmds = list(wf.plan.b.commands)
        mode = s.mode
        s.mode = "build"
        system = f"{platform.system()} {platform.release()}"
        try:
            result = safely(console, lambda: spin(s, console, "Debugging", lambda: wf.tell(
                f"After applying your changes, the command `{cmd}` failed on this computer ({system}) with:\n```\n{output}\n```\n"
                "Fix the cause. Reply with the complete files to change. If the command itself is wrong for this computer, or needs a tool that is "
                "not installed (for example 'is not recognized' or 'command not found'), change the command under ### COMMANDS to one that works "
                "here, or put the command that installs the tool first (pip, npm, winget, brew or apt, whichever fits this system).")))
        finally:
            s.mode = mode
        return bool(result and (result.changes or wf.plan.b.commands != before_cmds))

    def commit_message(files: list[str]) -> str:
        label = next((v.label for v in reversed(wf.plan.b.versions) if not v.label.startswith(("applied", "rejected"))), "Pal changes")
        try:
            reply = s.client.ask(f"Write a git commit message for this change: {label}. Files: {', '.join(files)}. One line under 72 characters, "
                                 "imperative mood, then optionally a blank line and up to 3 short bullet lines. Reply with the message only.")
            text = ui.clean_terminal(reply.text).strip().strip("`").strip()
            return text[:600] or label
        except Exception:  # noqa: BLE001
            return label[:72]

    res = wf.apply(run_command, debug, commit_message)
    if res.rolled_back and choice["anyway"]:               # apply the changes without running the failing check
        keep = wf.plan.custom.get("can-exec")
        wf.plan.custom["can-exec"] = False
        try:
            res = wf.apply(run_command, None, commit_message)
        finally:
            if keep is None:
                wf.plan.custom.pop("can-exec", None)
            else:
                wf.plan.custom["can-exec"] = keep
    if res.written and not res.rolled_back:
        n = len(res.written)
        ui.say(console, Text.assemble(("\U0001F680 ", ""), (f"Applied {n} file{'s' if n != 1 else ''}", f"bold {theme.RED_HI}"),
                                      (f"  to {ui.short_path(s.cwd)}", theme.INK)))
    if res.rolled_back:
        ui.say(console, Text(f"\u274C {res.failed_command} still fails: every file was put back as it was. The changes are still pending.", style=theme.RED_HI))
        ui.say(console, Text("  /apply tries again  \u00B7  /diff shows the changes  \u00B7  /reject drops them", style=theme.MUTED))
    elif res.commit:
        ui.say(console, Text.assemble(("\u2705 ", ""), ("Committed ", theme.INK), (str(res.commit), theme.WHITE), ("  \u00B7  /rewind undoes it", theme.MUTED)))
    elif res.written and is_git_repo(s.cwd) and not wf.plan.setting("auto-commit"):
        ui.say(console, Text("  Not committed (auto-commit is off: /set auto-commit on).", style=theme.MUTED))
    console.print()


def debug_flow(wf: Workflow, cmd: str, console: Console, tries: int = DEBUG_TRIES) -> bool:
    """/debug <command>: run it; while it fails, show the Build agent the output, apply its fix (asking first, unless auto-apply is on) and run it
    again, at most `tries` fixes. If it still fails, offer to put every file back as it was. True when the command passes in the end."""
    s = wf.session
    say = lambda text, style=theme.MUTED: ui.say(console, Text(text, style=style))  # noqa: E731
    before: dict[str, str | None] = {}                  # every file a fix touched, as it was before /debug
    written: list[str] = []
    for attempt in range(tries + 1):
        say(f"  $ {cmd}", theme.WHITE)
        ok, out = spin(s, console, "Running…", lambda: run_shell(cmd, s.cwd))
        tail = "\n".join(out.strip().splitlines()[-12:])
        if tail:
            ui.say(console, Padding(Text(ui.clean_terminal(tail), style=theme.MUTED), (0, 0, 0, 4)))
        if ok:
            say("  ✓ passes" + (f" after {attempt} fix{'es' if attempt != 1 else ''}" if attempt else ""), f"bold {theme.WHITE}")
            if written and wf.plan.setting("auto-commit") and is_git_repo(s.cwd):
                sha = git_commit(s.cwd, sorted(set(written)), f"Fix: {cmd}"[:72])
                if sha:
                    say(f"  committed {sha}", f"bold {theme.WHITE}")
            console.print()
            return True
        say("  ✗ fails", theme.RED_HI)
        if attempt == tries:
            break
        mode, s.mode = s.mode, "build"
        try:
            ask = (f"Running `{cmd}` in the project fails with:\n```\n{ui.clean_terminal(out.strip())[-3000:]}\n```\n"
                   "Find the cause and fix it. Reply with the complete files to change.")
            result = safely(console, lambda q=ask: spin(s, console, "Pal is debugging…", lambda: wf.tell(q)))
        finally:
            s.mode = mode
        if result is None:
            return False
        if result.text:
            ui.show_reply(console, result, s)
        if not result.changes:
            say("  The Build agent proposed no file change, so /debug stops here.")
            console.print()
            return False
        if not parses_or_confirmed(wf, console):
            return False
        files = ", ".join(sorted(wf.plan.b.pending))
        if not wf.plan.setting("auto-apply") and not ui.confirm(console, f"Apply {files} and run  {cmd}  again?"):
            say("  Stopped. The fix stays in the sandbox: /diff, /apply.")
            console.print()
            return False
        for path in wf.plan.b.pending:
            before.setdefault(path, wf.disk(path))
        res = wf.apply(lambda c: (None, ""))            # write the fix; the command is run by this loop, not by apply
        written += res.written
    say(f"  Still failing after {tries} fix{'es' if tries != 1 else ''}.", theme.RED_HI)
    if before and ui.confirm(console, f"Put the {len(before)} file(s) /debug changed back as they were?"):
        wf._roll_back(before)
        say("  Put back.")
    console.print()
    return False


def handle_flow(cmd: str, rest: str, wf: Workflow, console: Console) -> bool:
    """The workflow commands. True when `cmd` was one of them."""
    say = lambda text, style=theme.MUTED: ui.say(console, Text(text, style=style))  # noqa: E731
    b = wf.plan.b
    try:
        if cmd == "/plans":
            for i, name in enumerate(wf.store.names(), 1):
                say(f"{i:>3}  {name}" + ("   ◀ current" if name == wf.plan.name else ""), theme.WHITE if name == wf.plan.name else theme.INK)
        elif cmd == "/current":
            say(f"Plan {wf.plan.name}  ·  branch {wf.plan.branch}  ·  autonomy {wf.plan.autonomy}", f"bold {theme.WHITE}")
            say(f"{len(b.steps)} steps ({b.done} built)  ·  {len(b.context)} files in context  ·  {len(b.pending)} pending  ·  {len(b.convo) // 2} messages")
            if b.steps:
                show_steps(wf, console)
        elif cmd == "/rename":
            wf.rename(rest)
            say(f"Renamed to {rest}.")
        elif cmd == "/delete-plan":
            wf.delete(rest or wf.plan.name)
            say(f"Deleted. Now on plan {wf.plan.name}.")
        elif cmd == "/load":
            note = ""
            if rest.startswith("-n "):
                note, rest = rest[3:].strip().strip("\"'"), ""
            from pal.terminal.cli import split_paths
            added, problems = wf.load(split_paths(rest), note)
            for a in added:
                say(f"  + {a}", theme.INK)
            for p in problems:
                say(f"  {p}", theme.RED_HI)
            say(f"{len(b.context)} files in context. They go with every request; a file the model already has is not sent again.")
        elif cmd == "/ls":
            rows = wf.context_rows()
            if not rows:
                say("The context is empty. Add files with  /load <file or folder or glob>.")
            grid = Table.grid(padding=(0, 2))
            for i, name, tokens, problem in rows:
                grid.add_row(Text(f"{i}", style=theme.MUTED), Text(name, style=theme.WHITE), Text(f"{tokens:,} tok", style=theme.INK), Text(problem, style=theme.RED_HI))
            if rows:
                ui.say(console, grid)
                say(f"total about {sum(r[2] for r in rows):,} tokens")
        elif cmd == "/rm":
            from pal.terminal.cli import split_paths
            gone = wf.remove(split_paths(rest))
            say("removed: " + (", ".join(gone) or "nothing matched"))
        elif cmd in ("/clear-context", "/unload"):
            say(f"Removed {wf.clear()} item(s) from the context.")
        elif cmd == "/continue":
            continue_flow(wf, console, every=rest.strip() == "all")
            return True
        elif cmd == "/diff":
            show_diff(wf, console)
            return True
        elif cmd == "/review":
            from pal.terminal.review import open_review
            path = open_review(wf)
            say(f"Opened the review in your browser ({path}).")
        elif cmd == "/apply":
            apply_flow(wf, console, yes=rest.strip() in ("-y", "--yes"))
            return True
        elif cmd == "/reject":
            gone = wf.reject(None if rest.strip() in ("all", "--all") else rest.split())
            say("dropped: " + (", ".join(gone) or "nothing (give a pending file, or all)") + ("   (/rewind brings them back)" if gone else ""))
        elif cmd == "/log":
            vs = wf.log()
            if not vs:
                say("No versions yet: the Build agent's changes appear here.")
            for v in reversed(vs[-30:]):
                say(f"  {v.id}  {time.strftime('%H:%M', time.localtime(v.t))}  {len(v.pending):>2} pending  {v.label}", theme.INK)
        elif cmd == "/rewind":
            n = wf.rewind(int(rest) if rest.strip().isdigit() else 1, to="" if not rest.strip() or rest.strip().isdigit() else rest.strip())
            say(f"Went back {n} version(s). {len(wf.plan.b.pending)} file(s) pending.")
        elif cmd == "/convo":
            for m in b.convo[-20:]:
                ui.say(console, Text.assemble((f"{m['role']} ({m['agent']}): ", theme.RED_HI if m["role"] == "you" else theme.WHITE), (ui.clean_terminal(m["text"][:500]), theme.INK)))
        elif cmd == "/branches":
            for name in wf.plan.branches:
                say(f"  {name}" + ("   ◀ current" if name == wf.plan.branch else ""), theme.WHITE if name == wf.plan.branch else theme.INK)
        elif cmd == "/checkout":
            created = wf.checkout(rest.strip())
            say(f"{'Created and switched to' if created else 'Switched to'} branch {rest.strip()}.")
        elif cmd == "/delete-branch":
            wf.delete_branch(rest.strip())
            say(f"Deleted branch {rest.strip()}.")
        elif cmd == "/autonomy":
            level = rest.strip().lower()
            if level in AUTONOMY:
                wf.plan.autonomy, wf.plan.custom = level, {}
                wf.save()
            say(f"Autonomy: {wf.plan.autonomy}" + ("" if level in AUTONOMY else "   (choose none, basic, plus, semi or full)"), f"bold {theme.WHITE}")
            for name in SETTINGS:
                say(f"  {'on ' if wf.plan.setting(name) else 'off'}  {name:<18} {SETTING_ABOUT[name]}", theme.INK if wf.plan.setting(name) else theme.MUTED)
            if wf.plan.autonomy == "full":
                say("Full runs commands and applies changes without asking: use it on a clean git branch.", theme.RED_HI)
        elif cmd == "/set":
            name, _, value = rest.partition(" ")
            if name not in SETTINGS or value.strip().lower() not in ("on", "off", "true", "false"):
                say(f"Use: /set <setting> on|off   settings: {', '.join(SETTINGS)}")
            else:
                wf.plan.custom[name] = value.strip().lower() in ("on", "true")
                wf.save()
                say(f"{name}: {'on' if wf.plan.setting(name) else 'off'}")
        elif cmd == "/debug":
            if not rest.strip():
                say("Use: /debug <command>   for example  /debug npm test   or  /debug python -m pytest -q")
            elif wf.plan.setting("can-exec") or ui.confirm(console, f"Commands are off for this plan (can-exec). Run  {rest.strip()}  anyway?"):
                debug_flow(wf, rest.strip(), console)
        elif cmd == "/run":
            if not rest.strip():
                say("Use: /run <command>")
            elif ui.confirm(console, f"Run  {rest.strip()}  in {ui.short_path(wf.session.cwd)}?"):
                ok, out = spin(wf.session, console, "Running…", lambda: run_shell(rest.strip(), wf.session.cwd))
                ui.say(console, Padding(Text(ui.clean_terminal("\n".join(out.strip().splitlines()[-30:])), style=theme.INK), (0, 0, 0, 2)))
                say("✓ exit 0" if ok else "✗ failed", theme.WHITE if ok else theme.RED_HI)
        else:
            return False
    except ValueError as e:
        say(str(e), theme.RED_HI)
    console.print()
    return True
