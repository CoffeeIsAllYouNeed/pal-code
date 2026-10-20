"""After a reply: the pending-changes menu, apply (with the commands, rollback, debugging and the commit), diffs, reject and debug."""
from __future__ import annotations

import difflib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from pal.terminal.repl import term
from pal.terminal.repl.term import BOLD, WHITE, Exit, accent, c, out, spinner
from pal.terminal.workflow import _change, git_commit, is_git_repo

APPLY = "_apply.sh"


def sorted_paths(wf) -> list[str]:
    """The pending paths, sorted, with the commands last (as "_apply.sh")."""
    paths = sorted(wf.plan.b.pending)
    if wf.plan.b.commands:
        paths.append(APPLY)
    return paths


# -- the menu after a reply --------------------------------------------------------------------------------------------------------------
OPTIONS = [  # key, command, description, colour
    ("d", "diff ui", "Review diffs in browser UI", term.HI_CYAN),
    ("g", "git diff format", "Review diffs in git diff format", term.HI_CYAN),
    ("r", "reject", "Reject some or all pending changes", term.HI_RED),
    ("a", "apply", "Apply all pending changes", term.HI_GREEN),
    ("f", "full auto", "Apply and debug in full auto mode", term.HI_YELLOW),
]


def show_hotkey_menu(diffs: list[str]) -> None:
    has_script = APPLY in diffs
    n = len(diffs) - (1 if has_script else 0)
    if n > 0:
        out(c(f"{BOLD};{term.HI_GREEN}", f"\U0001F9D0 {n} {'file has' if n == 1 else 'files have'} pending changes"))
        for d in diffs:
            if d != APPLY:
                out(f"• \U0001F4C4 {d}")
        out()
    if has_script:
        out(c(f"{BOLD};{term.HI_YELLOW}", "\U0001F680 Commands pending"))
        out()
    rows = [[c(f"{col};{BOLD}", f"({k})"), cmd, desc] for k, cmd, desc, col in OPTIONS if k != "f" or has_script]
    out(term.table(rows, align_all="L"), end="")
    prompt_line()


def prompt_line(end_text: str = "to exit menu/keep iterating>") -> None:
    q = f"{term.HI_MAGENTA};{BOLD}"
    out(c(q, "Press a hotkey,") + " " + c(f"{WHITE};{BOLD}", "↓") + " " + c(q, "to select, or") + " " + c(f"{WHITE};{BOLD}", "enter") + " "
        + c(q, end_text), end="")


def changes_menu(ctx, diffs: list[str]) -> None:
    """The pending-changes menu: d, g, r, a, f, ↓ for a list, Enter (or q) to keep iterating."""
    while True:
        show_hotkey_menu(diffs)
        try:
            key = term.get_key()
        except Exception as e:  # noqa: BLE001
            out(f"\nError getting key: {e}")
            continue
        if key == "down":
            descs = [d for _, _, d, _ in OPTIONS] + ["Exit menu"]
            out()
            try:
                pick = term.select("Select an action", descs)
            except term.Interrupted:
                return
            key = next((k for k, _, d, _ in OPTIONS if d == pick), "q")
        else:
            out()
        if key == "d":
            out()
            diff_ui(ctx.wf, from_tell_menu=True)
            out()
            continue
        if key == "g":
            out()
            diff_git(ctx.wf)
            out()
            continue
        if key == "r":
            out()
            try:
                reject(ctx.wf, [])
            except Exit:
                pass
            out()
            diffs = sorted_paths(ctx.wf)
            if not diffs:
                return
            continue
        if key == "a":
            out("(a)")
            try:
                apply_cmd(ctx)
            except Exit:
                pass
            out()
            return
        if key == "f":
            out("(f)")
            out(c(f"{term.HI_YELLOW};{BOLD}", "⚠️  Full auto mode allows automatic apply, execution, and multiple rounds of debugging without review."))
            out()
            try:
                apply_cmd(ctx, full=True)
            except Exit:
                pass
            out()
            return
        if key in ("q", "enter"):
            return
        out("\nInvalid hotkey")


def after_reply(ctx, auto_apply: bool, skip_menu: bool) -> None:
    """What follows a finished tell: apply by itself (auto-apply), or the menu when changes are pending."""
    wf = ctx.wf
    if auto_apply:
        cfg = wf.plan.config()
        apply_plan(ctx, auto_commit=cfg["auto-commit"], no_commit=not cfg["auto-commit"], auto_exec=cfg["auto-exec"] or cfg["auto-debug"],
                   no_exec=not cfg["can-exec"], auto_debug=cfg["auto-debug-tries"] if cfg["auto-debug"] else 0)
        return
    if skip_menu:
        return
    spinner.start("")
    time.sleep(0.5)
    diffs = sorted_paths(wf)
    spinner.stop()
    out()
    if diffs:
        changes_menu(ctx, diffs)


# -- apply --------------------------------------------------------------------------------------------------------------------------------
def apply_cmd(ctx, full: bool = False, commit: bool | None = None, no_exec: bool | None = None, auto_exec: bool | None = None,
              debug: int | None = None, skip: bool = False) -> None:
    """\\apply: the flags come from the plan's config unless given."""
    wf = ctx.wf
    if full:
        spinner.start("")
        old = wf.plan.config()
        changed = wf.plan.autonomy != "full"
        wf.plan.set_auto("full")
        wf.save()
        spinner.stop()
        if changed:
            from pal.terminal.repl.cmds import auto_mode_table
            out(auto_mode_table(old), end="")
    cfg = wf.plan.config()
    skip_commit = cfg["skip-commit"]
    auto_commit = cfg["auto-commit"] if commit is None else commit
    no_exec = (not cfg["can-exec"]) if no_exec is None else no_exec
    auto_exec = cfg["auto-exec"] if auto_exec is None else auto_exec
    tries = (cfg["auto-debug-tries"] if cfg["auto-debug"] else 0) if debug is None else debug
    if tries and no_exec:
        term.fail("--debug can't be used with --no-exec")
    apply_plan(ctx, auto_commit=auto_commit, no_commit=(skip_commit or skip) and not commit, auto_exec=auto_exec or tries > 0, no_exec=no_exec,
               auto_debug=tries)


class Rollback:
    def __init__(self) -> None:
        self.before: dict[str, str | None] = {}

    def has_changes(self) -> bool:
        return bool(self.before)


def rollback(wf, rb: Rollback, say: bool = True) -> None:
    root = wf.session.cwd
    for path, old in rb.before.items():
        p = root / path
        try:
            if old is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(old, encoding="utf-8", newline="")
        except OSError:
            pass
    if say:
        out("\U0001F6AB Rolled back all changes")


def _write_files(wf, pending: dict[str, str]) -> tuple[list[str], Rollback, list[str]]:
    """Write the pending files. (updated paths, what to restore, problems)."""
    s = wf.session
    rb = Rollback()
    changes = []
    for path, new in pending.items():
        old = wf.disk(path)
        if old == new.replace("\\`\\`\\`", "```"):
            continue
        rb.before[path] = old
        changes.append(_change(path, old, new.replace("\\`\\`\\`", "```")))
    written = s.apply(changes) if changes else []
    ok = [p for p, status in written if status == "written"]
    problems = [f"{p}: {status}" for p, status in written if status != "written"]
    rb.before = {p: rb.before[p] for p in ok}
    return ok, rb, problems


def apply_plan(ctx, auto_confirm: bool = True, auto_commit: bool = False, no_commit: bool = False, auto_exec: bool = False, no_exec: bool = False,
               auto_debug: int = 0, exec_command: str = "", attempt: int = 0) -> None:
    wf = ctx.wf
    spinner.start("")
    pending = wf.plan.b.pending
    commands = list(wf.plan.b.commands)
    has_exec = bool(commands) and not exec_command
    if not pending and not has_exec:
        spinner.stop()
        out("\U0001F937‍♂️ No changes to apply")
        return
    if pending:
        from pal.terminal.security import worst
        level = worst([f for _, _, _, fs in wf.diffs() for f in fs])
        if level == "critical":
            spinner.stop()
            out(c(f"{term.HI_RED};{BOLD}", "\U0001F6A8 A critical security problem was found in the pending changes (\\diff shows it)"))
            if not term.confirm("Apply changes anyway?"):
                out("Apply plan canceled")
                return
            spinner.resume()
        if not auto_confirm:
            spinner.stop()
            n = len(pending)
            if not term.confirm(f"Apply changes to {n} file{'s' if n > 1 else ''}?"):
                raise Exit(0)
            spinner.resume()
        if has_exec and not no_exec:
            spinner.stop()
            out("\U0001F504 Tentatively applying changes")
            spinner.resume()
    updated, rb, problems = _write_files(wf, pending)
    for p in problems:
        spinner.stop()
        term.simple_error(f"Failed to apply {p}")
    after = {p: pending[p] for p in updated}

    def on_success() -> None:
        spinner.start("")
        if not updated:
            wf.mark_applied([], rb.before, after)
            spinner.stop()
            out("✅ Applied changes, but no files were updated")
            return
        commit_err, sha, msg = "", "", ""
        if is_git_repo(wf.session.cwd) and not no_commit:
            spinner.stop()
            ok = auto_commit
            if not ok:
                out("✏️  Pal can commit these updates with an automatically generated message.")
                out()
                ok = term.confirm("Commit Pal updates now?")
            if ok:
                msg = commit_summary(wf)
                sha = git_commit(wf.session.cwd, updated, "\U0001F916 Pal → " + msg)
                if not sha:
                    commit_err = "git refused the commit"
        wf.mark_applied(updated, rb.before, after, sha, msg)
        spinner.stop()
        n = len(updated)
        out(f"✅ Applied changes, {n} file{'s' if n > 1 else ''} updated")
        for f in updated:
            out(f" • \U0001F4C4 {f}")
        if commit_err:
            out()
            term.simple_error(f"Failed to commit changes: {commit_err}")

    if has_exec and not no_exec:
        handle_script(ctx, "\n".join(commands), rb, on_success, auto_exec, auto_debug, attempt,
                      dict(auto_confirm=auto_confirm, auto_commit=auto_commit, no_commit=no_commit, no_exec=no_exec, exec_command=exec_command, auto_exec=auto_exec))
    else:
        on_success()


def commit_summary(wf) -> str:
    """A one-line commit title for what is being applied, written by the helper model (a single call, outside the conversation)."""
    from pal.llm_wrapper.components.base import LLMSyscall
    label = next((v.label for v in reversed(wf.plan.b.versions) if not v.label.startswith(("applied", "rejected"))), "")
    client = wf.session.client
    core = getattr(client, "helper", None) or getattr(client, "core", None)
    if core is None:
        return label[:72]
    ask = (f"Write a git commit message title for this change: {label[:600]}\nFiles: {', '.join(sorted(wf.plan.b.pending))}\n"
           "One line under 72 characters, imperative mood. Reply with the line only.")
    try:
        reply = core.chat(LLMSyscall(messages=[{"role": "system", "content": "You write succinct one-line git commit message titles."},
                                               {"role": "user", "content": ask}], temperature=0.0, max_tokens=60))
        from pal.terminal.session import strip_thinking
        text = strip_thinking(reply.text).strip().strip("`\"'").strip()
        line = text.splitlines()[0].strip() if text else ""
        return line[:100] or label[:72]
    except Exception:  # noqa: BLE001
        return label[:72]


def _render_bash(script: str) -> str:
    import io

    from rich.console import Console
    from rich.markdown import Markdown
    buf = io.StringIO()
    Console(file=buf, width=min(term.term_width(), 80), force_terminal=term.colours_on(), color_system="truecolor").print(
        Markdown(f"```bash\n{script}\n```", code_theme="monokai"))
    return buf.getvalue().strip()


def handle_script(ctx, script: str, rb: Rollback, on_success: Callable[[], None], auto_exec: bool, auto_debug: int, attempt: int, flags: dict) -> None:
    wf = ctx.wf
    spinner.stop()
    out(accent("\U0001F680 Commands to execute \U0001F447"))
    out(_render_bash(script))
    ok = True if auto_exec else term.confirm("Execute now?")
    if not ok:
        if rb.has_changes():
            pick = term.select("Skipping execution. Apply file changes or roll back?", ["Apply file changes", "Roll back file changes"])
            if pick == "Roll back file changes":
                rollback(wf, rb)
                raise Exit(0)
            on_success()
        else:
            out("\U0001F645‍♂️ Skipped execution")
            out("\U0001F937‍♂️ No changes to apply")
        return
    out(c(f"{term.HI_YELLOW};{BOLD}", "\U0001F449 For long-running commands, use ctrl+c to exit"))
    out(accent("\U0001F680 Executing... output below \U0001F447"))
    out()
    success, status, output, interrupted = run_script(script, wf.session.cwd)
    if interrupted:
        out(c(f"{term.HI_YELLOW};{BOLD}", "\U0001F449 Execution interrupted"))
        yes, cancel = term.confirm_cancel("Did the commands succeed?")
        if cancel:
            if rb.has_changes():
                rollback(wf, rb)
            raise Exit(0)
        success = yes
    if not success:
        out()
        out(c(f"{term.HI_RED};{BOLD}", "\U0001F6A8 Commands failed"))
        on_exec_fail(ctx, status, output, attempt, rb, on_success, auto_debug, flags)
        return
    out()
    out("✅ Commands succeeded")
    on_success()


def _script_lines(script: str) -> list[str]:
    keep = []
    for line in script.splitlines():
        t = line.strip()
        if t.startswith("#!/") or t.startswith("set -") or t.endswith("pipefail") or t.startswith("trap"):
            continue
        keep.append(line)
    return keep


def run_script(script: str, cwd: Path) -> tuple[bool, int, str, bool]:
    """Run the commands in the project folder, output shown as it comes. (success, exit status, output, interrupted)."""
    lines = _script_lines(script)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if os.name == "nt":
        body = "\r\n".join(["@echo off"] + [f"{ln}\r\nif errorlevel 1 exit /b %errorlevel%" for ln in lines if ln.strip()])
        fd, path = tempfile.mkstemp(suffix=".cmd", prefix="pal_apply_")
        os.write(fd, body.encode("utf-8"))
        os.close(fd)
        args = ["cmd.exe", "/d", "/c", path]
        kw: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}       # type: ignore[attr-defined]
    else:
        shell = os.environ.get("SHELL") or "/bin/bash"
        if not shutil.which(shell) and not Path(shell).exists():
            shell = "/bin/sh"
        head = "#!/bin/zsh" if shell.endswith("zsh") else "#!/bin/bash"
        body = head + "\n\nset -euo pipefail\n" + "\n".join(lines) + "\n"
        fd, path = tempfile.mkstemp(suffix=".sh", prefix="pal_apply_")
        os.write(fd, body.encode("utf-8"))
        os.close(fd)
        os.chmod(path, 0o755)  # noqa: S103
        args = [shell, "-c", path]                      # the script's own first line picks bash (or zsh)
        kw = {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp}
    output: list[str] = []
    interrupted = False
    try:
        proc = subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kw)  # noqa: S603
    except OSError as e:
        Path(path).unlink(missing_ok=True)
        out(f"failed to start command: {e}")
        return False, -1, str(e), False

    def pump() -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            out(line)
            output.append(line + "\n")

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        while proc.poll() is None:
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
        out()
        out(c(f"{term.HI_YELLOW};{BOLD}", "\U0001F449 Caught interrupt. Exiting gracefully..."))
        _signal(proc, signal.SIGINT)
        try:
            proc.wait(2)
        except subprocess.TimeoutExpired:
            out(c(f"{term.HI_YELLOW};{BOLD}", "\U0001F449 Commands didn't exit after 2 seconds. Sending SIGKILL."))
            _signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
            proc.wait()
    reader.join(1)
    Path(path).unlink(missing_ok=True)
    code = proc.returncode if proc.returncode is not None else -1
    return code == 0, code, "".join(output), interrupted


def _signal(proc, sig) -> None:
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT if sig == signal.SIGINT else sig)   # type: ignore[attr-defined]
        else:
            os.killpg(proc.pid, sig)
    except (OSError, ValueError):
        pass


def on_exec_fail(ctx, status: int, output: str, attempt: int, rb: Rollback, on_success: Callable[[], None], auto_debug: int, flags: dict) -> None:
    wf = ctx.wf
    proceed = reset = False
    if auto_debug > 0:
        if attempt >= auto_debug:
            out(c(f"{term.HI_RED};{BOLD}", f"Commands failed {attempt} {'time' if attempt == 1 else 'times'}."))
        else:
            proceed = True
    if not proceed:
        pick = term.select("What do you want to do?", ["Debug and retry once", "Debug in full auto mode", "Rollback changes and exit", "Apply changes and exit"])
        if pick == "Debug and retry once":
            proceed = True
        elif pick == "Debug in full auto mode":
            proceed = reset = True
            spinner.start("")
            if wf.plan.autonomy != "full":
                wf.plan.set_auto("full")
                wf.save()
                flags.update(auto_commit=True, auto_confirm=True, auto_exec=True)
                auto_debug = wf.plan.setting("auto-debug-tries") or 5
                spinner.stop()
                out()
                out("✅ Full auto mode enabled")
                out()
            else:
                spinner.stop()
        elif pick == "Rollback changes and exit":
            if rb.has_changes():
                rollback(wf, rb)
            raise Exit(1)
        else:
            on_success()
            return
    if rb.has_changes():
        rollback(wf, rb)
    shown = f"Execution failed with exit status {status}. Output:\n\n{output}\n\n--\n\n"
    send = shown + debug_hint()
    ctx.tell(send, shown, mode="build", apply_debug=True)
    if reset:
        attempt = 0
    auto_exec = flags.get("auto_exec", False) or auto_debug > 0
    apply_plan(ctx, auto_confirm=flags["auto_confirm"], auto_commit=flags["auto_commit"], no_commit=flags["no_commit"], auto_exec=auto_exec,
               no_exec=flags["no_exec"], auto_debug=auto_debug, exec_command=flags.get("exec_command", ""), attempt=attempt + 1)


def debug_hint() -> str:
    system = f"{platform.system()} {platform.release()}"
    return (f"(This computer: {system}.) Fix the cause: reply with the complete files to change. If a command is wrong for this computer, or needs a "
            "tool that is not installed ('is not recognized', 'command not found'), change it under ### COMMANDS to one that works here, or put "
            "the command that installs the tool first (pip, npm, winget, brew or apt, whichever fits this system).")


# -- \debug ---------------------------------------------------------------------------------------------------------------------------------
def debug_cmd(ctx, args: list[str]) -> None:
    wf = ctx.wf
    tries = 5
    if args and args[0].lstrip("-").isdigit():
        tries = int(args[0])
        if tries <= 0:
            term.fail("Tries must be greater than 0")
        args = args[1:]
        if not args:
            term.fail("No command specified")
    if not args:
        term.usage_error("requires at least 1 arg(s), only received 0")
    cmd = " ".join(args)
    auto_commit = wf.plan.setting("auto-commit")
    for attempt in range(tries):
        success, status, output, interrupted = run_script(cmd, wf.session.cwd)
        if interrupted:
            out(c(f"{term.HI_YELLOW};{BOLD}", "\U0001F449  Execution interrupted"))
            yes, cancel = term.confirm_cancel("Did the command succeed?")
            if cancel:
                raise Exit(0)
            success = yes
        if success:
            if attempt == 0:
                out(f"✅ Command {c(f'{BOLD};{term.HI_CYAN}', cmd)} succeeded on first try")
            else:
                out(f"✅ Command {c(f'{BOLD};{term.HI_CYAN}', cmd)} succeeded after {attempt} fix {'attempt' if attempt == 1 else 'attempts'}")
            return
        if attempt == tries - 1:
            out(f"Command failed after {tries} tries")
            raise Exit(1)
        shown = f"'{cmd}' failed with exit status {status}. Output:\n\n{output}\n\n--\n\n"
        ctx.tell(shown + debug_hint(), shown, mode="build", apply_debug=True)
        apply_plan(ctx, auto_confirm=True, auto_commit=auto_commit, no_commit=not auto_commit, auto_exec=True, exec_command=cmd)


# -- diffs ------------------------------------------------------------------------------------------------------------------------------------
def git_diff_text(wf, colour: bool = True) -> str:
    chunks = []
    for path, new in sorted(wf.plan.b.pending.items()):
        old = wf.disk(path)
        a = [] if old is None else old.splitlines(keepends=True)
        b = new.splitlines(keepends=True)
        if a == b:
            continue
        head = [f"diff --git a/{path} b/{path}"] + (["new file mode 100644"] if old is None else [])
        body = list(difflib.unified_diff(a, b, "/dev/null" if old is None else f"a/{path}", f"b/{path}", n=3))
        lines = head + [ln.rstrip("\n") for ln in body]
        if colour:
            painted = []
            for ln in lines:
                if ln.startswith(("diff --git", "new file", "---", "+++")):
                    painted.append(c(f"{BOLD};{WHITE}", ln))
                elif ln.startswith("@@"):
                    painted.append(c(term.MUTED, ln))
                elif ln.startswith("+"):
                    painted.append(c(WHITE, ln))
                elif ln.startswith("-"):
                    painted.append(c(term.RED_HI, ln))
                else:
                    painted.append(ln)
            lines = painted
        chunks.append("\n".join(lines))
    if wf.plan.b.commands:
        script = "\n".join(wf.plan.b.commands)
        lines = [f"diff --git a/{APPLY} b/{APPLY}", "new file mode 100644", "--- /dev/null", f"+++ b/{APPLY}",
                 f"@@ -0,0 +1,{len(wf.plan.b.commands)} @@"] + [f"+{x}" for x in script.splitlines()]
        chunks.append("\n".join(c(WHITE, ln) if colour and ln.startswith("+") and not ln.startswith("+++") else ln for ln in lines))
    return "\n".join(chunks)


def page(text: str) -> None:
    less = shutil.which("less")
    if less and sys.stdout.isatty():
        try:
            subprocess.run([less, "-R"], input=text.encode("utf-8"), env={**os.environ, "LESS": "FRX", "LESSCHARSET": "utf-8"}, check=False)  # noqa: S603
            return
        except OSError as e:
            out(f"Failed to page output: {e}")
    out(text)


def diff_git(wf, plain: bool = False) -> None:
    spinner.start("")
    text = git_diff_text(wf, colour=not plain and term.colours_on())
    spinner.stop()
    if not text:
        out("\U0001F937‍♂️ No pending changes")
        return
    if plain:
        out(text)
        out()
        return
    page(text)
    out()


TEMPLATE = """<!doctype html>
<html lang="en-us">
  <head>
    <meta charset="utf-8" />
    <title>pal code · diffs</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/vs.min.css" />
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/diff2html@3.4.48/bundles/css/diff2html.min.css" />
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/diff2html@3.4.48/bundles/js/diff2html-ui.min.js"></script>
  </head>
  <script>
    const diffString = __DIFF__;
    document.addEventListener('DOMContentLoaded', function () {
      var targetElement = document.getElementById('myDiffElement');
      var configuration = { outputFormat: '__FORMAT__' };
      var diff2htmlUi = new Diff2HtmlUI(targetElement, diffString, configuration);
      diff2htmlUi.draw();
      diff2htmlUi.highlightCode();
    });
  </script>
  <body>
    <div id="myDiffElement"></div>
  </body>
</html>
"""


def _serve(html: str, fmt: str):
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://localhost:{srv.server_address[1]}/{fmt}"


def diff_ui(wf, side: bool = True, from_tell_menu: bool = False) -> None:
    import json
    import webbrowser
    spinner.start("")
    text = git_diff_text(wf, colour=False)
    spinner.stop()
    if not text:
        out("\U0001F937‍♂️ No pending changes")
        return
    fmt = "side-by-side" if side else "line-by-line"
    while True:
        srv, url = _serve(TEMPLATE.replace("__DIFF__", json.dumps(text)).replace("__FORMAT__", fmt), fmt)
        out(c(term.HI_GREEN, f"Showing {fmt} diffs in your default browser..."))
        out()
        out("If it doesn't open automatically, use this URL:")
        out(url)
        try:
            if not webbrowser.open(url):
                raise OSError("no browser")
        except Exception as e:  # noqa: BLE001
            out(f"Failed to open URL automatically: {e}")
            out("Please open the URL manually in your browser.")
        out()
        while True:
            if fmt == "line-by-line":
                out(c(f"{BOLD};{term.HI_GREEN}", "(s)") + " for side-by-side view")
            else:
                out(c(f"{BOLD};{term.HI_GREEN}", "(l)") + " for line-by-line view")
            out(c(f"{BOLD};{term.HI_GREEN}", "(g)") + " for git diff format")
            prompt_line("to go back>" if from_tell_menu else "to exit menu/continue>")
            key = term.get_key()
            out()
            if key == "down":
                other = "side-by-side" if fmt == "line-by-line" else "line-by-line"
                try:
                    pick = term.select("Select an action", [other, "git diff", "exit menu"])
                except term.Interrupted:
                    srv.shutdown()
                    return
                key = {"side-by-side": "s", "line-by-line": "l", "git diff": "g"}.get(pick, "q")
            if key in ("s", "l") and (key == "s") != (fmt == "side-by-side"):
                fmt = "side-by-side" if key == "s" else "line-by-line"
                srv.shutdown()
                break
            if key == "g":
                diff_git(wf)
                continue
            if key in ("q", "enter"):
                out()
                srv.shutdown()
                return
            if key in ("s", "l"):
                continue
            out()
            term.simple_error("Invalid hotkey")
            out()


# -- reject -----------------------------------------------------------------------------------------------------------------------------------
def reject(wf, args: list[str], all_: bool = False) -> None:
    spinner.start("")
    files = sorted(wf.plan.b.pending) + ([APPLY] if wf.plan.b.commands else [])
    if not files:
        term.fail("No pending changes to reject")

    def drop(paths: list[str]) -> None:
        cmd = APPLY in paths
        real = [p for p in paths if p != APPLY]
        if real:
            wf.reject(real)
        if cmd:
            _drop_commands(wf)
        n = len(paths)
        if all_:
            wf.log_event("\U0001F6AB Rejected all pending changes")
        else:
            wf.log_event(f"\U0001F6AB Rejected pending changes to file{'s' if n > 1 else ''}:" + "".join(f"\n • {p}" for p in sorted(paths)))

    if all_:
        drop(files)
        spinner.stop()
        n = len(files)
        out(f"✅ Rejected changes to {n} file{'s' if n > 1 else ''}")
        for p in files:
            out(f"• \U0001F4C4 {p}")
        return
    if args:
        for a in args:
            if a not in files:
                term.fail(f"File {a} not found in plan or has no changes to reject")
        drop(args)
        spinner.stop()
        n = len(args)
        out(f"✅ Rejected changes to {n} file{'s' if n > 1 else ''}")
        for p in sorted(args):
            out(f"• \U0001F4C4 {p}")
        return
    spinner.stop()
    try:
        picked = term.multi_select("Select files to reject:", files)
    except term.Interrupted:
        term.fail("Error getting file selection: interrupt")
    if not picked:
        out("No files selected")
        return
    spinner.start("")
    drop(picked)
    spinner.stop()
    n = len(picked)
    out(f"✅ Rejected changes to {n} file{'s' if n > 1 else ''}")
    for p in sorted(picked):
        out(f"• \U0001F4C4 {p}")


def _drop_commands(wf) -> None:
    import hashlib

    from pal.terminal.workflow import Version
    b = wf.plan.b
    b.versions.append(Version(hashlib.sha1(f"{time.time()}reject-commands".encode()).hexdigest()[:7], "rejected commands", time.time(), b.pending, []))  # noqa: S324
    wf.save()
