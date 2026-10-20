"""One build: the plan's steps, one after another, each a request to the model, each reply read for files, checked, and saved under the policy.

Everything that happens is reported as an event (`emit(kind, text, **data)`), so the page can show the work as it happens. Nothing the model wrote is
ever run: files are read, checked for mistakes, and written, if the policy allows.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pal.studio.catalog import ROLES
from pal.studio.checks import check_project
from pal.studio.policy import Policy
from pal.terminal.files import parse_files

FORMAT = ("Reply with the files you create or change. Write each file as a line `### FILE: path/name.ext` followed by its COMPLETE content in a code block. "
          "No partial files, no '...' placeholders. After the files you may add one line starting with NOTE: if something needs saying.")
KIND_RULES = {
    "web_app": "It is a web page: plain HTML, CSS and JavaScript that works when index.html is opened in a browser. No build step, no frameworks to install, no external URLs or CDNs. Start with index.html.",
    "cli_tool": "It is a command-line program in Python 3 using only the standard library. Start with main.py.",
    "script": "It is a Python 3 script using only the standard library. Start with main.py.",
    "library": "It is a small Python 3 library using only the standard library, with a clear public API.",
    "document": "It is a document: write it as Markdown files.",
    "other": "Use plain files that need nothing to be installed.",
}
MAX_REPAIRS = 1


def default_name(plan: dict, step: dict) -> str:
    if step["role"] == "docs":
        return "README.md"
    if step["role"] == "tester":
        return "test_main.py"
    return {"web_app": "index.html", "cli_tool": "main.py", "script": "main.py", "library": "main.py", "document": "README.md"}.get(plan["product"]["kind"], "main.txt")


def first_message(plan: dict, step: dict, with_references: bool) -> str:
    team = ", ".join(ROLES[r].name for r in plan["roles"])
    p = plan["product"]
    return "\n".join([
        f"We are building: {p['name']}. {p['summary']}",
        KIND_RULES.get(p["kind"], KIND_RULES["other"]),
        f"The team: {team}. Each step is done by one of them, and every file must work with the files already written (same names, same functions).",
        "The reference files the person attached are available to you." if with_references else "",
        FORMAT, "",
        f"Step 1 ({ROLES[step['role']].name}): {step['task']}", ROLES[step["role"]].prompt])


def step_message(step: dict, i: int, n: int) -> str:
    return (f"Step {i} of {n} ({ROLES[step['role']].name}): {step['task']}\n{ROLES[step['role']].prompt}\n"
            "Keep file names and functions consistent with the files already written. " + FORMAT)


class Build:
    def __init__(self, plan: dict, client, workspace: Path | None, policy: Policy, emit: Callable, cancelled: Callable[[], bool] = lambda: False,
                 references: list[dict] | None = None):
        self.plan, self.client, self.workspace, self.policy, self.emit, self.cancelled = plan, client, workspace, policy, emit, cancelled
        self.references = references if plan["permissions"].get("read_references") else []
        self.files: dict[str, str] = {}                 # everything the build wrote (or would have written, in a dry run)
        self.totals: dict[str, Any] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "main_tokens": 0, "helper_tokens": 0, "seconds": 0.0,
                       "files": 0, "blocked": 0, "problems_left": 0, "steps_done": 0, "steps": len(plan["steps"]), "stopped": ""}

    # -- one request to the model ----------------------------------------------------------------------------------------
    def _ask(self, message: str, label: str, refs: list[dict] | None = None) -> str:
        reply = self.client.ask(message, refs)
        t = self.totals
        t["calls"] += max(reply.calls, 1)
        t["prompt_tokens"] += reply.prompt_tokens
        t["completion_tokens"] += reply.completion_tokens
        t["main_tokens"] += reply.prompt_tokens + reply.completion_tokens
        t["helper_tokens"] += reply.helper_tokens
        t["seconds"] = round(t["seconds"] + reply.seconds, 3)
        self.emit("model_call", f"{label}: the model read {reply.prompt_tokens:,} tokens and wrote {reply.completion_tokens:,} in {reply.seconds:.1f}s"
                  + (f" (Pal's helper used {reply.helper_tokens:,} more)" if reply.helper_tokens else ""),
                  prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens, helper_tokens=reply.helper_tokens, seconds=round(reply.seconds, 2))
        return reply.text

    # -- reading and saving the files in a reply -----------------------------------------------------------------------
    def _apply(self, reply: str, step: dict) -> set[str]:
        parsed, unnamed = parse_files(reply, default_name(self.plan, step))
        changed: set[str] = set()
        if unnamed:
            self.emit("warning", f"{unnamed} code block(s) had no file name and were skipped")
        for pf in parsed:
            d = self.policy.check_write(pf.path, pf.content)
            if d.allowed:
                existed = d.path.lower() in {p.lower() for p in self.files}
                self.files[d.path] = pf.content
                if self.workspace is not None:
                    target = self.workspace / d.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(pf.content, encoding="utf-8", newline="\n")
                changed.add(d.path)
                self.totals["files"] = len(self.files)
                self.emit("file", f"{'changed' if existed else 'wrote'} {d.path} ({len(pf.content.encode('utf-8')):,} bytes)", path=d.path, action="changed" if existed else "created")
            elif d.path and "dry run" in d.reason:
                self.files[d.path] = pf.content
                changed.add(d.path)
                self.totals["files"] = len(self.files)
                self.emit("file", f"would write {d.path} (dry run: not saved)", path=d.path, action="dry")
            else:
                self.totals["blocked"] += 1
                self.emit("blocked", f"did not save {d.path or 'a file'}: {d.reason}", path=d.path, reason=d.reason)
        return changed

    @staticmethod
    def _describe(problems: dict[str, list[str]]) -> str:
        return "\n".join(f"- {path}: {'; '.join(p)}" for path, p in problems.items())

    # -- the build ---------------------------------------------------------------------------------------------------------
    def run(self) -> dict:
        steps, n = self.plan["steps"], len(self.plan["steps"])
        for i, step in enumerate(steps, 1):
            if self.cancelled():
                self.totals["stopped"] = "cancelled"
                self.emit("cancelled", "stopped by you")
                return self.totals
            role = ROLES[step["role"]]
            self.emit("step_start", f"Step {i} of {n}: {role.name}: {step['task']}", step=i, total=n, role=step["role"])
            first = i == 1
            msg = first_message(self.plan, step, bool(self.references)) if first else step_message(step, i, n)
            try:
                reply = self._ask(msg, f"step {i}", self.references if first else None)
            except Exception as e:  # noqa: BLE001  (the model failed: say so, and stop cleanly)
                self.totals["stopped"] = "error"
                self.emit("error", f"the model failed on step {i}: {type(e).__name__}: {str(e)[:200]}")
                return self.totals
            changed = self._apply(reply, step)
            problems = check_project(self.files, only=changed, links=False)
            repairs = 0
            while problems and repairs < MAX_REPAIRS and not self.cancelled():
                self.emit("check", "found problems: " + "; ".join(f"{p}: {', '.join(v)}" for p, v in list(problems.items())[:3]), problems=problems)
                repairs += 1
                try:
                    fix = self._ask("These problems were found in the files:\n" + self._describe(problems) + "\nReply with the corrected COMPLETE files only. " + FORMAT, f"step {i} fix")
                except Exception as e:  # noqa: BLE001
                    self.emit("error", f"the fix request failed: {type(e).__name__}")
                    break
                changed |= self._apply(fix, step)
                problems = check_project(self.files, only=changed, links=False)
            if problems:
                self.emit("check", "problems still there after one fix: " + "; ".join(f"{p}: {', '.join(v)}" for p, v in list(problems.items())[:3]), problems=problems)
            else:
                self.emit("check", "no problems found in the files")
            self.totals["steps_done"] = i
            self.emit("step_done", f"Step {i} done: {len(self.files)} file(s) so far", step=i)
        self._final_check(steps[-1] if steps else None)
        return self.totals

    def _final_check(self, step: dict | None) -> None:
        """When every step is done: do the files fit together (does every page link to files that exist)? One fix is allowed, then the result is recorded."""
        if step is None or self.cancelled() or self.totals["stopped"]:
            return
        problems = check_project(self.files)
        if problems:
            self.emit("check", "project check, found: " + "; ".join(f"{p}: {', '.join(v)}" for p, v in list(problems.items())[:3]), problems=problems)
            try:
                fix = self._ask("These problems were found in the finished project:\n" + self._describe(problems) + "\nReply with the missing or corrected COMPLETE files only. " + FORMAT, "final fix")
                self._apply(fix, step)
            except Exception as e:  # noqa: BLE001
                self.emit("error", f"the final fix request failed: {type(e).__name__}")
            problems = check_project(self.files)
        self.totals["problems_left"] = sum(len(v) for v in problems.values())
        self.emit("check", "project check: the files fit together" if not problems else f"project check: {self.totals['problems_left']} problem(s) left: "
                  + "; ".join(f"{p}: {', '.join(v)}" for p, v in list(problems.items())[:3]), problems=problems)
