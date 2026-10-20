"""The plan: what the bot proposes and what the person edits. One normaliser enforces every rule, whoever changed the plan.

The rules that matter:
  * a risky permission (going online) can be on only if the PERSON switched it on (`user_set`): the bot may suggest it, never enable it;
  * a locked permission (running code, touching files outside the project, deleting) is always off;
  * a tool that needs a permission is dropped when that permission is off, and the plan says so;
  * unknown roles, tools and permissions are dropped, so a model's invention can never become an action.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from pal.llm_wrapper.modes import PROVIDER_KEYS, mode_for
from pal.studio.catalog import DEFAULT_PERMISSIONS, OFFERED_TOOLS, PERMISSION_BY_ID, ROLE_ORDER, ROLES, tool_needs

KINDS = ("web_app", "cli_tool", "library", "script", "document", "other")
MAX_STEPS, MAX_ROLES = 8, 5
_FOLDER = re.compile(r"[^a-z0-9]+")


_DEVICE = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))})


def slug(name: str) -> str:
    """A safe folder name for a product. A name that is a Windows device name (con, nul, aux ...) would break there, so it gets a prefix."""
    out = _FOLDER.sub("-", name.lower()).strip("-")[:40] or "project"
    return f"p-{out}" if out in _DEVICE else out


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _text(value, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def default_steps(roles: list[str], summary: str | None = "") -> list[dict]:
    task = {"designer": "Design the layout and write the styles and markup.", "coder": "Write the working code and the main files.",
            "tester": "Write tests for what exists.", "docs": "Write the README: what it is, how to run it, one example.",
            "reviewer": "Review every file written so far and give corrected versions of any with mistakes."}
    return [{"role": r, "task": task[r]} for r in ROLE_ORDER if r in roles]


def normalize(raw: Mapping | None, *, installed: Iterable[str] = (), recommended: str | None = None, user_set: Iterable[str] = (),
              ready_tools: Iterable[str] | None = None) -> tuple[dict, list[str]]:
    """Any proposed plan -> (a valid plan, the notes about what had to change). `installed` are the local models that exist; `ready_tools` the
    tools that can be used right now (None: every offered tool)."""
    raw = dict(raw or {})
    notes: list[str] = []
    user_set = set(user_set)
    installed = list(installed)
    ready = set(OFFERED_TOOLS if ready_tools is None else ready_tools)

    prod = _mapping(raw.get("product"))
    kind = prod.get("kind") if prod.get("kind") in KINDS else "web_app"
    name = _text(prod.get("name"), 60) or "My project"
    product = {"name": name, "summary": _text(prod.get("summary"), 400), "kind": kind, "folder": slug(name)}

    roles = []
    for r in raw.get("roles") or []:
        if isinstance(r, str) and r in ROLES and r not in roles:
            roles.append(r)
        elif isinstance(r, str) and r not in ROLES:
            notes.append(f"dropped an agent that does not exist: {_text(r, 30)}")
    roles = [r for r in ROLE_ORDER if r in roles][:MAX_ROLES]
    if "coder" not in roles:
        roles.insert(0, "coder")
        notes.append("added the Coder: a build needs someone to write the code")
        roles = [r for r in ROLE_ORDER if r in roles]

    perms = dict(DEFAULT_PERMISSIONS)
    given = _mapping(raw.get("permissions"))
    for pid, want in given.items():
        p = PERMISSION_BY_ID.get(pid)
        if p is None:
            notes.append(f"dropped a permission that does not exist: {_text(pid, 30)}")
            continue
        if p.locked:
            if want:
                notes.append(f"{p.title} stays off: {p.why_locked}")
            continue
        if want and p.risky and pid not in user_set:
            notes.append(f"{p.title} stays off: only you can switch it on (use the switch in Access)")
            continue
        perms[pid] = bool(want)

    tools = []
    for t in raw.get("tools") or []:
        if not isinstance(t, str) or t in tools:
            continue
        if t not in OFFERED_TOOLS:
            notes.append(f"dropped a tool that is not available to builds: {_text(t, 30)}")
            continue
        need = tool_needs(t)
        if need and not perms.get(need):
            notes.append(f"left out {t}: it needs '{PERMISSION_BY_ID[need].title}', which is off")
            continue
        if t not in ready:
            notes.append(f"left out {t}: it is not switched on in the settings")
            continue
        tools.append(t)

    m = _mapping(raw.get("model"))
    if m.get("mode") == "api" and m.get("provider") in PROVIDER_KEYS and _text(m.get("model"), 120):
        model = {"mode": "api", "provider": m["provider"], "model": _text(m["model"], 120), "api_mode": mode_for(m["provider"])}
    else:
        pick = m.get("model") if m.get("mode") == "local" and m.get("model") in installed else (recommended or (installed[0] if installed else ""))
        model = {"mode": "local", "model": pick or "", "provider": "ollama", "api_mode": ""}
        if m.get("mode") == "local" and m.get("model") and m.get("model") not in installed and installed:
            notes.append(f"{_text(m['model'], 40)} is not installed: using {pick}")

    steps = []
    for s in raw.get("steps") or []:
        if isinstance(s, Mapping) and s.get("role") in roles and _text(s.get("task"), 300):
            steps.append({"role": s["role"], "task": _text(s["task"], 300)})
    steps = steps[:MAX_STEPS] or default_steps(roles, product["summary"])

    reasons = {}
    for key, why in _mapping(raw.get("reasons")).items():
        kind_, _, ident = str(key).partition(":")
        if (kind_ == "role" and ident in roles) or (kind_ == "tool" and ident in tools) or (kind_ == "permission" and ident in PERMISSION_BY_ID) or key in ("model", "metrics"):
            reasons[str(key)] = _text(why, 200)

    return {"product": product, "roles": roles, "tools": tools, "permissions": perms, "model": model, "steps": steps, "reasons": reasons,
            "metrics": bool(raw.get("metrics"))}, notes


def diff(old: Mapping, new: Mapping) -> list[str]:
    """What changed between two plans, in words the person can read."""
    out: list[str] = []
    for key, label in (("roles", "agent"), ("tools", "tool")):
        a, b = set(old.get(key, [])), set(new.get(key, []))
        out += [f"added {label}: {ROLES[x].name if key == 'roles' and x in ROLES else x}" for x in new.get(key, []) if x not in a]
        out += [f"removed {label}: {ROLES[x].name if key == 'roles' and x in ROLES else x}" for x in old.get(key, []) if x not in b]
    for pid, now in new.get("permissions", {}).items():
        if old.get("permissions", {}).get(pid) != now:
            out.append(f"{'turned on' if now else 'turned off'}: {PERMISSION_BY_ID[pid].title.lower()}")
    if old.get("model") != new.get("model"):
        nm = new["model"]
        out.append(f"model: {nm['model'] or 'none'} ({'API, ' + nm['provider'] if nm['mode'] == 'api' else 'local'})")
    if old.get("steps") != new.get("steps"):
        out.append(f"steps changed ({len(new.get('steps', []))} now)")
    if old.get("product") != new.get("product"):
        out.append("product description changed")
    if bool(old.get("metrics")) != bool(new.get("metrics")):
        out.append("build twice to compare: " + ("on" if new.get("metrics") else "off"))
    return out
