"""The builder bot: it reads what the person wants and proposes a plan: which agents, which tools, what may be accessed, which model.

The bot is a LOCAL model (the strongest one Ollama has). It is asked for one JSON object; the answer is untrusted, so it goes through the plan
normaliser, which enforces every safety rule. If the model's answer cannot be read after one retry, simple keyword rules make the plan instead, and the
plan says so. The bot only ever proposes: it cannot switch on a risky permission (see plan.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pal.llm_wrapper.components.base import LLMSyscall
from pal.llm_wrapper.modes import PROVIDER_KEYS
from pal.studio import plan as planlib
from pal.studio.catalog import OFFERED_TOOLS, PERMISSIONS, ROLES

EXAMPLE = {
    "product": {"name": "Meeting notes to action items", "summary": "A page where I paste meeting notes and get a list of action items.", "kind": "web_app"},
    "roles": ["designer", "coder", "docs", "reviewer"],
    "tools": [],
    "permissions": {"write_project": True, "modify_own_files": True, "read_references": True, "network_access": False},
    "model": {"mode": "local", "model": "<one of the installed models>"},
    "steps": [{"role": "designer", "task": "Design the page layout and styles."}, {"role": "coder", "task": "Write the logic that finds action items."},
              {"role": "docs", "task": "Write the README."}, {"role": "reviewer", "task": "Review all files and fix mistakes."}],
    "reasons": {"role:designer": "the product has a user interface", "role:reviewer": "a second look catches mistakes small models make", "model": "the strongest local model"},
    "metrics": False,
}


def system_prompt(info: dict) -> str:
    tools = [t for t in OFFERED_TOOLS if t in info.get("ready_tools", OFFERED_TOOLS)]
    from pal import tool_catalog
    return "\n".join([
        "You are the planner of Pal Studio. A person tells you what they want built; you choose HOW it will be built.",
        "Reply with ONE JSON object and nothing else: no prose, no code fences.",
        "",
        "Agents you may choose (ids): " + "; ".join(f"{r.id} = {r.description}" for r in ROLES.values()),
        "Tools you may choose (names; most builds need none): " + "; ".join(f"{t} = {tool_catalog.BY_NAME[t].summary}" for t in tools),
        "Permissions (true/false): " + "; ".join(f"{p.id} = {p.description}" + (" YOU MAY ONLY SUGGEST THIS; the person switches it on" if p.risky else "") + (" (locked off)" if p.locked else "") for p in PERMISSIONS),
        "Models: local (Ollama), installed: " + (", ".join(info.get("installed", [])) or "none") + ". Use mode \"local\" unless the product clearly needs a stronger model, "
        "then use mode \"api\" with a provider (" + ", ".join(PROVIDER_KEYS) + ") and a model name.",
        "Only choose what the product needs: a small page needs designer, coder and reviewer; a script may need only coder and tester. Always include coder.",
        "Each step is {\"role\", \"task\"} in build order, at most 8, ending with the reviewer when there is one. kind is one of: " + ", ".join(planlib.KINDS) + ".",
        "Give a short reason (under 20 words) for each role, tool and the model, under \"reasons\", keyed like role:coder, tool:calculator, model.",
        "",
        "Example of the shape:", json.dumps(EXAMPLE),
    ])


def extract_json(text: str) -> dict:
    """The first complete JSON object in a model's reply (fences, prose before or after, and trailing commas are tolerated)."""
    text = text.strip()
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\" and not esc)
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = re.sub(r",(\s*[}\]])", r"\1", text[start:i + 1])
                    try:
                        obj = json.loads(blob)
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in the reply")


# -- the fallback: simple keyword rules ------------------------------------------------------------------------------------------------------------
_WEB = re.compile(r"\b(web ?site|web ?page|landing|page|web app|webapp|html|browser|dashboard|ui|interface|portfolio|calculator app|game)\b", re.I)
_CLI = re.compile(r"\b(cli|command[- ]line|script|automation|scraper|converter|parser|csv|tool|bot)\b", re.I)
_TEST = re.compile(r"\btests?\b|unit test|pytest", re.I)


def heuristic_plan(command: str, info: dict) -> dict:
    web, cli = bool(_WEB.search(command)), bool(_CLI.search(command))
    kind = "web_app" if web or not cli else "cli_tool"
    roles = ["designer", "coder", "docs", "reviewer"] if kind == "web_app" else ["coder", "tester", "docs", "reviewer"]
    if _TEST.search(command) and "tester" not in roles:
        roles.insert(-2, "tester")
    name = " ".join(command.split()[:6]).strip(" .,:;") or "My project"
    return {"product": {"name": name[:60], "summary": " ".join(command.split())[:400], "kind": kind}, "roles": roles, "tools": [],
            "model": {"mode": "local", "model": info.get("recommended") or ""},
            "reasons": {"role:coder": "every build needs code written", "model": "the strongest local model that is installed"}}


_ROLE_WORDS = {"designer": ("designer", "design", "styling", "layout"), "coder": ("coder", "code", "developer"), "tester": ("tester", "tests", "testing", "test"),
               "docs": ("docs", "documentation", "readme", "docs writer"), "reviewer": ("reviewer", "review")}


def heuristic_revise(plan: dict, instruction: str, info: dict) -> dict:
    """A few plain edits when the model's answer cannot be read: add or remove an agent or a tool, switch the comparison on or off."""
    text = instruction.lower()
    new = json.loads(json.dumps(plan))
    adding = bool(re.search(r"\b(add|include|also|with|use|need)\b", text))
    removing = bool(re.search(r"\b(remove|drop|without|no|skip|don't|do not|exclude)\b", text))
    for role, words in _ROLE_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            if removing and not adding and role in new["roles"]:
                new["roles"].remove(role)
            elif adding and role not in new["roles"]:
                new["roles"].append(role)
    from pal import tool_catalog
    for t in OFFERED_TOOLS:
        names = {t, t.replace("_", " "), tool_catalog.BY_NAME[t].title.lower()}
        if any(n in text for n in names):
            if removing and not adding and t in new["tools"]:
                new["tools"].remove(t)
            elif t not in new["tools"] and not removing:
                new["tools"].append(t)
    if re.search(r"\b(compare|comparison|twice|metrics?|with and without)\b", text):
        new["metrics"] = not removing
    new["steps"] = []                 # the steps follow the agents
    return new


@dataclass
class PlanResult:
    plan: dict
    notes: list[str] = field(default_factory=list)
    source: str = "model"                  # "model" | "fallback"
    changes: list[str] = field(default_factory=list)
    message: str = ""


def describe(plan: dict, source: str, notes: list[str], model_note: str = "") -> str:
    names = [ROLES[r].name for r in plan["roles"]]
    pm = plan["model"]
    parts = [f"Here is my plan for **{plan['product']['name']}**.",
             f"Agents: {', '.join(names)}." + (f" Tools: {', '.join(plan['tools'])}." if plan["tools"] else " No extra tools."),
             "It saves files only inside its own project folder, does not run the code it writes, and " + ("may go online." if plan["permissions"].get("network_access") else "does not go online.")]
    parts.append(f"Model: {pm['model'] or 'none chosen'} ({'API, ' + pm['provider'] if pm['mode'] == 'api' else 'local, on this machine'})." + (f" {model_note}" if model_note else ""))
    if source == "fallback":
        parts.append("I could not read a plan from the model, so I used simple rules. Please check it.")
    parts += notes[:4]
    parts.append("Change anything on the right, or tell me what to change, then press Build.")
    return " ".join(parts)


class Planner:
    def __init__(self, core, info: dict):
        """`core` is a wrapper to chat with; `info` has installed (model names), recommended, ready_tools."""
        self.core, self.info = core, info

    def _ask(self, system: str, user: str, history: list[dict] | None = None) -> str:
        msgs = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
        return self.core.chat(LLMSyscall(messages=msgs, temperature=0.0, max_tokens=1800)).text

    def _normalize(self, raw: dict, user_set) -> tuple[dict, list[str]]:
        return planlib.normalize(raw, installed=self.info.get("installed", []), recommended=self.info.get("recommended"), user_set=user_set,
                                 ready_tools=self.info.get("ready_tools"))

    def _from_model(self, user: str, user_set) -> tuple[dict, list[str]] | None:
        system, history = system_prompt(self.info), []  # type: (str, list[dict])
        for attempt in range(2):
            try:
                reply = self._ask(system, user, history)
                raw = extract_json(reply)
                if not raw.get("roles") and not raw.get("product"):
                    raise ValueError("that JSON has no product or roles")
                return self._normalize(raw, user_set)
            except Exception as e:  # noqa: BLE001  (the model is down, or its answer cannot be read)
                if attempt == 0 and isinstance(e, ValueError):
                    history = [{"role": "user", "content": user}, {"role": "assistant", "content": "(not valid)"}]
                    user = f"Your last reply could not be read ({e}). Reply with ONE JSON object only, in the shape of the example."
                    continue
                return None
        return None

    def plan(self, command: str, user_set=()) -> PlanResult:
        got = self._from_model(f"The person wants: {command}", user_set)
        source = "model"
        if got is None:
            got, source = self._normalize(heuristic_plan(command, self.info), user_set), "fallback"
        plan, notes = got
        return PlanResult(plan, notes, source, [], describe(plan, source, notes))

    def revise(self, plan: dict, instruction: str, user_set=()) -> PlanResult:
        user = ("Here is the current plan as JSON:\n" + json.dumps(plan) + f"\n\nThe person says: {instruction}\n"
                "Reply with the full updated plan as ONE JSON object. Change only what they asked for.")
        got, source = self._from_model(user, user_set), "model"
        if got is None:
            got, source = self._normalize(heuristic_revise(plan, instruction, self.info), user_set), "fallback"
        new, notes = got
        changes = planlib.diff(plan, new)
        msg = ("Done: " + "; ".join(changes) + "." if changes else "I made no change to the plan.")
        if source == "fallback":
            msg += " (I could not read the model's answer, so I applied simple rules: check the plan.)"
        if not changes and source == "fallback":
            msg = "I did not understand that. Try the controls on the right, or say for example 'add the tester' or 'build twice to compare'."
        return PlanResult(new, notes, source, changes, " ".join([msg, *notes[:4]]))
