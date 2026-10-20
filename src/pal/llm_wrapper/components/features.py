"""Function calling and structured output, in two flavours.

Native: the backend supports it (`WrapperCapabilities.supports_tools` / `supports_json`) and the core uses the provider's own
mechanism. Emulated: for backends that do not (an in-process Hugging Face model, for one), the adapter puts the contract in
the prompt and validates what comes back. Both produce the same result shape, so callers never know which was used.
"""
from __future__ import annotations

import json
import re
from typing import Any


def normalize_tools(tools: list[dict] | None) -> list[dict]:
    """Accept OpenAI-style {"type": "function", "function": {...}} or flat {"name", "description", "parameters"}."""
    out = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name")
        if not name:
            raise ValueError("every tool needs a name")
        out.append({"name": name, "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}})
    return out


def is_response_format(rf: dict | None) -> bool:
    """True for a provider-style response format ({"type": "json_object"} and friends); False for a JSON schema, which
    also has a `type` key ({"type": "object", ...})."""
    return isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema", "text")


def parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {"value": val}
        except ValueError:
            return {"_raw": raw}
    return {}


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating ``` fences and a sentence before or after it."""
    t = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S | re.I)
    if fence:
        t = fence.group(1)
    try:
        return json.loads(t)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = t.find(open_c), t.rfind(close_c)
        if 0 <= a < b:
            try:
                return json.loads(t[a:b + 1])
            except ValueError:
                continue
    raise ValueError("no valid JSON found in the reply")


def json_instruction(schema: dict | None) -> str:
    base = "Reply with only a single valid JSON value. No explanation, no markdown fences."
    if schema and not is_response_format(schema):
        return base + " It must match this JSON schema: " + json.dumps(schema, separators=(",", ":"))
    return base


def tools_instruction(tools: list[dict]) -> str:
    listing = "; ".join(f'{t["name"]}: {t["description"]} parameters={json.dumps(t["parameters"], separators=(",", ":"))}'
                        for t in tools)
    return ('You can call tools. To call one or more, reply with only JSON of the form '
            '{"tool_calls": [{"name": "<tool name>", "parameters": {...}}]} and nothing else. '
            'If no tool is needed, answer normally in plain text. Tools: ' + listing)


def parse_emulated_tool_calls(text: str, tools: list[dict]) -> list[dict] | None:
    """The tool calls a model asked for in the emulated format, or None when it answered in plain text."""
    try:
        data = extract_json(text)
    except ValueError:
        return None
    calls = data.get("tool_calls") if isinstance(data, dict) else None
    if not isinstance(calls, list) or not calls:
        return None
    out = []
    for c in calls:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            out.append({"name": c["name"], "parameters": parse_arguments(c.get("parameters", c.get("arguments")))})
    return out or None


class ToolCallError(ValueError):
    """The model asked for a tool that was not offered."""


def check_tool_names(calls: list[dict], tools: list[dict]) -> None:
    known = {t["name"] for t in tools}
    for c in calls:
        if c["name"] not in known:
            raise ToolCallError(f"the model called an unknown tool: {c['name']!r}")
