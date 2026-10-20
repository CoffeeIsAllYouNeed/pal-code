"""The local LLM does the task itself. It may decline with ESCALATE."""
from __future__ import annotations

from typing import Optional

from pal.llm_wrapper.components.base import LLMSyscall, LLMWrapper
from pal.task_router.components.base import LOCAL_TAG, LocalAnswer

_SYSTEM = (
    "You answer short, easy tasks directly and correctly. Be brief. "
    "If you are not fully certain, or the task needs more than a short answer, reply with exactly: ESCALATE"
)
_FILE_CAP = 8000


def _clean(content: str, cap: int) -> str:
    c = content[len(LOCAL_TAG):].lstrip() if content.startswith(LOCAL_TAG) else content
    return c if len(c) <= cap else c[:cap] + " ..."


def answer(core: LLMWrapper, message: str, history: list[dict], max_history: int = 6,
           attachments: Optional[list] = None) -> LocalAnswer:
    recent = [{"role": m["role"], "content": _clean(str(m["content"]), 600)}
              for m in history[-max_history:] if m.get("role") in ("user", "assistant")] if max_history else []
    files = "".join(f"\n\n[file: {a['name']}]\n{a['text'][:_FILE_CAP]}" for a in (attachments or []))
    messages = [{"role": "system", "content": _SYSTEM}] + recent + [{"role": "user", "content": message + files}]
    try:
        resp = core.chat(LLMSyscall(messages=messages, temperature=0, max_tokens=400))
    except Exception as e:
        return LocalAnswer(None, f"local model error: {e.__class__.__name__}: {e}")
    text = (resp.text or "").strip()
    pt, ct = getattr(resp, "prompt_tokens", 0) or 0, getattr(resp, "completion_tokens", 0) or 0
    if not text:
        return LocalAnswer(None, "local model gave an empty answer", pt, ct)
    if text.upper().startswith("ESCALATE"):
        return LocalAnswer(None, "local model asked to escalate", pt, ct)
    return LocalAnswer(text, "", pt, ct)
