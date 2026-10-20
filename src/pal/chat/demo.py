"""The experiment tab's demo: the same 6 messages, sent two ways, measured.

  naive  : what a plain chat client does. Full history every turn (attachments included), the raw
           message, no output rules, every message to the API model.
  pal : ChatService.send() with everything on (routing, input savers, direct answers, document
           index, paging, output rules).

Both arms use the same API model and count the tokens the provider itself reports. Latency is
wall-clock time per turn; Pal's includes the local-model time, so a slow local model can make
Pal *slower* on some turns, and the summary says so instead of hiding it. Demo turns read and
write no memory and are not added to the Stats.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from pal.chat.service import ChatService
from pal.errors import NotFoundError, UserInputError
from pal.llm_wrapper.components.base import LLMSyscall

SURVEY_NAME = "student-survey.md"
SURVEY = """# Papertrail student survey (120 students, 4 campuses)

## Who answered
- 120 students: 58 undergraduates, 41 master's students, 21 PhD students.
- 71% take notes on a laptop, 19% on a tablet, 10% on paper only.
- 64% use more than one note app today; the most common are Google Docs (62%), Notion (38%) and OneNote (27%).

## What they dislike about current tools
- Finding old notes is the top complaint: 57% say they often cannot find a note they wrote last semester.
- 41% say syncing between phone and laptop fails at least once a month.
- 33% say the tools are too complicated for lecture notes and slow to open in class.
- 18% say subscription prices are too high for a student budget.

## What they would pay
- 46% would pay nothing for a note app.
- 38% would pay up to 3 dollars a month.
- 12% would pay 3 to 6 dollars a month.
- 4% would pay more than 6 dollars a month.
- 72% say a free tier with a paid upgrade is what they prefer.

## Where they hear about new apps
- Instagram: 61%. Student WhatsApp and Telegram groups: 48%.
- Campus posters and notice boards: 35%. Friends and classmates: 33%.
- Reddit: 22%. Email newsletters from the university: 14%. TikTok: 12%.

## Features they asked for most
- Search that also finds words inside handwriting and photos of slides: 52%.
- Works offline in lecture halls: 44%.
- Share a notebook with a study group: 39%.
- Turn notes into flashcards: 27%.

## Quotes
- "I write everything down and then never find it again."
- "I would try anything that opens instantly and does not need a login."
- "I only download new apps when three friends already use them."
"""

STEPS: list[dict[str, Any]] = [
    {"message": "Hi there! So, I was basically wondering if you could please help me plan a small launch for "
                "my new note-taking app called Papertrail. It is really just for students, and I would like to "
                "keep the whole thing under a budget of 500 dollars, if that is at all possible.",
     "route": "api"},
    {"message": "Thanks so much. Here are my survey notes, please just read them and tell me the three most "
                "important things that students say.",
     "attach": True, "route": "api"},
    {"message": "what is 500*0.35", "route": None},
    {"message": "Quick question, what does ROI stand for?", "route": None},
    {"message": "Given the survey notes and the budget, could you please suggest the three channels that I "
                "should focus on, and roughly how to split the 500 dollars between them?",
     "route": "api"},
    {"message": "Great, now please put everything together into a one page launch brief for Papertrail: goal, "
                "audience, budget split, channels, an email subject line, and a first week timeline.",
     "route": "api"},
]

_MAX_RUNS = 20
_TTL = 3600.0
_NAIVE_SYSTEM_NOTE = ""  # a naive client adds nothing to the base prompt: no output rules


@dataclass
class _Run:
    id: str
    sid: str
    provider: str
    model: str
    naive_history: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    local: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _public_steps() -> list[dict]:
    return [{"message": s["message"], "attachment": SURVEY_NAME if s.get("attach") else None} for s in STEPS]


class Demo:
    def __init__(self, service: ChatService):
        self.service = service
        self._runs: OrderedDict[str, _Run] = OrderedDict()
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self, provider: str, model: str, api_key: str | None) -> dict:
        self.service.api_core(provider, model, api_key)  # fail fast on a bad provider, model or missing key
        run = _Run(id=uuid.uuid4().hex, sid=f"demo-{uuid.uuid4().hex[:24]}", provider=provider, model=model,
                   local=self.service.local_status())
        with self._lock:
            self._expire()
            self._runs[run.id] = run
            while len(self._runs) > _MAX_RUNS:
                _, old = self._runs.popitem(last=False)
                self.service.reset(old.sid)
        return {"run_id": run.id, "steps": _public_steps(), "local": run.local}

    def _expire(self) -> None:
        now = time.time()
        for k in [k for k, r in self._runs.items() if now - r.created > _TTL]:
            self.service.reset(self._runs.pop(k).sid)

    def _get(self, run_id: str) -> _Run:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise NotFoundError("demo run not found or expired: start it again")
        return run

    def cancel(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.pop(run_id, None)
        if run:
            self.service.reset(run.sid)

    # -- one step ---------------------------------------------------------------------------------------------
    def step(self, run_id: str, api_key: str | None) -> dict:
        run = self._get(run_id)
        with run.lock:
            i = len(run.steps)
            if i >= len(STEPS):
                raise UserInputError("the demo already finished")
            spec = STEPS[i]
            api = self.service.api_core(run.provider, run.model, api_key)

            # naive arm: full history, raw message, the file re-sent every time it is in the history
            user_text = spec["message"] + (f"\n\n[file: {SURVEY_NAME}]\n{SURVEY}" if spec.get("attach") else "")
            messages = [{"role": "system", "content": self.service.settings.base_system_prompt}]
            messages += run.naive_history + [{"role": "user", "content": user_text}]
            t0 = time.perf_counter()
            try:
                resp = api.chat(LLMSyscall(messages=messages, max_tokens=2048))
            except Exception as e:  # noqa: BLE001
                from pal.errors import ProviderError
                secret = getattr(getattr(api, "config", None), "api_key", None) or ""
                text = str(e).replace(secret, "***") if secret else str(e)
                raise ProviderError(f"naive arm: {type(e).__name__}: {text}") from e
            naive = {"input_tokens": resp.prompt_tokens or 0, "output_tokens": resp.completion_tokens or 0,
                     "latency": round(time.perf_counter() - t0, 3), "reply": resp.text or ""}

            # Pal arm: the real service, on its own throwaway session
            attachments = [{"name": SURVEY_NAME, "text": SURVEY, "kind": "document"}] if spec.get("attach") else []
            res = self.service.send(run.sid, run.provider, run.model, api_key, spec["message"], attachments,
                                    None, route=spec["route"], ephemeral=True)
            pal = {"route": res.handled_by, "reason": res.route_reason, "input_tokens": res.input_tokens,
                      "output_tokens": res.output_tokens, "latency": res.latency.get("total", 0.0),
                      "latency_local": res.latency.get("local", 0.0), "latency_api": res.latency.get("api", 0.0),
                      "tools": [t["name"] for t in res.tools], "reply": res.reply,
                      "reduced_tokens_saved": getattr(res.saver, "tokens_saved", 0),
                      "history_turns_sent": (res.context or {}).get("turns_sent"),
                      "history_tokens_saved": (res.context or {}).get("tokens_saved", 0),
                      "document_chunks": (res.documents or {}).get("chunks", 0)}

            # commit only after both arms succeeded, so a retry of this step stays consistent
            run.naive_history += [{"role": "user", "content": user_text},
                                  {"role": "assistant", "content": naive["reply"]}]
            row = {"step": i + 1, "message": spec["message"], "naive": naive, "pal": pal}
            run.steps.append(row)
            out: dict[str, Any] = {"row": row, "done": len(run.steps) == len(STEPS)}
            if out["done"]:
                out["summary"] = self._summary(run)
                self.cancel(run.id)
            return out

    # -- totals ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _pct(saved: float, base: float) -> float:
        return round(100 * saved / base, 1) if base else 0.0

    def _summary(self, run: _Run) -> dict:
        rows = run.steps
        n_in = sum(r["naive"]["input_tokens"] for r in rows)
        n_out = sum(r["naive"]["output_tokens"] for r in rows)
        m_in = sum(r["pal"]["input_tokens"] for r in rows)
        m_out = sum(r["pal"]["output_tokens"] for r in rows)
        n_lat = sum(r["naive"]["latency"] for r in rows)
        m_lat = sum(r["pal"]["latency"] for r in rows)
        routes = [r["pal"]["route"] for r in rows]
        saved_tokens = (n_in + n_out) - (m_in + m_out)
        saved_lat = n_lat - m_lat
        local_ok = run.local.get("ok")
        notes = []
        if local_ok is False:
            notes.append("The local model was not reachable, so input reduction and local answering did not run. "
                         "The saving below comes from direct answers, the output rules and document handling only. "
                         "Start Ollama and run the demo again to see the full effect.")
        if saved_lat < 0:
            notes.append("Pal was slower overall: the local model's time is included, and on a slow machine it "
                         "can cost more than it saves.")
        notes.append("Pal's output rules make replies shorter, which is part of the token saving and also "
                     "changes the style of the answers.")
        return {
            "naive": {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out,
                      "latency": round(n_lat, 2), "api_calls": len(rows)},
            "pal": {"input_tokens": m_in, "output_tokens": m_out, "total_tokens": m_in + m_out,
                       "latency": round(m_lat, 2),
                       "api_calls": sum(1 for r in routes if r == "api"),
                       "local_answers": routes.count("local"), "direct_answers": routes.count("tool"),
                       "tool_calls": sum(len(r["pal"]["tools"]) for r in rows)},
            "tokens_saved": saved_tokens, "tokens_saved_pct": self._pct(saved_tokens, n_in + n_out),
            "input_saved": n_in - m_in, "output_saved": n_out - m_out,
            "latency_saved": round(saved_lat, 2), "latency_saved_pct": self._pct(saved_lat, n_lat),
            "api_calls_avoided": len(rows) - sum(1 for r in routes if r == "api"),
            "local": run.local, "notes": notes,
            "result": rows[-1]["pal"]["reply"], "result_naive": rows[-1]["naive"]["reply"],
        }
