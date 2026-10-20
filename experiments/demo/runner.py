"""The two ways of running the same session.

BASELINE is a plain chat client: it sends the system prompt, the whole conversation so far, and, in the first message of each, the
whole documentation (and later the whole project source), to the model, every turn. No trimming, no retrieval, no tools, no routing.
It is deliberately naive, as the comparison needs a client that does nothing about token cost.

PAL-OS is the real chat service: the same session through the actual pipeline, with the same local models.
"""
from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass, field

from experiments.demo.models import Recorder, RecordingCore, SimScript, count_messages
from experiments.demo.scenario import Scenario, Turn
from pal.llm_wrapper.components.base import LLMSyscall

BASELINE_SYSTEM = "You are a helpful assistant."


@dataclass
class Outcome:
    turn: int
    kind: str
    reply: str
    seconds: float                      # latency of the whole turn
    overhead: float                     # of which: time outside the models (savers, retrieval, scheduling, graph build)
    handled_by: str                     # "main" for the baseline; "direct", "local" or "api" for Pal
    ok: bool | None                     # did the answer contain what it should (None = not checked)


@dataclass
class ArmResult:
    name: str
    outcomes: list = field(default_factory=list)
    recorder: Recorder = field(default_factory=Recorder)
    notes: dict = field(default_factory=dict)


def check(turn: Turn, reply: str) -> bool | None:
    if turn.expect_regex:
        return bool(re.search(turn.expect_regex, reply))
    if turn.expect:
        flat = reply.lower().replace(",", "")
        return all(e.lower() in flat for e in turn.expect)
    return None


def _turn_seconds(rec: Recorder, turn: int, wall: float) -> tuple[float, float]:
    """(latency, overhead) of a turn: the time the models took (measured, or computed for a simulated model) plus the time the rest
    of the work took on this machine."""
    calls = rec.of_turn(turn)
    overhead = max(0.0, wall - sum(c.wall for c in calls))
    return sum(c.seconds for c in calls) + overhead, overhead


def baseline_messages(sc: Scenario, upto: int, replies: list) -> list:
    """The conversation a plain client holds just before turn `upto` is answered: everything so far, with the attachments pasted in."""
    msgs = [{"role": "system", "content": BASELINE_SYSTEM}]
    for t in sc.turns[:upto]:
        text = t.text
        if t.attach == "document":
            text = f"{sc.document}\n\n{text}"
        elif t.attach == "project":
            text = f"Here is the source of the project:\n{sc.project_text()}\n\n{text}"
        msgs.append({"role": "user", "content": text})
        if t.i <= len(replies):
            msgs.append({"role": "assistant", "content": replies[t.i - 1]})
    return msgs


def baseline_peak_tokens(sc: Scenario, reply_tokens: int = 120) -> int:
    """The largest prompt the baseline will send: how big a context window the models need."""
    replies = ["x" * (4 * reply_tokens)] * len(sc.turns)
    return max(count_messages(baseline_messages(sc, n, replies[:n - 1])) for n in range(1, len(sc.turns) + 1))


def run_baseline(sc: Scenario, main: RecordingCore, script: SimScript | None, progress=None) -> ArmResult:
    arm, replies = ArmResult("baseline", recorder=main.rec), []
    for t in sc.turns:
        main.rec.turn = t.i
        if script is not None:
            script.answer, script.route = t.answer, "api"
        msgs = baseline_messages(sc, t.i, replies)
        t0 = time.perf_counter()
        reply = main.chat(LLMSyscall(messages=msgs, temperature=0, max_tokens=512)).text
        wall = time.perf_counter() - t0
        replies.append(reply)
        secs, over = _turn_seconds(main.rec, t.i, wall)
        arm.outcomes.append(Outcome(t.i, t.kind, reply, secs, over, "main", check(t, reply)))
        if progress:
            progress("baseline", t, arm.outcomes[-1])
    return arm


def run_pal(sc: Scenario, main: RecordingCore, helper: RecordingCore, script: SimScript | None, progress=None) -> ArmResult:
    from pal.chat import ChatService
    from pal.config import load_settings
    from pal.knowledge_graph import KnowledgeGraph

    arm = ArmResult("pal", recorder=main.rec)
    home = tempfile.mkdtemp(prefix="pal-demo-")
    settings = load_settings(env={}, overrides={
        "storage_dir": home, "knowledge_graph": {"dir": home + "/graph"}, "memory": {"enabled": False},
        "tools": {"enabled": True}, "server": {"session_ttl_minutes": 240}})
    svc = ChatService(settings, local_core=helper, graph=KnowledgeGraph(settings.knowledge_graph))
    svc.api_core = lambda provider, model, key=None: main
    try:
        for t in sc.turns:
            main.rec.turn = helper.rec.turn = t.i
            if script is not None:
                script.answer, script.short_answer, script.route = t.answer, t.short_answer, t.route
            t0 = time.perf_counter()
            attachments = []
            if t.attach == "document":
                attachments = [{"name": sc.document_name, "text": sc.document, "kind": "document"}]
            elif t.attach == "project":
                attachments = [svc.graph.ingest("demo", f"{sc.project_name}.zip", sc.project_zip())]   # building the graph is paid for here
            res = svc.send("demo", "local", main.model, None, t.text, attachments)
            wall = time.perf_counter() - t0
            secs, over = _turn_seconds(main.rec, t.i, wall)
            arm.outcomes.append(Outcome(t.i, t.kind, res.reply, secs, over, res.handled_by, check(t, res.reply)))
            if progress:
                progress("pal", t, arm.outcomes[-1])
        arm.notes["metrics"] = svc.metrics.chat_summary()
    finally:
        svc.close()
    return arm
