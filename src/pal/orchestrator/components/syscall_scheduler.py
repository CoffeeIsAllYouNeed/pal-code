"""
Centralized system-call scheduler (AIOS section 3.3).

All LLM inference requests -- from Pipeline.run(), from orchestrator
Agents, from anywhere else in the system -- get submitted here as
SyscallRequests instead of each caller hitting an LLMWrapper directly.
That's the actual point of centralizing: request management
(queueing, ordering, time-slicing) lives in exactly one place, so
every other module only has to focus on producing/consuming a
request/result pair, not on how or when its turn comes.

Two scheduling strategies, same as AIOS:
- FIFO: process queued requests in arrival order, straight through to
  completion. Simple, but a long request can make everything behind it
  wait its full duration.
- Round Robin: give each queued request a bounded time slice
  (`quantum`), then move to the next one, cycling back to unfinished
  requests later. This is what needs the context-interrupt mechanism in
  context_interrupt.py -- pausing an LLM generation mid-stream and
  resuming it later without restarting from scratch.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, LLMWrapper
from pal.orchestrator.components.context_interrupt import ContextInterruptController, ContextSnapshot


class SchedulingStrategy(str, Enum):
    FIFO = "fifo"
    ROUND_ROBIN = "round_robin"


class SyscallStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class SyscallRequest:
    """One queued LLM inference request. `core` is whichever LLMWrapper
    the submitting module wants this run against -- the scheduler
    doesn't care which deployment option it is, only when it runs."""

    agent_name: str
    core: LLMWrapper
    syscall: LLMSyscall
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: float = field(default_factory=time.time)
    priority: int = 0  # FIFO ignores this; reserved for future strategies

    # internal scheduling state
    status: SyscallStatus = SyscallStatus.QUEUED
    accumulated_text: str = ""
    slices_run: int = 0
    result: Optional[LLMResponse] = None
    error: Optional[str] = None
    # Set when a Round Robin slice is interrupted, so the next tick
    # knows to RESUME (text-based: via the continuation baked into
    # `syscall` already; logits-based: via snapshot.beam_state) rather
    # than starting a fresh generation.
    snapshot: Optional[ContextSnapshot] = None


class SyscallScheduler:
    """
    One centralized queue for every LLM syscall in the system,
    regardless of which module submitted it. Callers submit() a request
    and get back an id; results are collected via get_result() once
    processing has advanced far enough (run_fifo() / run_round_robin()
    drive that, they aren't automatic/background by default -- call
    tick() from whatever event loop the rest of the system already
    has, e.g. the orchestrator's own tick()).
    """

    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.FIFO, rr_quantum: float = 2.0):
        self.strategy = strategy
        self.rr_quantum = rr_quantum
        self.queue: deque[SyscallRequest] = deque()
        self.requests: dict[str, SyscallRequest] = {}
        self._interrupt_controller = ContextInterruptController()

    def submit(self, agent_name: str, core: LLMWrapper, syscall: LLMSyscall, priority: int = 0) -> str:
        req = SyscallRequest(agent_name=agent_name, core=core, syscall=syscall, priority=priority)
        self.requests[req.id] = req
        self.queue.append(req)
        return req.id

    def get_result(self, request_id: str) -> SyscallRequest:
        return self.requests[request_id]

    def pending_count(self) -> int:
        return len(self.queue)

    # -- scheduling strategies -------------------------------------------------

    def run_fifo_once(self) -> Optional[SyscallRequest]:
        """Pops the head of the queue and runs it straight through to
        completion. No interruption -- a long request occupies the
        core for its full duration before the next one gets to start,
        by design."""
        if not self.queue:
            return None

        req = self.queue.popleft()
        req.status = SyscallStatus.RUNNING
        try:
            req.result = req.core.chat(req.syscall)
            req.accumulated_text = req.result.text
            req.status = SyscallStatus.DONE
        except Exception as e:
            req.error = str(e)
            req.status = SyscallStatus.ERROR
        return req

    def run_round_robin_tick(self) -> Optional[SyscallRequest]:
        """Runs the head-of-queue request for at most `rr_quantum`
        seconds, via ContextInterruptController -- which picks the
        text-based or logits-based snapshot approach depending on what
        the request's core supports. If it finishes within that slice,
        it's marked DONE and NOT re-queued. If the slice runs out
        first:
        - text-based: the partial output is folded into a continuation
          syscall (build_continuation) and pushed to the back of the
          queue.
        - logits-based: the request's `syscall` is left UNCHANGED
          (nothing to rebuild -- the search-tree state itself is what
          gets resumed) and its `snapshot` is stored for the next tick
          to resume from directly.
        Either way, every other queued request gets its own slice
        before this one is revisited -- that's the "round" in Round
        Robin."""
        if not self.queue:
            return None

        req = self.queue.popleft()
        req.status = SyscallStatus.RUNNING
        try:
            partial = self._interrupt_controller.run_slice(
                req.core, req.syscall, self.rr_quantum, resume_snapshot=req.snapshot
            )
        except Exception as e:
            req.error = str(e)
            req.status = SyscallStatus.ERROR
            return req

        req.slices_run += 1
        if partial.cumulative:
            # Logits mode: beam decoding naturally tracks the full
            # sequence generated so far, so this IS the total, not a
            # delta to append.
            req.accumulated_text = partial.text_so_far
        else:
            # Text mode: each resumed stream only yields its own new
            # continuation text.
            req.accumulated_text += partial.text_so_far

        if partial.complete:
            req.result = LLMResponse(text=req.accumulated_text)
            req.status = SyscallStatus.DONE
            req.snapshot = None
        else:
            if partial.snapshot.mode == "text":
                req.syscall = self._interrupt_controller.build_continuation(req.syscall, partial)
                req.snapshot = None
            else:
                req.snapshot = partial.snapshot
            req.status = SyscallStatus.QUEUED
            self.queue.append(req)

        return req

    def tick(self) -> Optional[SyscallRequest]:
        """Advances the scheduler by exactly one step, using whichever
        strategy this instance was configured with."""
        if self.strategy == SchedulingStrategy.FIFO:
            return self.run_fifo_once()
        return self.run_round_robin_tick()

    def drain(self, max_ticks: int = 10_000) -> list[SyscallRequest]:
        """Runs tick() until the queue is empty (every request reaches
        DONE or ERROR) or max_ticks is hit, as a safety bound against
        an unbounded RR cycle. Returns the requests in the order they
        finished."""
        finished = []
        ticks = 0
        while self.queue and ticks < max_ticks:
            req = self.tick()
            ticks += 1
            if req is not None and req.status in (SyscallStatus.DONE, SyscallStatus.ERROR):
                finished.append(req)
        return finished
