"""The Notes Manager: everything about the notes the chat keeps across sessions, in one place.

  * WHAT IS KEPT      only messages you typed (never attachments, never model replies), at least `min_chars` long, and never anything that
                      looks like a secret (API keys, passwords, private keys).
  * WHAT IS RECALLED  notes from *other* sessions that match the current question, within a token budget, added to that one request.
  * UPKEEP            old notes beyond a cap are pruned; duplicate notes are merged in the background (no model, an OTHER-class job nobody
                      waits for), every `tidy_every` turns.
  * CONTROL           count, clear, forget one session's notes, and the RAM / disk numbers behind them.

It sits on the Memory Manager (RAM cache, LRU-K eviction) which sits on the Storage Manager (the disk). It owns its own lock, so callers never
have to. Nothing here leaves the machine except the recalled notes that go to the API model inside a request.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from pal.config import MemoryConfig
from pal.llm_scheduler.types import TaskClass
from pal.notes_manager.components.privacy import looks_secret
from pal.notes_manager.components.recall import Recall, build_recall
from pal.notes_manager.components.tidy import tidy_steps
from pal.tokens import estimate_tokens

log = logging.getLogger(__name__)
_MAX_NOTES_PRUNE = 50

__all__ = ["NotesManager", "Recall", "looks_secret"]


class NotesManager:
    def __init__(self, sdk, cfg: MemoryConfig | dict | None = None, count: Callable[[str], int] = estimate_tokens,
                 max_notes: int = 1000, scheduler=None):
        self.sdk = sdk
        self.cfg = MemoryConfig.coerce(cfg)
        self.count = count
        self.max_notes = max_notes
        self.scheduler = scheduler
        self._lock = threading.RLock()
        self._new_notes = 0

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def attach_scheduler(self, scheduler) -> None:
        """The scheduler that runs the background tidy-up (it needs a local line to run on)."""
        self.scheduler = scheduler

    # -- keeping and recalling --------------------------------------------------------------------------------------------
    def remember(self, session_id: str, message: str) -> bool:
        text = message.strip()
        if not self.enabled or len(text) < self.cfg.min_chars or looks_secret(text):
            return False
        try:
            with self._lock:
                self.sdk.remember(text[:1000], tags=["user"], session_id=session_id)
                self._prune()
        except Exception:
            log.exception("memory write failed")  # notes are best-effort: never fail the chat
            return False
        return True

    def recall(self, session_id: str, query: str) -> Recall | None:
        if not self.enabled or not self.cfg.recall_notes or not query.strip():
            return None
        try:
            with self._lock:
                notes = self.sdk.memory.retrieve(query, top_k=self.cfg.recall_notes * 3, min_score=self.cfg.min_score)
        except Exception:
            log.exception("memory recall failed")
            return None
        return build_recall(notes, session_id, self.cfg.recall_notes, self.cfg.budget_tokens, self.count)

    # -- control ------------------------------------------------------------------------------------------------------------
    def count_notes(self) -> int:
        with self._lock:
            return len(self.sdk.memory)

    __len__ = count_notes

    def clear(self) -> int:
        with self._lock:
            n = len(self.sdk.memory)
            self.sdk.memory.clear()
            self._new_notes = 0
            return n

    def forget_session(self, session_id: str) -> int:
        """Delete the notes that one session wrote. Returns how many."""
        with self._lock:
            mine = [n for n in self.sdk.memory.storage.read_all() if n.session_id == session_id]
            for n in mine:
                self.sdk.memory.manager.delete(n.id)
            return len(mine)

    def stats(self) -> dict:
        """How many notes there are, how they split between RAM and disk, how full the RAM cache is, and how many are new since the last tidy."""
        with self._lock:
            return {"enabled": self.enabled, "notes": len(self.sdk.memory), "since_tidy": self._new_notes, "max_notes": self.max_notes, **self.sdk.memory.stats()}

    # -- upkeep -----------------------------------------------------------------------------------------------------------
    def tidy_steps(self, near: float = 0.9, batch: int = 100):
        """Merge duplicate and near-duplicate notes (see components/tidy.py). A generator, so a scheduler can pause it."""
        return tidy_steps(self.sdk.memory, self._lock, near, batch)

    def tidy(self) -> int:
        """Run tidy_steps to the end. Returns how many notes were removed."""
        gen = self.tidy_steps()
        while True:
            try:
                next(gen)
            except StopIteration as done:
                return int(done.value or 0)

    def after_turn(self, ephemeral: bool = False) -> None:
        """Called after each chat turn. Every `tidy_every` turns it hands the tidy-up to the scheduler as a background job."""
        local = getattr(self.scheduler, "local", None)
        if ephemeral or local is None:
            return
        try:
            every = self.cfg.tidy_every
            self._new_notes += 1
            if not every or not self.enabled or self._new_notes < every:
                return
            self._new_notes = 0
            local.submit_steps(self.tidy_steps(), TaskClass.OTHER, "tidy memory")
        except Exception:
            log.exception("background upkeep could not be scheduled")      # never fail the chat because of housekeeping

    def _prune(self) -> None:
        over = len(self.sdk.memory) - self.max_notes
        if over <= 0:
            return
        notes = sorted(self.sdk.memory.storage.read_all(), key=lambda n: (n.retrieval_count, n.timestamp))
        for n in notes[: min(over + _MAX_NOTES_PRUNE, len(notes))]:
            self.sdk.memory.manager.delete(n.id)
