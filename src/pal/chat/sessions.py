"""Chat sessions: one per browser tab. Bounded (max count, idle TTL) and thread-safe, so a
long-running server does not leak memory or disk."""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from pal.chat.tools import ToolBox
from pal.context_manager import Turn
from pal.documents import DocumentIndex
from pal.errors import UserInputError
from pal.sdk.pipeline import AgentSDK

log = logging.getLogger(__name__)

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_session_id(value: str) -> str:
    if not _SESSION_ID.match(value or ""):
        raise UserInputError("invalid session id (1-64 letters, digits, - or _)")
    return value


@dataclass
class Session:
    id: str
    agent_id: str
    sdk: AgentSDK
    docs: DocumentIndex | None
    toolbox: ToolBox
    turns: list[Turn] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    on_close: Callable[[Session], None] | None = None

    def touch(self) -> None:
        self.last_used = time.time()

    def close(self) -> None:
        if self.on_close:
            try:
                self.on_close(self)
            except Exception:
                log.exception("session cleanup failed for %s", self.id[:8])


class SessionStore:
    def __init__(self, factory: Callable[[str], Session], max_sessions: int = 200, ttl_seconds: float = 4 * 3600):
        self._factory = factory
        self.max_sessions = max_sessions
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def get_or_create(self, session_id: str) -> Session:
        validate_session_id(session_id)
        evicted: list[Session] = []
        with self._lock:
            evicted += self._expire_locked()
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = self._factory(session_id)
                self._sessions[session_id] = sess
                while len(self._sessions) > self.max_sessions:
                    _, old = self._sessions.popitem(last=False)
                    evicted.append(old)
            self._sessions.move_to_end(session_id)
            sess.touch()
        for old in evicted:  # cleanup outside the store lock
            with old.lock:
                old.close()
        return sess

    def _expire_locked(self) -> list[Session]:
        now = time.time()
        gone = [k for k, s in self._sessions.items() if now - s.last_used > self.ttl]
        return [self._sessions.pop(k) for k in gone]

    def remove(self, session_id: str) -> bool:
        with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            return False
        with sess.lock:
            sess.close()
        return True

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def close_all(self) -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), OrderedDict()
        for s in sessions:
            with s.lock:
                s.close()
