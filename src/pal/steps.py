"""Live steps of a request, for the screen: "Trimming your message", "Asking mistral:7b", "Checking the syntax" ... Each step is announced
when it starts and when it ends (with a short detail and its time). Nothing listens by default, so this costs nothing outside pal cli.

    with steps.listen(callback):          # callback(event, key, text) with event "begin" or "end"
        ...
    with steps.step("context", "Gathering the relevant history, documents and code") as s:
        ...
        s.detail = "3 document passages"
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_LISTENERS: list[Callable[[str, str, str], None]] = []
_LOCK = threading.Lock()
_ACTIVE = threading.local()          # the steps running in this thread, innermost last

# the engine's timed stages, in plain words
LABELS = {
    "direct": "Checking for a direct answer (no model needed)",
    "input_saver": "Trimming your message",
    "context": "Picking the relevant history, documents and code",
}


def _emit(event: str, key: str, text: str) -> None:
    with _LOCK:
        listeners = list(_LISTENERS)
    for cb in listeners:
        try:
            cb(event, key, text)
        except Exception:  # noqa: BLE001  (a display problem never stops the work)
            pass


class _Step:
    def __init__(self) -> None:
        self.detail = ""


@contextmanager
def step(key: str, text: str | None = None) -> Iterator[_Step]:
    s = _Step()
    if not _LISTENERS:
        yield s
        return
    label = text or LABELS.get(key, key)
    t0 = time.perf_counter()
    stack = _ACTIVE.__dict__.setdefault("stack", [])
    stack.append(s)
    _emit("begin", key, label)
    try:
        yield s
    finally:
        stack.pop()
        took = time.perf_counter() - t0
        _emit("end", key, label + (f"  ·  {s.detail}" if s.detail else "") + (f"  ({took:.1f}s)" if took >= 0.1 else ""))


def detail(text: str) -> None:
    """What the step running now did, shown after its name ("12 tokens less")."""
    stack = getattr(_ACTIVE, "stack", None)
    if stack:
        stack[-1].detail = text


def note(key: str, text: str) -> None:
    """A step that is done at once (no waiting)."""
    if _LISTENERS:
        _emit("end", key, text)


@contextmanager
def listen(callback: Callable[[str, str, str], None]) -> Iterator[None]:
    with _LOCK:
        _LISTENERS.append(callback)
    try:
        yield
    finally:
        with _LOCK:
            if callback in _LISTENERS:
                _LISTENERS.remove(callback)


# -- the reply as it is written: the screen shows it while the model streams it ------------------------------------------------------
_TOKENS: list[Callable[[str], None]] = []


def wants_tokens() -> bool:
    return bool(_TOKENS)


def token(piece: str) -> None:
    for cb in list(_TOKENS):
        try:
            cb(piece)
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def stream_to(callback: Callable[[str], None]) -> Iterator[None]:
    with _LOCK:
        _TOKENS.append(callback)
    try:
        yield
    finally:
        with _LOCK:
            if callback in _TOKENS:
                _TOKENS.remove(callback)


# -- stopping a reply: (s) in the stream view ends the model call at the next streamed piece ------------------------------------------
class Stopped(BaseException):
    """Raised inside the model call when the person stopped the reply (a BaseException, so no fallback path swallows it)."""


_STOP = threading.Event()


def request_stop() -> None:
    _STOP.set()


def clear_stop() -> None:
    _STOP.clear()


def stop_requested() -> bool:
    return _STOP.is_set()


def check_stop() -> None:
    if _STOP.is_set():
        raise Stopped()
