"""Light token savers for calls that come from agents Pal does not control (adapted frameworks).

The full token-saving pipeline (input trimming by a local model, document and code-graph retrieval, paged history,
output rules) belongs to the chat turn, where Pal owns the prompt. A framework such as AutoGen owns its own
prompts, so here only safe, deterministic steps run:
  * tidy: trailing spaces and runs of blank lines are removed outside code blocks (code is never touched);
  * optional exact-repeat cache: an identical request inside the TTL is answered with no model call. Off by default,
    because agent loops sometimes retry on purpose and expect a different answer.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

from pal.llm_wrapper.components.base import LLMResponse

_TRAILING = re.compile(r"[ \t]+(?=\n)")
_BLANKS = re.compile(r"\n{3,}")


def tidy(text: str) -> str:
    parts = text.split("```")
    for i in range(0, len(parts), 2):            # even parts are outside fenced code
        parts[i] = _BLANKS.sub("\n\n", _TRAILING.sub("", parts[i]))
    return "```".join(parts)


@dataclass
class CallStats:
    calls: int = 0
    cache_hits: int = 0
    chars_tidied: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class CallSavers:
    def __init__(self, tidy_messages: bool = True, cache: bool = False, cache_size: int = 64, cache_ttl: float = 300.0):
        self.tidy_messages = tidy_messages
        self.cache = cache
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.stats = CallStats()
        self._lock = threading.Lock()
        self._store: OrderedDict[str, tuple[float, LLMResponse]] = OrderedDict()

    def prepare(self, messages: list[dict]) -> list[dict]:
        if not self.tidy_messages:
            return messages
        out = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                t = tidy(c)
                self.stats.chars_tidied += len(c) - len(t)
                m = {**m, "content": t}
            out.append(m)
        return out

    def key(self, model: str, messages: list[dict], kwargs: dict) -> str:
        blob = json.dumps([model, messages, sorted(kwargs.items())], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        if not self.cache:
            return None
        with self._lock:
            hit = self._store.get(key)
            if hit is None or time.monotonic() - hit[0] > self.cache_ttl:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            self.stats.cache_hits += 1
            return replace(hit[1], prompt_tokens=0, completion_tokens=0, meta={**hit[1].meta, "cache": True})

    def put(self, key: str, resp: LLMResponse) -> None:
        if not self.cache:
            return
        with self._lock:
            self._store[key] = (time.monotonic(), resp)
            while len(self._store) > self.cache_size:
                self._store.popitem(last=False)
