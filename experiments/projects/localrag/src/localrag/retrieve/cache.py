"""A small least-recently-used cache of query results, emptied whenever the index changes."""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from localrag.core.types import QueryResult


class QueryCache:
    def __init__(self, size: int = 256):
        self.size = size
        self._items: OrderedDict[str, QueryResult] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(query: str, top_k: int, filters: dict[str, Any] | None) -> str:
        return json.dumps([query, top_k, filters or {}], sort_keys=True)

    def get(self, key: str) -> QueryResult | None:
        found = self._items.get(key)
        if found is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return found

    def put(self, key: str, result: QueryResult) -> None:
        if self.size <= 0:
            return
        self._items[key] = result
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        """Forget everything; called when documents are added or removed, since old answers may no longer be right."""
        self._items.clear()

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._items)}
