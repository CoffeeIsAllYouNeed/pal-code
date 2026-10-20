"""Remembers the vectors of recently embedded texts, so repeated queries and re-ingested chunks cost nothing."""
from __future__ import annotations

from collections import OrderedDict

from localrag.embed.base import Embedder


class CachedEmbedder(Embedder):
    def __init__(self, inner: Embedder, size: int = 2048):
        super().__init__(inner.dim)
        self.name = inner.name
        self.inner, self.size = inner, size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self.misses += len(missing)
            for text, vec in zip(missing, self.inner.embed(missing), strict=True):
                self._cache[text] = vec
        self.hits += len(texts) - len(missing)
        out = []
        for text in texts:
            self._cache.move_to_end(text)
            out.append(self._cache[text])
        while len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return [list(v) for v in out]

    def stats(self) -> dict[str, int]:
        """How many lookups were answered from memory, how many had to be computed, and how many texts are held."""
        return {"hits": self.hits, "misses": self.misses, "size": len(self._cache)}
