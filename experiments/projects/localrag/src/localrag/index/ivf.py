"""An approximate vector index: vectors are grouped into clusters (k-means) and a query only searches the closest few clusters."""
from __future__ import annotations

from localrag.embed.similarity import centroid, dot, normalize, top_k
from localrag.index.vector import FlatVectorIndex


class IVFIndex(FlatVectorIndex):
    kind = "ivf"

    def __init__(self, dim: int, n_lists: int = 16, n_probes: int = 4):
        super().__init__(dim)
        self.n_lists, self.n_probes = n_lists, n_probes
        self._centroids: list[list[float]] = []
        self._lists: list[list[str]] = []
        self._dirty = True

    def add(self, item_id: str, vector: list[float]) -> None:
        super().add(item_id, vector)
        self._dirty = True

    def remove(self, item_id: str) -> bool:
        removed = super().remove(item_id)
        self._dirty = self._dirty or removed
        return removed

    def build(self, iterations: int = 6) -> None:
        """Cluster the stored vectors. Starting centroids are evenly spaced through the sorted ids, so builds are repeatable."""
        ids = self.ids()
        k = min(self.n_lists, len(ids))
        self._centroids = [list(self._vectors[ids[(i * len(ids)) // k]]) for i in range(k)] if k else []
        for _ in range(iterations if k else 0):
            groups: list[list[str]] = [[] for _ in range(k)]
            for item_id in ids:
                groups[self._nearest(self._vectors[item_id], 1)[0]].append(item_id)
            self._centroids = [normalize(centroid((self._vectors[i] for i in g), self.dim)) if g else c for g, c in zip(groups, self._centroids, strict=True)]
        self._lists = [[] for _ in range(k)]
        for item_id in ids:
            self._lists[self._nearest(self._vectors[item_id], 1)[0]].append(item_id)
        self._dirty = False

    def _nearest(self, vector: list[float], n: int) -> list[int]:
        order = top_k(((str(i), dot(vector, c)) for i, c in enumerate(self._centroids)), n)
        return [int(i) for i, _ in order]

    def search(self, query: list[float], k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Search the closest clusters only; small collections are searched exactly, as clustering would not help."""
        if len(self) < self.n_lists * 4:
            return super().search(query, k, allowed)
        if self._dirty:
            self.build()
        candidates = [i for c in self._nearest(query, self.n_probes) for i in self._lists[c] if allowed is None or i in allowed]
        found = top_k(((i, dot(query, self._vectors[i])) for i in candidates), k)
        return found if len(found) >= min(k, len(self)) or allowed is not None else super().search(query, k, allowed)
