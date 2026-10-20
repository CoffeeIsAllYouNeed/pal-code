"""An exact vector index: every query is compared with every stored vector."""
from __future__ import annotations

from typing import Any

from localrag.core.errors import IndexingError
from localrag.embed.similarity import dot, top_k


class FlatVectorIndex:
    kind = "flat"

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: dict[str, list[float]] = {}

    def add(self, item_id: str, vector: list[float]) -> None:
        """Store a unit-length vector under an id, replacing any earlier vector with that id."""
        if len(vector) != self.dim:
            raise IndexingError(f"vector has {len(vector)} values, the index expects {self.dim}")
        self._vectors[item_id] = vector

    def remove(self, item_id: str) -> bool:
        """Forget a vector; False if the id was not stored."""
        return self._vectors.pop(item_id, None) is not None

    def vector(self, item_id: str) -> list[float]:
        if item_id not in self._vectors:
            raise IndexingError(f"no vector stored for {item_id}")
        return self._vectors[item_id]

    def search(self, query: list[float], k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """The k stored ids most similar to the query, best first, optionally only among the allowed ids."""
        if len(query) != self.dim:
            raise IndexingError(f"query has {len(query)} values, the index expects {self.dim}")
        pool = self._vectors if allowed is None else {i: v for i, v in self._vectors.items() if i in allowed}
        return top_k(((i, dot(query, v)) for i, v in pool.items()), k)

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._vectors

    def ids(self) -> list[str]:
        return sorted(self._vectors)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "dim": self.dim, "vectors": self._vectors}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlatVectorIndex:
        index = cls(int(data["dim"]))
        for item_id, vec in data["vectors"].items():
            index.add(item_id, vec)
        return index
