"""The embedder interface: anything that turns text into fixed-length vectors can be used by the indexes."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    name = "base"

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """One unit-length vector of length dim for each text, in order."""

    def embed_one(self, text: str) -> list[float]:
        """The vector of a single text."""
        return self.embed([text])[0]
