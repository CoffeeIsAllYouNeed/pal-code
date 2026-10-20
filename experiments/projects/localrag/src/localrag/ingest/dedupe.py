"""Near-duplicate detection: two chunks that share most of their five-word shingles are the same text with small edits."""
from __future__ import annotations

from collections import defaultdict

from localrag.core.ids import stable_hash
from localrag.core.textutil import tokenize


def shingles(text: str, k: int = 5) -> frozenset[str]:
    """The set of hashed k-word windows of the text; texts shorter than k words give one shingle of the whole text."""
    words = tokenize(text, drop_stopwords=False, min_len=1)
    if len(words) <= k:
        return frozenset({stable_hash(" ".join(words), length=8)})
    return frozenset(stable_hash(" ".join(words[i:i + k]), length=8) for i in range(len(words) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """The overlap of two shingle sets: shared shingles divided by all distinct shingles."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NearDuplicateFilter:
    """Remembers the shingles of the texts it has seen and says whether a new text repeats one of them."""

    def __init__(self, threshold: float = 0.9, k: int = 5):
        self.threshold, self.k = threshold, k
        self._seen: list[frozenset[str]] = []
        self._by_shingle: dict[str, list[int]] = defaultdict(list)

    def is_duplicate(self, text: str) -> bool:
        sh = shingles(text, self.k)
        candidates = {i for s in sh for i in self._by_shingle.get(s, ())}
        return any(jaccard(sh, self._seen[i]) >= self.threshold for i in candidates)

    def add(self, text: str) -> None:
        sh = shingles(text, self.k)
        self._seen.append(sh)
        for s in sh:
            self._by_shingle[s].append(len(self._seen) - 1)

    def check_and_add(self, text: str) -> bool:
        """True if the text is new (and is now remembered); False if it duplicates an earlier one."""
        if self.is_duplicate(text):
            return False
        self.add(text)
        return True

    def __len__(self) -> int:
        return len(self._seen)
