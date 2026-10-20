import time
from dataclasses import dataclass, field

from pal.context_manager.utils.relevance import recency_score, text_similarity


@dataclass
class Segment:
    id: str
    content: str
    role: str = "user"
    timestamp: float = field(default_factory=time.time)


class ContextPager:
    def __init__(self, hot_window: int = 6, cold_threshold: float = 0.35):
        self.hot_window = hot_window
        self.cold_threshold = cold_threshold
        self.cold_store: dict[str, Segment] = {}

    def page_out(self, history: list[Segment]) -> tuple[list[Segment], list[str]]:
        if len(history) <= self.hot_window:
            return history, []

        hot = history[-self.hot_window:]
        cold = history[: -self.hot_window]
        cold_refs = []
        for seg in cold:
            self.cold_store[seg.id] = seg
            cold_refs.append(seg.id)
        return hot, cold_refs

    def page_in(self, query: str, cold_refs: list[str], top_k: int = 2) -> list[Segment]:
        candidates = [self.cold_store[r] for r in cold_refs if r in self.cold_store]
        if not candidates:
            return []

        texts = [c.content for c in candidates]
        sims = text_similarity(query, texts)  # embeddings if available, keyword overlap otherwise
        now = time.time()

        scored = []
        for seg, sim in zip(candidates, sims):
            score = 0.7 * sim + 0.3 * recency_score(seg.timestamp, now)
            scored.append((score, seg))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [seg for score, seg in scored[:top_k] if score >= self.cold_threshold]
