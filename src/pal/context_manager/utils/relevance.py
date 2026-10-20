import re
import time

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _EMBEDDER = None
    _EMBED_AVAILABLE = True
except Exception:  # missing OR broken (e.g. a torch DLL error)
    _EMBED_AVAILABLE = False


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def embedding_similarity(query: str, segments: list[str]) -> list[float]:
    if not _EMBED_AVAILABLE or not segments:
        return [0.0] * len(segments)
    model = _get_embedder()
    q_vec = model.encode([query])[0]
    s_vecs = model.encode(segments)
    sims = s_vecs @ q_vec / (np.linalg.norm(s_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-8)
    return sims.tolist()


def recency_score(timestamp: float, now: float = None) -> float:
    now = now or time.time()
    age = max(0.0, now - timestamp)
    return 1.0 / (1.0 + age / 60.0)


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset("a an the and or but if of to in on at for with by from as is are was were be been it this that "
                  "these those i you he she we they me my your our their what which who how why when where do does did "
                  "can could should would will please".split())


def _keywords(text: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD.findall(text)) if w not in _STOP and (len(w) > 1 or w.isdigit())}


def keyword_overlap(query: str, text: str) -> float:
    """Share of the query's meaningful words that also appear in `text` (0 to 1). No model needed."""
    q, t = _keywords(query), _keywords(text)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def text_similarity(query: str, texts: list[str]) -> list[float]:
    """Embedding similarity when sentence-transformers is installed, keyword overlap otherwise."""
    sims = embedding_similarity(query, texts)
    if texts and all(s == 0.0 for s in sims):
        sims = [keyword_overlap(query, t) for t in texts]
    return sims
