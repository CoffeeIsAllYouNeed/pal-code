"""
Vector store for agent knowledge bases -- backed by chromadb
(pip install chromadb) when it's installed, one collection per agent so
different agents' knowledge bases stay isolated. Falls back to a small
in-memory store, scored via the same embedding/keyword-overlap
similarity helpers context_manager and memory already use, when
chromadb isn't available. Functionally equivalent at the scale this
project targets -- the real difference is chromadb's store is disk-
backed (survives a restart), the fallback isn't.
"""
import logging
from typing import Optional

from pal.context_manager.utils.relevance import embedding_similarity
from pal.memory.utils.keyword_overlap import keyword_overlap

log = logging.getLogger(__name__)

try:
    import chromadb
    _CHROMADB_AVAILABLE = True
except Exception:  # missing OR broken (e.g. a torch DLL error)
    _CHROMADB_AVAILABLE = False


class _NaiveVectorStore:
    """In-process fallback: one dict of doc_id -> (text, metadata) per
    agent, scored at query time rather than indexed ahead of time."""

    def __init__(self):
        self._collections: dict[str, dict[str, tuple[str, dict]]] = {}

    def _collection(self, agent_id: str) -> dict:
        return self._collections.setdefault(agent_id, {})

    def upsert(self, agent_id: str, doc_id: str, text: str, metadata: dict) -> None:
        self._collection(agent_id)[doc_id] = (text, metadata)

    def delete(self, agent_id: str, doc_id: str) -> None:
        self._collection(agent_id).pop(doc_id, None)

    def query(self, agent_id: str, query_text: str, top_k: int) -> list[dict]:
        collection = self._collection(agent_id)
        if not collection:
            return []
        doc_ids = list(collection.keys())
        texts = [collection[d][0] for d in doc_ids]

        sims = embedding_similarity(query_text, texts)
        if all(s == 0.0 for s in sims):
            sims = [keyword_overlap(query_text, t) for t in texts]

        scored = sorted(zip(doc_ids, sims), key=lambda x: x[1], reverse=True)
        return [
            {"id": doc_id, "text": collection[doc_id][0], "metadata": collection[doc_id][1], "score": score}
            for doc_id, score in scored[:top_k]
        ]


class VectorStoreBackend:
    def __init__(self, persist_directory: Optional[str] = None):
        self.backend = "chromadb" if _CHROMADB_AVAILABLE else "naive_fallback"
        if _CHROMADB_AVAILABLE:
            self._client = (
                chromadb.PersistentClient(path=persist_directory)
                if persist_directory
                else chromadb.EphemeralClient()
            )
            self._collections = {}
        else:
            self._naive = _NaiveVectorStore()

    def _chroma_collection(self, agent_id: str):
        if agent_id not in self._collections:
            # chromadb collection names have length/charset
            # restrictions -- prefix and truncate to stay valid while
            # keeping per-agent namespacing.
            safe_name = f"agent_{agent_id}".replace(" ", "_")[:63]
            self._collections[agent_id] = self._client.get_or_create_collection(safe_name)
        return self._collections[agent_id]

    def delete_agent(self, agent_id: str) -> None:
        """Drop an agent's whole collection."""
        if self.backend == "chromadb":
            self._collections.pop(agent_id, None)
            try:
                self._client.delete_collection(f"agent_{agent_id}".replace(" ", "_")[:63])
            except Exception:  # noqa: BLE001  (collection was never created)
                log.debug("no collection to delete for %s", agent_id)
        else:
            self._naive._collections.pop(agent_id, None)

    def upsert(self, agent_id: str, doc_id: str, text: str, metadata: dict = None) -> None:
        metadata = metadata or {}
        if self.backend == "chromadb":
            self._chroma_collection(agent_id).upsert(
                ids=[doc_id], documents=[text], metadatas=[metadata] if metadata else None
            )
        else:
            self._naive.upsert(agent_id, doc_id, text, metadata)

    def delete(self, agent_id: str, doc_id: str) -> None:
        if self.backend == "chromadb":
            self._chroma_collection(agent_id).delete(ids=[doc_id])
        else:
            self._naive.delete(agent_id, doc_id)

    def query(self, agent_id: str, query_text: str, top_k: int = 3) -> list[dict]:
        if self.backend == "chromadb":
            collection = self._chroma_collection(agent_id)
            if collection.count() == 0:
                return []
            result = collection.query(query_texts=[query_text], n_results=min(top_k, collection.count()))
            ids = result.get("ids", [[]])[0]
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0] or [{}] * len(ids)
            dists = result.get("distances", [[]])[0] or [None] * len(ids)
            return [
                {"id": doc_id, "text": text, "metadata": meta, "score": (1 - dist) if dist is not None else None}
                for doc_id, text, meta, dist in zip(ids, docs, metas, dists)
            ]
        return self._naive.query(agent_id, query_text, top_k)
