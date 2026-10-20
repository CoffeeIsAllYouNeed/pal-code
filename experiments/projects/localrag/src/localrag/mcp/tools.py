"""The tools LocalRAG offers to an AI client: ingest, search, browse by section, fetch, delete and inspect."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from localrag.core.errors import IngestError
from localrag.engine import LocalRAG
from localrag.mcp.registry import Tool, ToolRegistry

_STRING = {"type": "string"}


def _inside(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == r or r in resolved.parents for r in roots)


def build_tools(engine: LocalRAG, allowed_roots: list[Path] | None = None) -> ToolRegistry:
    """Register the tools. ingest_path can only read below the allowed folders (by default the current directory)."""
    roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]
    registry = ToolRegistry()

    def ingest_text(a: dict[str, Any]) -> Any:
        return engine.ingest_text(a["text"], a["source"], a.get("metadata"))

    def ingest_path(a: dict[str, Any]) -> Any:
        target = Path(a["path"]).expanduser()
        if not _inside(target, roots):
            raise IngestError("that path is outside the folders this server may read")
        return engine.ingest_path(target, a["recursive"])

    def search(a: dict[str, Any]) -> Any:
        result = engine.search(a["query"], a["top_k"], a.get("filters"))
        return {"context": result.context, "citations": [c.__dict__ for c in result.citations], "took_ms": round(result.took_ms, 2),
                "hits": [{"rank": h.rank, "score": round(h.score, 4), "chunk_id": h.chunk.id, "source": h.chunk.metadata.get("source"),
                          "retriever": h.retriever} for h in result.hits]}

    def get_document(a: dict[str, Any]) -> Any:
        doc = engine.get_document(a["name"])
        return {"doc_id": doc.id, "source": doc.source, "metadata": doc.metadata, "outline": engine.outline(a["name"]),
                "text": doc.text[:a["max_chars"]], "truncated": len(doc.text) > a["max_chars"]}

    registry.register(Tool("ingest_text", "Add a piece of text to the index. The source name's extension (.md, .html, .py ...) decides how it is read.", {
        "type": "object", "properties": {"text": {**_STRING, "minLength": 1}, "source": {**_STRING, "default": "note.txt"},
                                         "metadata": {"type": "object"}}, "required": ["text"], "additionalProperties": False}, ingest_text, False))
    registry.register(Tool("ingest_path", "Add a file, or every supported file in a folder, to the index.", {
        "type": "object", "properties": {"path": {**_STRING, "minLength": 1}, "recursive": {"type": "boolean", "default": True}},
        "required": ["path"], "additionalProperties": False}, ingest_path, False))
    registry.register(Tool("search", "Find the passages of the indexed documents that best answer a query, with citations. Returns a context text ready to cite as [1], [2].", {
        "type": "object", "properties": {"query": {**_STRING, "minLength": 1, "maxLength": 1000}, "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                                         "filters": {"type": "object", "description": "metadata filter, e.g. {\"source\": \"guide.md\"}"}},
        "required": ["query"], "additionalProperties": False}, search))
    registry.register(Tool("tree_search", "Browse documents by section without vectors: finds the sections whose titles and summaries match the query.", {
        "type": "object", "properties": {"query": {**_STRING, "minLength": 1}, "top_n": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}},
        "required": ["query"], "additionalProperties": False}, lambda a: engine.tree_search(a["query"], a["top_n"])))
    registry.register(Tool("get_document", "Read a document (its outline and the start of its text) by id or source name.", {
        "type": "object", "properties": {"name": {**_STRING, "minLength": 1}, "max_chars": {"type": "integer", "minimum": 100, "maximum": 100000, "default": 4000}},
        "required": ["name"], "additionalProperties": False}, get_document))
    registry.register(Tool("get_chunk", "Read one chunk by its id, as given in search results.", {
        "type": "object", "properties": {"chunk_id": {**_STRING, "minLength": 1}}, "required": ["chunk_id"], "additionalProperties": False},
        lambda a: engine.get_chunk(a["chunk_id"]).to_dict()))
    registry.register(Tool("delete_document", "Remove a document and all its chunks from the index.", {
        "type": "object", "properties": {"name": {**_STRING, "minLength": 1}}, "required": ["name"], "additionalProperties": False},
        lambda a: {"chunks_removed": engine.delete_document(a["name"])}, False))
    registry.register(Tool("list_documents", "List the indexed documents.", {"type": "object", "properties": {}, "additionalProperties": False},
                           lambda a: engine.list_documents()))
    registry.register(Tool("stats", "Show index statistics: documents, chunks, vocabulary and cache use.", {"type": "object", "properties": {}, "additionalProperties": False},
                           lambda a: engine.stats()))
    return registry
