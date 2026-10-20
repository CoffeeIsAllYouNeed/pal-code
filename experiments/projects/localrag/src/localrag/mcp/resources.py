"""The documents as MCP resources: a client can list them and read their full text by uri (localrag://doc/<id>)."""
from __future__ import annotations

import json

from localrag.engine import LocalRAG
from localrag.mcp.registry import Resource, ResourceRegistry

DOC_URI = "localrag://doc/"
STATS_URI = "localrag://stats"


def build_resources(engine: LocalRAG) -> ResourceRegistry:
    """The resource registry. The list is computed on each request, so it follows documents as they are added and removed."""
    def lister() -> list[Resource]:
        items = [Resource(STATS_URI, "index statistics", "application/json", lambda: json.dumps(engine.stats(), indent=2),
                          "Counts of documents, chunks and cache use")]
        for d in engine.index.store.documents():
            mime = "text/markdown" if d.metadata.get("format") == "markdown" else "text/plain"
            items.append(Resource(f"{DOC_URI}{d.id}", d.source, mime, lambda d=d: d.text, str(d.metadata.get("title", ""))))
        return items

    return ResourceRegistry(lister, [{"uriTemplate": DOC_URI + "{id}", "name": "document", "mimeType": "text/plain",
                                      "description": "The full text of an indexed document"}])
