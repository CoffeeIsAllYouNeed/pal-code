# LocalRAG

A local-first retrieval-augmented generation (RAG) engine that is also a Model Context Protocol (MCP) server. Point it at your
documents and code, then ask questions from the command line, from Python, or from any MCP client such as an AI assistant. Everything
runs on your machine with no dependencies beyond the Python standard library.

## What it does

- **Ingests** text, markdown, HTML, CSV, JSON and source code. Markdown is cut at headings, Python at function and class boundaries,
  everything else at sentence boundaries. Near-duplicate chunks are dropped.
- **Searches** with a hybrid of vector similarity (a feature-hashing embedder) and BM25 keyword scoring, fused with reciprocal rank
  fusion, then reranked and diversified (maximal marginal relevance). Every result carries a citation.
- **Browses by section** with a PageIndex-style document tree and no vectors at all: sections are scored by title and summary.
- **Serves MCP** over stdio: nine tools, documents as resources, and two prompt templates.
- **Measures itself**: `localrag eval` builds a golden set from your own index and reports recall, precision, MRR, MAP and nDCG.

## Quick start

```
pip install -e .
localrag ingest ./notes ./src
localrag query "how are chunks cut at headings"
localrag tree "installation steps"
localrag eval
```

From Python:

```python
from localrag.engine import LocalRAG

engine = LocalRAG()
engine.ingest_path("./notes")
result = engine.search("how long to boil pasta", top_k=3)
print(result.context)          # numbered passages, ready to give to a model
print(result.citations[0])     # source, character span and a snippet
engine.save()                  # writes .localrag/snapshot.json
```

## Use it from an MCP client

Add LocalRAG to the client's server list (the exact file depends on the client):

```json
{"mcpServers": {"localrag": {"command": "python", "args": ["-m", "localrag", "serve-mcp", "--allow", "/path/to/your/docs"]}}}
```

The `ingest_path` tool can only read inside the folders given with `--allow` (by default the current directory). See `docs/MCP.md`.

## Layout

```
src/localrag/
  core/       errors, settings, ids, text helpers, data types
  ingest/     loaders, chunkers, near-duplicate filter, ingestion pipeline
  embed/      embedders, vector arithmetic, idf, embedding cache
  index/      vector indexes (flat, ivf), BM25, fusion, filters, store, snapshots, hybrid index
  retrieve/   query processing, reranking, MMR, citations, context, cache, retrieval pipeline
  tree/       document tree builder, summaries, tree search, rendering
  mcp/        JSON-RPC, schema validation, registries, middleware, server, transports, tools
  eval/       metrics, golden sets, synthetic golden sets, runner, report
  cli/        the command-line interface
  engine.py   the LocalRAG class that ties everything together
```

More: `docs/ARCHITECTURE.md`, `docs/RETRIEVAL.md`, `docs/MCP.md`, `docs/CONFIG.md`.

## Tests

```
python -m pytest
```

## Limits

The default embedder hashes words and character n-grams, so it matches on shared vocabulary, not on meaning: "car" and "automobile" are
unrelated to it. Plug in a neural embedder with `register_embedder`. Collections of a few hundred thousand chunks are comfortable; vector
search is pure Python. Snapshots store text, not vectors, and rebuild the indexes on load.
