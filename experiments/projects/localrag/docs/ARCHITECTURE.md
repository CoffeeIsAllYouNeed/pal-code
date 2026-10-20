# Architecture

LocalRAG is a pipeline of small layers. Each layer depends only on the ones above it in this list.

1. **core**: `errors` (one base class, `LocalRAGError`), `config` (all settings and defaults), `ids` (stable hashes), `textutil`
   (normalising, tokenising, stemming, sentence splitting) and `types` (`Document`, `Chunk`, `SearchHit`, `Citation`, `QueryResult`).
2. **ingest**: bytes in, `Document` and `Chunk` objects out. A *loader* turns one file type into text; a *chunker* decides where to cut;
   `NearDuplicateFilter` drops chunks that repeat an earlier chunk of the same file.
3. **embed**: an `Embedder` turns text into unit-length vectors. `HashingEmbedder` hashes words, word pairs and character n-grams into
   `embedding_dim` slots. `CachedEmbedder` remembers recent texts. `register_embedder` adds new embedders.
4. **index**: `FlatVectorIndex` (exact) and `IVFIndex` (k-means clusters, approximate), `InvertedIndex` (BM25), `fusion`, `filters`,
   `ChunkStore` (documents and chunks), `persistence` (atomic, checksummed snapshots) and `HybridIndex`, which keeps the vector and keyword
   indexes in step.
5. **retrieve**: `process_query`, the rerankers, `mmr_select`, citations, the context builder, `QueryCache` and `RetrievalPipeline`.
6. **tree**: `build_tree` makes a section tree from headings, `summarize` writes extractive summaries, `TreeIndex` searches the trees.
7. **engine**: `LocalRAG` owns one instance of everything above and takes a lock around changes, so it can be shared between threads.
8. **mcp** and **cli**: two front ends over the engine.

## Data flow

Ingestion: `ingest_text(text, source)` -> loader (chosen by the file extension) -> `Document` -> chunker (chosen by `chunk_strategy`, or by
file type when it is `auto`) -> near-duplicate filter -> `HybridIndex.add_document` (store, embed, vector index, keyword index) -> tree
built from the document -> query cache cleared.

Search: `search(query)` -> `process_query` -> `HybridIndex.search` (vector and keyword candidates, fused) -> reranker -> MMR -> citations
and context -> `QueryResult`, cached by query text, `top_k` and filters.

## Decisions worth knowing

- **Pure Python, no dependencies.** The package installs anywhere and is easy to read. The cost is speed on very large collections.
- **Feature hashing instead of a trained model.** Embeddings are deterministic, need no download and no GPU, and cost almost nothing. They
  capture vocabulary overlap, so the keyword index and the reranker carry most of the precision.
- **Hybrid search.** The vector index finds chunks with similar wording, BM25 finds exact rare terms. Reciprocal rank fusion combines the
  two rankings without needing their scores to be comparable.
- **Chunk text is indexed together with its heading or symbol name**, because a heading such as "Install" or a function name such as
  `compute_fee` often says what a chunk is about when its body does not.
- **Snapshots store text, not vectors.** They stay small and cannot go stale when the embedder changes; the indexes are rebuilt on load.
  Writes go through a temporary file and `os.replace`, and loading verifies a SHA-256 checksum.
- **Errors are values the caller can handle.** A failure inside an MCP tool becomes a result with `isError: true`, so one bad call cannot
  stop the server.

## Security

The MCP server refuses messages larger than `max_message_bytes`, limits requests per minute with a token bucket, can require an
`authToken` in `initialize` (compared in constant time), and lets `ingest_path` read only inside folders given with `--allow`.
