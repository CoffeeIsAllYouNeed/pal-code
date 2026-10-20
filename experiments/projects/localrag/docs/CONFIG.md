# Configuration reference

Every setting has a default, so LocalRAG works with no configuration. Override settings in a JSON file (`--config file.json`), with nested objects named as below, or with the `LOCALRAG_*` environment variables listed at the end. Unknown names and values of the wrong type are errors. This file is generated from `src/localrag/core/config.py` by `scripts/gen_config_doc.py`.

Example `localrag.json`:

```json
{"chunk": {"chunk_size": 800}, "retrieve": {"top_k": 8}, "index": {"vector_index": "ivf"}}
```

## `chunk`

How documents are cut into chunks.

| Setting | Default | Meaning |
|---|---|---|
| `chunk_size` | `512` | target characters per chunk |
| `chunk_overlap` | `64` | characters shared by neighbouring chunks |
| `chunk_strategy` | `'auto'` | auto, fixed, sentence, markdown or code |
| `min_chunk_chars` | `40` | chunks shorter than this are merged into a neighbour |

## `embed`

How text becomes vectors.

| Setting | Default | Meaning |
|---|---|---|
| `embedding_name` | `'hashing'` | which embedder to use |
| `embedding_dim` | `256` | length of every vector |
| `word_weight` | `1.0` | weight of single words in a vector |
| `bigram_weight` | `0.5` | weight of word pairs |
| `char_ngram_weight` | `0.3` | weight of character n-grams |
| `embedding_cache_size` | `2048` | texts whose vectors are remembered |

## `index`

The vector index, the keyword index and how their results are fused.

| Setting | Default | Meaning |
|---|---|---|
| `vector_index` | `'flat'` | flat (exact) or ivf (approximate) |
| `ivf_lists` | `16` | clusters in the ivf index |
| `ivf_probes` | `4` | clusters searched per query |
| `bm25_k1` | `1.5` | BM25 term-frequency saturation |
| `bm25_b` | `0.75` | BM25 length normalisation |
| `fusion_method` | `'rrf'` | rrf (reciprocal rank fusion) or weighted |
| `rrf_k` | `60` | damping constant of reciprocal rank fusion |
| `hybrid_alpha` | `0.5` | weight of the vector scores in weighted fusion |

## `retrieve`

How a query is answered.

| Setting | Default | Meaning |
|---|---|---|
| `top_k` | `5` | hits returned |
| `candidate_k` | `30` | candidates fetched before reranking and diversification |
| `mmr_lambda` | `0.7` | 1.0 = pure relevance, 0.0 = pure diversity |
| `min_score` | `0.15` | hits scoring below this (after reranking, 0 to 1) are dropped |
| `use_reranker` | `True` | re-score candidates by how many query terms they contain |
| `context_tokens` | `1500` | token budget of the context text handed to a model |
| `query_cache_size` | `256` | query results remembered |

## `tree`

The document tree and tree search.

| Setting | Default | Meaning |
|---|---|---|
| `tree_max_depth` | `4` | deepest heading level that becomes a tree node |
| `summary_sentences` | `2` | sentences in a node's extractive summary |
| `tree_beam` | `2` | branches followed at each level of a tree search |

## `server`

The MCP server.

| Setting | Default | Meaning |
|---|---|---|
| `server_name` | `'localrag'` |  |
| `server_version` | `'0.1.0'` |  |
| `protocol_version` | `'2025-06-18'` | newest MCP revision this server speaks |
| `rate_limit_per_minute` | `600` | requests allowed per minute |
| `max_message_bytes` | `1000000` | larger incoming messages are refused |
| `auth_token` | `''` | when set, clients must send it in initialize |

## Top level

| Setting | Default | Meaning |
|---|---|---|
| `data_dir` | `'.localrag'` | Where the snapshot file is kept. |

## Environment variables

| Variable | Setting |
|---|---|
| `LOCALRAG_DATA_DIR` | `data_dir` |
| `LOCALRAG_CHUNK_SIZE` | `chunk.chunk_size` |
| `LOCALRAG_CHUNK_OVERLAP` | `chunk.chunk_overlap` |
| `LOCALRAG_EMBEDDING_DIM` | `embed.embedding_dim` |
| `LOCALRAG_TOP_K` | `retrieve.top_k` |
| `LOCALRAG_VECTOR_INDEX` | `index.vector_index` |
| `LOCALRAG_AUTH_TOKEN` | `server.auth_token` |
