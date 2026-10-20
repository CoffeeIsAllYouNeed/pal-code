# How retrieval works

## Search, step by step

1. **Process the query.** The text is normalised. Inline filters such as `source:guide.md` or `heading:"Getting started"` are removed from
   the text and become metadata filters. Abbreviations are expanded (`db` also searches `database`). A query with no searchable words,
   such as one made only of stopwords, is an error.
2. **Find candidates.** The vector index and the keyword index each return `candidate_k` chunks. The vector index ranks by cosine similarity
   of the hashed vectors; the keyword index ranks by BM25.
3. **Fuse.** With `fusion_method` set to `rrf`, each chunk scores the sum of `1 / (rrf_k + rank)` over the lists it appears in; scores are
   ignored, only ranks matter. With `weighted`, both score lists are rescaled to 0 to 1 and combined as `hybrid_alpha * vector + (1 - hybrid_alpha) * keyword`.
4. **Rerank.** `TermOverlapReranker` blends the fused score with the share of query words the chunk really contains, and adds a small bonus
   when two neighbouring query words also appear together. A chunk containing none of the query words has its score cut to a quarter.
5. **Drop weak hits.** Hits scoring below `min_score` are removed, so a query that matches nothing returns nothing.
6. **Diversify.** Maximal marginal relevance picks `top_k` hits, each time preferring the best score minus `(1 - mmr_lambda)` times its
   similarity to hits already chosen, so the results are not near-copies of each other.
7. **Cite and assemble.** Each hit gets a citation (source, character span, a snippet around the matched words). The context text numbers
   the passages `[1]`, `[2]` and stops at `context_tokens`, shortening the last passage rather than dropping it.

The result is cached by query text, `top_k` and filters. Adding or removing a document clears the cache.

## Tree search

`tree_search` does not use vectors. Each document is a tree of sections built from its headings (`tree_max_depth` levels), each with an
extractive summary of `summary_sentences` sentences. A section scores `2 * title match + 1 * summary match + 0.5 * match in its first 300
characters`. The search starts at each root and, at each level, follows the `tree_beam` children whose subtree matches best.
Use it for questions about where something is documented; use `search` for questions about what the documents say.

## Filters

A filter is a dictionary over chunk metadata: `{"source": "a.md"}` is equality; `{"year": {"gte": 2020}}` uses an operator (`eq`, `ne`, `gt`,
`gte`, `lt`, `lte`, `in`, `contains`, `exists`); `{"and": [...]}`, `{"or": [...]}` and `{"not": {...}}` combine filters. Chunks carry `source`,
`format`, `title`, `language` (code), `heading` and `heading_level` (markdown) and `symbol` (code).

## Measuring quality

`localrag eval` runs a golden set: queries, each paired with the sources that should answer it. Without `--golden`, queries are generated
from the index itself: for a sample of chunks, the query is the three or four rarest words of the chunk. The report gives recall@k,
precision@k, MRR, MAP and nDCG@k and lists queries that found nothing relevant. A synthetic set measures whether the index can find text
it contains; for a fair measure of your own questions, write a golden set with `--save-golden` as a starting point and edit it.
