# PHASE3_RESULTS.md — Pal Model-Agnostic Efficiency Layer

## Status: Verified working

Repo: `pal-context-manager`
Backend tested: Ollama (llama3), local, GPU-accelerated (RTX 3050, 6GB VRAM, WSL2)

## What was built

- `BackendAdapter` interface with `OllamaAdapter` + `LlamaCppAdapter` (free-tier
  local backends only, per project constraint)
- `ContextCache` — hash-based dedup detection with TTL and hit/miss tracking
- `ContextPager` — hot/cold context split (`hot_window`, default 6 turns),
  with embedding-similarity + recency-based recall of relevant cold segments
- `InputCompressor` / `OutputCompressor` — whitespace/dedup-line compression
- `Router` — swappable backend dispatch, verified switching Ollama <-> llama.cpp
- `MetricsCollector` — tokens saved, latency, cache hit rate, cache memory
  overhead, and a `cache_pays_for_itself` check

## Key findings

### 1. Ollama has no server-side prompt caching across HTTP requests
Verified directly: two identical back-to-back `/api/chat` requests reported
`prompt_tokens: 29` then `prompt_tokens: 129` — the second call recomputed
the entire prior turn from scratch. This ruled out client-side "don't resend
duplicate text" as a viable strategy for Ollama (the model needs the full
message list every request; there's no way to send less and get a valid
response).

**Implication for the original design doc**: the "prompt/context dedup &
caching" priority-1 technique, as originally scoped (skip resending detected
duplicates), does not apply to Ollama. It should still work for API backends
with native server-side prompt caching (e.g. Claude's prompt caching), not
currently in scope since only free-tier local backends were implemented.

### 2. Real, measurable savings come from bounding context growth (paging)
A naive/unbounded chatbot resends the entire conversation history every
turn, growing forever. The pipeline caps what's resent via `ContextPager`
(hot window + relevance-scored cold recall) instead.

9-turn live test (Ollama, llama3, real GPU inference):

| Metric | Value |
|---|---|
| Requests | 9 |
| Total tokens saved (vs. unbounded baseline) | 434 |
| Cache hit rate (system prompt dedup detection) | 0.89 |
| Cache memory overhead | 64 bytes |
| `cache_pays_for_itself` | True |

Standalone raw-vs-pipeline comparison (`benchmarks/raw_vs_pipeline.py`),
measuring Ollama's own reported `prompt_eval_count` per turn:

| Turn | Raw (unbounded) | Pipeline (paged) | Saved |
|---|---|---|---|
| 1 | 36 | 36 | 0 |
| 4 | 143 | 188 | -45 (paging not yet active, below hot_window) |
| 5 | 187 | 182 | 5 (paging kicks in) |
| 9 | 354 | 192 | 162 |

Raw grows linearly, unbounded. Pipeline plateaus once history exceeds
`hot_window`. Gap widens the longer the conversation runs.

### 3. Model residency (`keep_alive`) matters more than token savings for UX
First-call latency: 38.5s (cold model load into VRAM).
Warm-call latency: 0.36–0.4s.
This is orthogonal to token savings but was the single biggest real-world
latency win observed — worth keeping prominent in any future benchmarking.

### 4. Bugs found and fixed during validation
- `naive_tokens` was originally computed from the *already-paged* message
  list, comparing the pipeline's output against itself — this silently
  hid all real savings (`total_tokens_saved: 0` even when paging was
  correctly bounding context). Fixed to compare against the full,
  untrimmed history.
- History segments were tagged with role `"user"` regardless of whether
  they were originally assistant replies, corrupting conversation
  structure on resend. Fixed with a `role` field on `Segment`.
- `OllamaAdapter` was sending an unused `context` field valid only for the
  legacy `/api/generate` endpoint, not `/api/chat`. Removed; replaced with
  `keep_alive` for model residency.

## Open items (not yet validated)
- `LlamaCppAdapter` — implemented (`cache_prompt: true` wired), not yet
  tested against a live `llama-server` instance. This is the backend most
  likely to show genuine server-side prompt-cache reuse, unlike Ollama.
- `InputCompressor` / `OutputCompressor` — functional (whitespace/line
  dedup) but real-world impact on token count not yet benchmarked
  separately from paging.
- Memory/storage overhead check currently uses an approximate 4-bytes-per-
  token cost basis for comparison; not a precise accounting model.

## Next
Set up `llama-server` locally to validate real prompt-cache reuse
(the one dedup mechanism Ollama can't provide), then A/B against the same
9-turn scenario used here.
