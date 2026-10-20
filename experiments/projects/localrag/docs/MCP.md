# MCP server

`localrag serve-mcp` speaks the Model Context Protocol over stdio: one JSON-RPC 2.0 message per line on standard input, one reply per line
on standard output. Logs go to standard error, never standard output. It implements the subset of MCP that servers need: `initialize`,
`ping`, tools, resources and prompts.

## Handshake

1. The client sends `initialize` with its `protocolVersion`. The server answers with the same version if it speaks it (2025-06-18,
   2025-03-26 and 2024-11-05 are supported) and otherwise with its newest, plus its capabilities and `serverInfo`.
2. The client sends the `notifications/initialized` notification, which gets no reply.
3. Until `initialize` succeeds, every method except `ping` fails with code -32002.

If an `auth_token` is configured, `initialize` must carry the same value as `authToken` in its params, or it fails with code -32001.

## Tools

| Tool | What it does | Arguments |
|---|---|---|
| `search` | Passages that best answer a query, with citations and a context text | `query`, `top_k` (1 to 50, default 5), `filters` |
| `tree_search` | Sections whose titles and summaries match, found without vectors | `query`, `top_n` (1 to 20, default 5) |
| `get_document` | A document's outline and the start of its text | `name` (id or source), `max_chars` (default 4000) |
| `get_chunk` | One chunk by id | `chunk_id` |
| `list_documents` | The indexed documents | none |
| `stats` | Documents, chunks, vocabulary and cache use | none |
| `ingest_text` | Add text to the index | `text`, `source` (default `note.txt`), `metadata` |
| `ingest_path` | Add a file or a folder | `path`, `recursive` (default true) |
| `delete_document` | Remove a document | `name` |

Arguments are checked against each tool's JSON schema before it runs; a violation is a -32602 error that lists every problem. Tools that
change the index (`ingest_text`, `ingest_path`, `delete_document`) are annotated `readOnlyHint: false`. A failure inside a tool, such as
an unknown document, is a normal result with `isError: true` and the message as text.

## Resources and prompts

- `localrag://stats` is the index statistics as JSON. `localrag://doc/<id>` is the full text of each document; the list follows documents
  as they are added and removed. `resources/templates/list` describes the `localrag://doc/{id}` template.
- Prompts: `answer_with_citations` (argument `question`, optional `top_k`) builds a message with the retrieved passages and an instruction
  to cite them; `summarize_document` (argument `name`) builds a message with the outline and the start of the text.

## Error codes

| Code | Meaning |
|---|---|
| -32700 | parse error: the line was not valid JSON |
| -32600 | invalid request: not a JSON-RPC 2.0 message, or too large |
| -32601 | method not found |
| -32602 | invalid params: a bad argument, an unknown tool or a missing prompt argument |
| -32603 | internal error: a bug; the server keeps running |
| -32001 | unauthorized: the auth token is missing or wrong |
| -32002 | not initialized: send `initialize` first |
| -32003 | rate limited: more than `rate_limit_per_minute` requests |
| -32004 | resource not found |

## Limits and safety

- Messages larger than `max_message_bytes` (1,000,000) are refused.
- Requests are limited by a token bucket of `rate_limit_per_minute` (600) tokens that refills continuously.
- `ingest_path` reads only inside the folders passed with `--allow` (by default the current directory); a path that leaves them, including
  through `..`, is refused.
- Batches (JSON arrays) are supported; notifications in a batch produce no reply.
