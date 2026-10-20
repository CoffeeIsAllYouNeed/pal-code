# The engine, in detail

> The token-saving engine under `pal cli` and `pal web`: each component, how a turn flows through it, and the reasons behind each choice.
> For the overview (the two interfaces, the build workflow, the folder setup) start with [ARCHITECTURE.md](ARCHITECTURE.md).

## Repository layout

```
Pal/
├── pyproject.toml            packaging, dependencies (extras), pytest / ruff / mypy config
├── src/pal/            the package (src layout: tests always run against the installed code)
│   ├── cli.py                `pal serve | doctor | config`   (also `python -m pal`)
│   ├── config.py             typed, validated Settings (YAML + PAL__SECTION__KEY env overrides)
│   ├── config/default.yaml   packaged defaults
│   ├── errors.py             PalError hierarchy (each error knows its HTTP status)
│   ├── paths.py              data folder, safe ids/keys, atomic writes (same on Linux, macOS, Windows)
│   ├── tokens.py             token estimate (chars/4) or the provider's counter (opt-in)
│   ├── logging_setup.py
│   ├── web/                  Flask app factory + templates. Thin: no business logic
│   ├── agents/               Agent application layer. Agents only: CodingAgent, DocumentationAgent
│   ├── chat/                 ChatService (one turn end to end), sessions, tool loop, providers
│   ├── task_router/          local LLM answers easy tasks; gates + classifier + ESCALATE
│   ├── input_savers/         local-LLM input reduction with deterministic guards + verifier
│   ├── input_converters/     attachment -> markdown (pandoc, pypdf, python-pptx, openpyxl)
│   ├── knowledge_graph/      Knowledge Graph: source zip -> code graph (graphify); per-question excerpts and their saving
│   ├── documents/            chunker + per-session document index (on the SDK)
│   ├── context_manager/      Context Manager: which turns are shown (TurnPager) + one token budget (assembler); asks the blocks around it
│   ├── compression/          Compression: old assistant replies and facts of old turns (history.py) + the older NLP minifier (optional extra)
│   ├── notes_manager/        Notes Manager: what is kept, recalled, forgotten and tidied, with its own lock and background tidy-up
│   ├── memory/               Memory Manager: MemoryManager (RAM + LRU-K), MemoryPipeline, the note store
│   ├── storage_manager/      agent-scoped files + vector store, queue, purge
│   ├── tool_manager/         registry, validation, conflict resolver, built-in safe tools
│   ├── access_manager/       privilege groups + user-intervention gate
│   ├── sdk/                  AgentSDK: the one interface the chat uses for chat/memory/storage/tools
│   ├── metrics/              per-turn records and the summary; composes the three trackers, export
│   ├── token_tracker/        Token Tracker: counting, savings credited by source, tokens billed, cost
│   ├── latency_tracker/      Latency Tracker: a turn timed in stages; where the time goes
│   ├── memory_tracker/       Memory Tracker: gauges from the components that hold something, plus the process's own memory
│   ├── llm_wrapper/          LLM Wrapper: LiteLLM (your own key), OpenRouter (explore models), the two modes, the LLMAdapter
│   ├── local_llm/            Local LLM: models served on this machine (Ollama, vLLM, LM Studio, llama.cpp) + in-process Hugging Face
│   ├── llm_scheduler/        LLM Scheduler: one line per API key, preemptive priority for the local model, code slots
│   ├── studio/               Studio: the bot's plan (agents, tools, access, model), the build loop, the comparison, and its endpoints
│   ├── orchestrator/         multi-agent runtime: cgroups (Linux only), schedulers, tool device
│   └── agent_constraints/    AGENT_OUTPUT.md (terse-output rules sent to the API model)
├── tests/                    one folder per component; `tests/fakes.py` + `tests/helpers.py` are shared
├── experiments/              A/B and benchmark scripts (not part of the package)
├── docs/
├── .github/workflows/ci.yml  Linux, macOS, Windows x Python 3.10-3.13
├── Dockerfile, docker-compose.yml
└── install.sh, install.ps1   thin wrappers: venv + `pip install` + `pal doctor`
```

Every component follows the same shape: `pipeline.py` (public entry point), `components/` (the pieces),
`utils/` (shared helpers).

## Layers

```
1 Interfaces            web chat workspace | command line | (terminal chat, planned)
2 Service layer         HTTP API, bounded sessions, session configuration. Knows nothing about individual agents
3 Agent application     agents only: native (coding, documentation) and non-native (adapted frameworks)
4 Pal SDK            native APIs | ReAct and Reflexion patterns | framework adapters | light savers
5 Kernels               Pal kernel: the token-saving turn pipeline; the Context Manager with Knowledge Graph and Compression;
                        Notes, Memory, Storage, Tool and Access managers; the LLM Wrapper, Local LLM and LLM Scheduler;
                        the Token, Latency and Memory trackers
                        OS kernel (Ubuntu / Linux): files, network, system facts, reached by OS system calls
```

**Agent application layer = agents only.** `pal/agents/` holds `CodingAgent` and `DocumentationAgent` and nothing
else. The spec type, base class, patterns, adapters and key helpers are SDK code (`pal/sdk/`). The web page, CLI
and HTTP service are interfaces and service layer, not agents.

**Query routing: LLM queries to the Pal kernel, non-LLM queries to the OS kernel.** Every request an agent sends
through the SDK is a typed query, and `AgentSDK.send_request` decides where it goes by query class alone:

| Query | Goes to | Examples |
|---|---|---|
| `LLMQuery` | Pal kernel (model) | reasoning, planning, tool parsing, code generation |
| `MemoryQuery`, `StorageQuery`, `ToolQuery` | Pal kernel (its managers) | remember, recall, read, write, call a tool |
| `OSQuery` | OS kernel (Ubuntu / Linux) | file read, write, list, delete; network GET and POST; system facts; keyboard and mouse (not supported) |

An `OSQuery` never reaches a model, and an `LLMQuery` never touches the filesystem or the network; tests check both
directions with spies. The OS kernel interface (`pal/os_kernel/`) is a guarded doorway into Linux system calls:
each agent has its own workspace folder and a path must stay inside it (no `..`, absolute paths or symlink escapes);
size limits on reads, writes and downloads; http and https only, private, loopback and link-local addresses blocked
(DNS rebinding is not fully covered); overwrite and delete go through the access manager's confirmation gate and
default to no; every call is audited (operation and target, never contents). System facts come from `/proc` and
`/etc/os-release`. The chat's own tools stay read-only: the OS operations are SDK-level and can be offered to an agent as
tools with `register_os_tools(sdk)`.

**Native agent.** Written for Pal: a `NativeAgent` subclass with an `AgentSpec` (id, short role prompt, allowed
read-only tools, key variable, output cap). It reaches the system only through the SDK's pre-defined APIs (`sdk.chat`,
`remember`, `recall`, `write`, `read`, `search_knowledge`, `call_tool`, `request_access`). With a kernel attached, its
default `run()` is a full token-saving turn: `ChatService.send(..., agent="coding")` adds the agent's role to the system
prompt, forces the API route, uses Gemini with the agent's own key (`GEMINI_API_KEY_CODING`, `GEMINI_API_KEY_DOCS`, or a
shared `GEMINI_API_KEY`) and keeps every saver: input trimming, document index, code graph, paged history, output rules.
Free-tier care: capped output, spaced calls (`agents.min_interval_seconds`), plain messages on 429 or quota errors.

**Non-native agent.** Written with another framework, which has its own model client. It is not rewritten. An adapter
replaces the framework's core model-call function (or supplies its model client) so each call goes to the SDK; the
agent's code is unchanged. Adapters: AutoGen 0.2 (patches `OpenAIWrapper.create`), AutoGen 0.4+ (a model client, no
patching), Open Interpreter (patches `llm.completions`), MetaGPT (patches `BaseLLM.acompletion_text`, run in a thread so
the event loop is never blocked). ReAct and Reflexion are reasoning patterns, not libraries, so they ship as native
agents (`ReActAgent`, `ReflexionAgent`). Only text replies are redirected; framework tool calling is not.

**What savers apply to non-native agents.** The framework owns its prompts, so only safe steps run: tidying blank lines
and trailing spaces outside code blocks, an optional exact-repeat cache (off by default, because agent loops sometimes
retry on purpose), per-call stats, and the SDK's routing to any provider or local model. The full pipeline (input
trimming by a local model, retrieval, paging, output rules) applies to native agents and to chat.

## The LLM Scheduler (`pal/llm_scheduler/`)

The LLM Scheduler schedules model requests (it is not the agent scheduler: that one is in the orchestrator). Every model request is scheduled; code work runs beside the lines. The strategy was chosen for this architecture: model
jobs are short, cloud calls cannot be paused, and a free tier has a rate limit.

| What | Policy |
|---|---|
| **API model** (one key) | One line per API key, strictly **first come, first served**, one call at a time. Chat and every agent that uses the key wait in the same line. Calls start at least `scheduler.api_min_interval[provider]` apart (Gemini: 4 s). A rate-limit error (429, quota) pauses the line and retries the same call after a growing delay, so the call ahead of everyone is never skipped. |
| **Local model** | **Preemptive priority**: 1) agent tasks, 2) chat tasks, 3) other tasks. First come, first served inside each class. One job at a time. A job of a better class pauses the running one at its next safe point, and the paused job resumes before later arrivals of its own class. |
| **Code work** | Converting files and building the code graph run in code, `scheduler.code_workers` at a time, and never wait in an LLM line. |

**What the classes mean.** *Agent task*: the local model's preparation for a request sent to an agent (trim the message,
check the trim). *Chat task*: the same preparation for plain chat, plus deciding whether a message is easy and answering
easy ones locally. *Other task*: background work nobody waits for (the Notes Manager merging duplicate notes, warming up the
model). An agent's own work (its answer) always goes to the API model, never the local model.

**What can be paused.** A model call cannot always be paused, so jobs come in three kinds. *Atomic* jobs (route, trim,
verify: short and structured) are never paused; a better job waits for them to finish. *Step* jobs (merge a batch of memory notes,
tidy a note) are paused between pieces, which costs nothing. *Stream* jobs (a plain-text local answer) are paused between
chunks on cores that really stream: the stream is closed so the model stops, the text so far is kept, and the job resumes
by continuing from that text. A job is paused at most `scheduler.local_max_pauses` times, then runs to the end. A
background job that waits longer than `scheduler.other_stale_seconds` is dropped, because nobody depends on it.

**Visible in Stats.** Each turn records how long its jobs waited in the local scheduler and in the API line, and how often
a local job was paused. If pausing costs more than it saves, set `scheduler.local_max_pauses: 0`; `scheduler.enabled: false`
turns scheduling off. The orchestrator's own `SyscallScheduler` (FIFO and Round Robin) is separate and is for multi-agent
runs.

Not built: scheduling for calls made
directly through the SDK (wrap a local model in `ScheduledWrapper` to put it behind the local scheduler).

## The Context Manager and the blocks around it (`context_manager/`, `compression/`, `knowledge_graph/`, `notes_manager/`)

One object decides everything that goes into a request besides the user's message, under **one token budget**
(`context.total_budget_tokens`, default 10000). Most of what it used to do is now a component of its own; it keeps two jobs and asks
the others.

| | Does | Where |
|---|---|---|
| **Context Manager** (keeps) | chooses which turns of the history are shown (the newest verbatim, relevant older ones recalled) and fits everything under the budget, dropping the least useful first | `context_manager/chat_context.py`, `turns.py`, `assembler.py` |
| **Compression** (asked) | shortens the older assistant replies and turns the oldest turns into a block of exact facts | `compression/history.py` (`HistoryCompressor`) |
| **Knowledge Graph** (asked) | the excerpts of an uploaded project that match the question, and the tokens saved against sending the project | `knowledge_graph/` (`KnowledgeGraph.context_block`) |
| **Notes Manager** (asked) | the notes from other sessions that match the question | `notes_manager/` (`NotesManager.recall`) |
| **Document index** (asked) | the parts of attached documents that match | `documents/` |
| **Token Tracker** (told) | is told what each part saved, so no part reaches into the turn's record itself | `token_tracker/` |

The context manager does no background work: tidying notes belongs to the Notes Manager.

**Compression of old assistant replies** (`compression/history.py`). No model, no new dependency. `HistoryCompressor.layer` shows the
history in layers by age and `facts` builds the block for the oldest turns:

| Layer | What the model sees |
|---|---|
| Newest turns (`hot_turns`, 6) | verbatim |
| Older turns that match the question | verbatim (recalled) |
| The next older turns whose reply can be shortened (up to `compress_turns` and `compress_budget_tokens`) | your message verbatim, the assistant's reply compressed |
| Everything older | only its exact details, in one `<facts>` block |

What is compressed: only old assistant replies. A code block keeps its signature lines and the body becomes `[... N lines omitted]`;
an earlier draft that a later reply replaced (same functions, same headings) becomes a one-line pointer to the newer turn; a reply
that saved a file becomes a pointer to the file; prose keeps its first and last sentence and every sentence holding an exact detail
or a decision word. What is never compressed: the system prompt, your messages, the newest turns, and exact details (numbers, names,
file names and paths, URLs, `code` spans, "quoted" strings, error messages). Originals are never changed or deleted, and a
relevant old turn is still recalled in full. A reply that cannot be made shorter is shown as it was, or, if it is short, only as facts.

The facts block lists exact details of the turns too old to show. Details from your own messages come first when space is short,
a fact repeated across turns is listed once, and counters such as "attempt 7" or "step 2 of 5" are not facts. It is added only when
it is at most 30% (`facts_max_ratio`) of the tokens of the turns it stands for, so a short chat gets nothing added.

The assembler fits the total under the budget by dropping the least useful first: notes, then the facts, then the oldest turns (down
to the last two), then document excerpts, then code excerpts.

**What it costs and saves, measured** (`experiments/compression_bench.py`; a scripted session with realistic long replies, not a
real model; input tokens estimated as characters / 4). A README rewritten and code edited over 34 turns, a budget figure given in
turn 1 and asked about at the end in other words:

| | tokens, whole session | last request | still holds the figure from turn 1 |
|---|---|---|---|
| send every earlier turn | 144,489 | 8,022 | yes |
| paging only (before this) | 63,587 | 1,554 | **no** |
| paging + facts only (`compress_turns: 0`) | 66,991 (+5%) | 1,727 | yes |
| paging + compression + facts (default) | 73,294 (+15%) | 2,032 | yes |

So compression is a saving against sending everything (49% fewer tokens here) but a cost against plain paging (+15%): it buys
retention of exact details and continuity. The knobs are `compress_budget_tokens` (lowering it from 800 to 400 cut the overhead from
+22% to +15%) and `compress_turns: 0` for the cheapest retention. Not measured: a real model, a real long session, or answer quality.

Not built: persisting turns to disk (originals live in the session's memory), and detecting decisions without a model.

## The Notes Manager (`pal/notes_manager/`)

Everything about the notes the chat keeps across sessions, in one place, with its own lock (callers never take one).

| Part | Does |
|---|---|
| `components/privacy.py` | what is never kept: anything that looks like a secret (API keys, tokens, passwords, private keys, long hashes) |
| `components/recall.py` | which notes from *other* sessions go into one request, within `memory.recall_notes` and `memory.budget_tokens` |
| `components/tidy.py` | merging duplicate and near-duplicate notes (no model), as a generator a scheduler can pause between batches |
| `pipeline.py` (`NotesManager`) | `remember`, `recall`, `forget_session`, `clear`, `count_notes`, `stats`, `tidy`, and `after_turn`, which hands the tidy to the LLM Scheduler as a background (`OTHER`) job every `memory.tidy_every` turns |

Only messages you typed are kept (never attachments or model replies), at least `memory.min_chars` long. Old notes beyond a cap are
pruned. It sits on the Memory Manager (RAM cache, LRU-K eviction) which sits on the Storage Manager (the disk): the Notes Manager
decides *what a note is*, the Memory Manager *where it lives*, the Storage Manager *how it is written*.

## The trackers (`token_tracker/`, `latency_tracker/`, `memory_tracker/`)

Three small components that measure; `metrics/` keeps the per-turn records and composes their summaries (every key of the summary is
unchanged).

| Tracker | Answers | How |
|---|---|---|
| **Token Tracker** | how many tokens, saved by which part, at what cost | `count` (estimate or the provider's counter), `credit(turn, source, tokens)` for `input_savers`, `context_paging`, `code_graph` and `documents` (an unknown source is an error, so a typo cannot lose a saving), `summarize` |
| **Latency Tracker** | where does a turn's time go | `stage(turn, name)` times a block as `direct`, `routing`, `local_answer`, `input_saver`, `context`, `api` or `tools`; `finish` adds `other` (the total minus the named stages); `summarize` gives each stage's mean, 95th percentile and share of all the time, and the slowest stage |
| **Memory Tracker** | how much memory, and where | components register gauges (notes with the RAM and disk split and how full the cache is, open sessions, indexed projects, the storage folder); `snapshot` calls them and adds the process's own memory (psutil, the Unix `resource` module, or Windows' own API; the Windows reading is untested) |

The web page's Stats panel shows the stage shares and the memory line; `/api/metrics` returns the memory snapshot beside the summary.
The stage numbers are measured with real clocks around each step of a turn; they were added so the next real run can show *where* the
seconds go instead of guessing.

## The Knowledge Graph (`pal/knowledge_graph/`)

An uploaded source zip is indexed once, locally, with graphify (AST only: no LLM call, nothing leaves the machine); each request then carries
only the nodes and excerpts that match the question, within `knowledge_graph.budget_tokens`. `KnowledgeGraph.context_block(session, question,
count)` returns the excerpts as a block, the tokens saved against sending the whole project, and a description for the turn's report; a graph
failure is logged and gives no block, so the chat never fails because the graph did. (The settings section was called `graph_context`; the
old name still loads.)

## Tools beyond the basics

Ten service tools (Google search, Wolfram, currency, words, moon phase and five Hugging Face image and audio tools) are in
`chat/services/`, one small module per family, all off by default. `docs/TOOLS.md` names every tool and says what became of each of the
AIOS paper's 17 (12 kept, 5 removed: Bing search, Google Places, IMDb rank, Shazam, TripAdvisor), and which request shapes were checked
against documentation and which were not.
They reach the network and the workspace only through two OS kernel operations added for them: `net_request` (auth headers from an
allow-list, binary upload from and binary download into the workspace, no key in any log) and `file_read_b64`.


The Tool Manager holds: `calculator`, `current_time`, `count_tokens`, `estimate_cost` (LiteLLM's prices), `summarize_text` (local
model), `save_file`, `wikipedia`, `arxiv_search`, the 10 service tools, and the seven tools for the user's own material. Documents:
`search_documents`, `read_document`, `document_outline`. Code project: `read_source`, `find_usages`, `search_source`,
`list_project_files`. These seven are registered in every session but offered to the model only when a document is attached or a
project is loaded. Three knowledge tools sit beside them: `file_to_markdown` (the input converter: a PDF, Word, PowerPoint, Excel, EPUB,
HTML or LaTeX file in the workspace becomes markdown, saved under `converted/` and added to the session's documents), `build_code_graph`
(graphify: index a source zip, or a folder of code files the agent wrote, so the project tools work on it) and `query_code_graph` (ask the
graph). Uploads are still converted and indexed automatically when they arrive; the originals of documents and source zips are also kept
in the session's workspace so these tools can work on them. That is 28 tools in the chat (8 core, 7 document and project, 3 knowledge,
10 service); with the SDK's `word_count` and the five optional OS tools, 34 distinct. `docs/TOOLS.md` names all of them, and
`pal tools` prints the list. (They were moved out of the Tool Manager in 0.11.0 and put back in 0.12.0; one list per agent, `AgentSpec.tools`,
governs them like every other tool.) `save_file` writes
a NEW file to the session's workspace and is only for an agent whose spec lists it (the documentation agent); an overwrite asks
the access manager and is declined by default; plain chat is refused. Files can be downloaded from `/api/workspace` until the chat ends: **a chat session's workspace (uploaded originals, generated and saved
files) is deleted when the session ends, is reset, expires or the server stops**, and workspaces left behind by a crash are pruned at start.
`wikipedia` and `arxiv_search` reach the network only with `tools.network: true`, only those two hosts, through the OS kernel
interface. There is deliberately no "fetch any URL" tool: a document could instruct the model to send the conversation to an
address of its choosing.

## One chat turn

```
message ──► direct answer?  arithmetic / "what time is it"          no model, ~0 ms
        └─► route          gates (files, code, long text) → API; local LLM classifies the rest
             ├─ local      the local LLM answers (or says ESCALATE)   no API call
             └─ api        reduce input (local LLM + guards + verifier)
                           attach: documents (indexed once) · code-graph excerpts · memory from other sessions
                           page history: last N turns + older turns only if relevant
                           API model  ──► optional read-only tool calls (max 3 rounds) ──► reply
        record: history (reduced message + file stubs only) · memory · metrics
```

Extra context is added to the request, never to the stored history, so it is not resent on later turns.

**SDK.** `ChatService` talks to the managers only through `AgentSDK`: `chat_response` (per-request API core),
`remember`/`recall` (memory manager), `write`/`read`/`add_knowledge`/`search_knowledge` (storage manager),
`call_tool` (tool manager). One SDK per chat session (agent id `chat-<session>`) plus one profile SDK that
owns cross-session memory. Ending a session purges its files and vectors; memory notes stay.

**Concurrency.** Flask/waitress run threads. Each session has a lock, so turns in one session are serialised;
different sessions run in parallel. The session store is bounded (`server.max_sessions`, idle TTL) and cleans up
what it evicts. Shared managers (storage queue, memory) are guarded.

**Tools.** Text protocol (`<tool>{"name": ..., "args": {...}}</tool>` and `<tool_result>`), so every provider
works without native tool-calling. Tools include `calculator` (AST evaluator, no `eval`), `current_time`, `search_documents`,
`read_document`, `read_source`. Tool results are marked as data, not instructions. The tool prompt is added only when a tool could
help (a number in the message, a document or a project in the session), so plain chat pays no tokens for it.

**Context paging.** `TurnPager` keeps the last `hot_turns` turns and pages older ones in when their keyword
overlap (or embedding similarity, if `sentence-transformers` is installed) with the question, blended with recency,
passes `recall_threshold`. The system-prompt cache and KV bridge in `context_manager` are not used on the chat
path: API providers cache prompts server-side, and the local model is only used for short calls.

**Metrics.** One `TurnMetric` per turn: route, provider/model, latencies (total, local overhead, API, tools),
API tokens, and estimated tokens saved by each mechanism. Savings are estimates (chars/4) unless
`token_counter: provider`.

## Portability

- Paths: `pathlib` everywhere, `~/.pal` or `PAL_HOME`; UTF-8 explicitly on every read and write.
- Storage keys and ids are validated for every OS at once (no `..`, drive letters, backslashes, Windows-reserved
  names or characters), and files are written atomically.
- Tool transport: Unix socket (owner-only) where available, loopback TCP on Windows; newline-framed JSON.
- Subprocesses use argument lists, never a shell; graphify runs as `sys.executable -m graphify`.
- Agent resource limits use Linux cgroup v2. Elsewhere the orchestrator falls back to a no-op manager and logs
  that limits are not enforced. The chat does not use the orchestrator.

## The LLM Wrapper and the Local LLM (AIOS section 3.2, Appendix A.2)

Deployment options for an LLM differ: which model, cloud or local device, which hardware, which inference framework. Each
LLM instance is therefore wrapped and driven only through unified inference system calls. The wrapper is a piece of software, not a
hardware allocation: the model runs on whatever hosts it, and no CPU amount is given to it.

**One interface for every model call: LiteLLM (its Python SDK).** Three kinds of wrapper build on it, so the key handling and the safety
rules live in one place:

| Component | Class | For | Backends |
|---|---|---|---|
| **LLM Wrapper** | `LiteLLMWrapper` | your own key for one provider, direct | `openai`, `anthropic`, `gemini` (`google`), `groq`, `deepseek`, `mistral`, `together`, `perplexity`, `azure_openai`, `bedrock`, and `litellm` (any LiteLLM model string) |
| **LLM Wrapper** | `OpenRouterWrapper` | exploring: one OpenRouter key reaches hundreds of models, some free; a model browser | `openrouter` |
| **Local LLM** | `LocalLLM` | models served on this machine | `ollama` (`ollama_chat/<model>`, default `http://localhost:11434`), `vllm` (`hosted_vllm/`), `lmstudio` (`lm_studio/`), `llamacpp` (`openai/`) |
| **Local LLM** | `HuggingFaceLocalLLM` | a model loaded into this process (LiteLLM calls model servers, it cannot do that) | `huggingface_local` |

**Two modes** (`llm_wrapper/modes.py`). *Explore models*: an OpenRouter key; your prompts pass through OpenRouter on the way to the model's lab;
free models are rate limited (about 20 a minute); the list changes and a model can disappear. *Use my own key*: your key for one provider, direct, no
middleman, that provider's terms. A mode is only a choice of which providers to offer (same wrapper underneath); local models are not a mode.
`pal models [--free] [--search WORDS]` browses OpenRouter's public model list (no key needed; it is kept for an hour; if it cannot be read you get
an error, never a stale guess), `pal chat --mode explore|own_key` picks the provider, `pal doctor` says which mode is ready, and the web page has a mode
switch with a model browser for Explore. The OpenRouter wrapper also sends the app's name as OpenRouter's attribution header. Not verified: OpenRouter's real
model list and fees were not reachable while building this; the parser follows its documented format and is tolerant of missing fields.

`LocalLLM` differs from a cloud wrapper where it matters: there is no key to protect (the OpenAI-style servers get a placeholder, so LiteLLM cannot fall back to
a real `OPENAI_API_KEY` from the environment and send it to a local process), a first call may load the model (a 300 s time-out), it costs nothing per token,
and it can report `installed_models()` and `is_running()`.

Because the wrapper holds users' API keys, the dependency is kept narrow: **the SDK only, never the LiteLLM proxy server**
(most of the 2026 CVEs were in the proxy); the version is **pinned** (`litellm==1.102.1`) and **hash-locked**
(`requirements/litellm.lock`, installed first by `run.py` with `--require-hashes`); the key is passed to each call, never put
in the environment or a global; telemetry and callbacks are off; the price table is the copy bundled with the pinned version,
not downloaded at import. Versions 1.82.7 and 1.82.8 were compromised on PyPI in March 2026 and later removed, which is why a
pin and a hash lock are used. `pal doctor` warns when the installed version differs from the pin. Review the pin regularly
(for example with `pip-audit`): do not trust PyPI's vulnerability data alone.

**Stopping a stream really stops the model.** The scheduler's pause relies on closing a stream. LiteLLM's streaming wrapper
has no `close()`, and without closing, the model server keeps generating. `close_stream` closes the generators underneath
the wrapper, which closes the connection. Tested against fake Ollama and OpenAI-style servers with the real pinned library:
the server stopped within a few chunks, and the scheduler's pause and resume worked end to end through a local runtime.
Not tested: a real Ollama, vLLM or LM Studio.

`LLMAdapter` (`pal/llm_wrapper/adapter.py`) is the AIOS Appendix A.2 wrapper over several instances, with its methods:
`setup_api_keys`, `initialize_llms`, `initialize_single_llm`, `execute_llm_syscall`, `get_model_response`, `process_response`,
`handle_completion_error`.

```python
from pal.llm_wrapper import LLMAdapter, RouterStrategy

llm = LLMAdapter([{"name": "llama3", "backend": "ollama"},
                  {"name": "gemini-3.5-flash", "backend": "gemini"}],
                 strategy=RouterStrategy.Sequential, use_context_manager=True)
llm.llm_chat(messages)                                   # text
llm.llm_chat_with_json_output(messages, response_format=schema)
llm.llm_chat_with_tool_call_output(messages, tools)      # function calling
```

- **Unified response.** Every call returns an `LLMAdapterResponse`: `response_message`, `tool_calls` (`[{"name", "parameters"}]`),
  `finished`, `error`, `status_code`, plus tokens, model, backend and latency. A failure is a response, never an exception
  (HTTP errors, timeouts, rate limits, bad JSON and unknown tools map to 4xx/5xx codes). Key-shaped strings are hidden in every
  error message (`pal/redact.py`), also ones for keys we were never told about.
- **Routing.** `RouterStrategy.Sequential` takes the configured instances in turn; a call can name the instances it may use.
- **Interrupt and resume.** With `use_context_manager=True`, a call with a `pid` and `time_limit` stops at the limit and a later
  call with the same `pid` continues (text-based through LiteLLM; logits-based for the in-process Hugging Face model). Plain text only.
- **Structured output and function calling** are native when LiteLLM reports the model supports them, and emulated (instructions
  in the prompt, then validation, one retry for JSON) when it does not. `WrapperCapabilities.supports_tools` and `supports_json` say which.
- **Cost.** Each reply carries its cost from LiteLLM's price table; the chat records dollars spent and an estimate of dollars saved
  (tokens saved times the model's input price). A local model is free: no price, no cost.
- **Fallback.** If the main provider is rate-limited, down or timing out (not for a bad key or request), a second model answers
  (`fallback.provider`, `fallback.model`; for example Groq or a free OpenRouter model). Off by default.
- **Drop-in.** The adapter also behaves as an `LLMWrapper`, so `AgentSDK.standalone(adapter)` runs on it. The chat itself calls
  the wrappers directly.

Not verified against real services: everything above was tested with LiteLLM's offline mock mode, fake servers on this
machine, and a fake `litellm` module. No real provider, and no real Ollama, vLLM or LM Studio, has been called.

## The Memory Manager: RAM/disk swapping with LRU-K eviction (AIOS 3.5)

AIOS 3.5 describes a memory manager for agent interaction history --
distinct from a traditional OS memory manager's physical RAM, but
similar in structure: memory resides in RAM by default, and once usage
approaches a block's capacity (their example: 80%), the manager swaps
items out to disk via an LRU-K (K-Least-Recently-Used) eviction policy,
which prioritizes retaining items that have been accessed at least K
times recently over items that have only been touched once or twice.

`pal/memory/` implements this as three components composed by
`MemoryManager`:

- **`components/lru_k.py`** -- `LRUKTracker` records the last K access
  timestamps per note and computes each note's *backward K-distance*:
  how long ago its K-th most recent access happened. A note touched
  fewer than K times ever gets infinite distance, so it's always
  evicted ahead of anything that has genuinely earned K recent
  accesses -- this is what "prioritizes retaining items accessed at
  least K times recently" actually means algorithmically; plain LRU
  (which only looks at the single most recent touch) can't make that
  distinction.
- **`components/storage_manager.py`** -- `StorageManager` is the disk
  side of the swap: one JSON file per note, keyed by id. It's a
  deliberately narrow stand-in for the full Storage Manager AIOS 3.6
  describes (a general read/write file layer for agents), which this
  project doesn't implement yet -- only the slice the memory manager
  needs: write a note, read one back by id, delete it, list every
  persisted id.
- **`components/manager.py`** -- `MemoryManager` owns the RAM cache
  (`dict[id, MemoryNote]`) plus a `block_limit` and `high_water_mark`
  (default 0.8, matching the spec's 80% example). Every `write()`
  persists to disk immediately -- durability isn't optional here, only
  RAM residency is -- and caches the note in RAM. Once the cache's size
  crosses `high_water_mark` of `block_limit`, `_maybe_evict()` asks the
  `LRUKTracker` for the coldest RAM-resident note and drops it from the
  cache. Nothing is lost: the note was already durably on disk, so the
  next `read()` for that id reloads it from disk and re-admits it to
  RAM -- a swap-in, symmetric with the swap-out.

`MemoryPipeline.retrieve()` scores every persisted note's content
(RAM- or disk-resident -- a non-indexed store has no cheaper way to
rank relevance without inspecting content) but only the notes that end
up in the returned `top_k` are actually `read()` through the manager,
which is what counts as a real access for LRU-K and can trigger a
swap-in. `MemoryPipeline.stats()` exposes `ram_count` / `disk_count` /
`usage_ratio` directly, for observing the eviction behavior rather than
only its effects.

## The Storage Manager: agent-scoped persistence (AIOS 3.6)

AIOS 3.6 describes a storage manager with two callers: the memory
manager, which swaps data to disk once an agent's memory usage exceeds
its allocation (scoped by the agent ID the memory manager passes in);
and the agent itself, which reads/writes its own files through a
storage API in the SDK -- calls that get converted into storage system
calls and queued by the scheduler, then processed by the storage
manager. Both local files and a vector database (chromadb in AIOS's
implementation) back the actual storage.

`pal/storage_manager/` mirrors that shape exactly:

- **`components/base.py`** -- `StorageSyscall`/`StorageResponse`, the
  same "syscall, not a direct backend call" shape `llm_wrapper` uses for
  inference. Every op (`READ`, `WRITE`, `DELETE`, `LIST`,
  `VECTOR_UPSERT`, `VECTOR_QUERY`) is scoped by `agent_id`.
- **`components/local_file_backend.py`** -- `LocalFileBackend`: one
  root subdirectory *per agent*, so different agents' files and
  knowledge bases never collide. Values round-trip as JSON when
  JSON-serializable, raw bytes otherwise -- the same backend covers
  both a swapped-out memory note and an arbitrary file an agent asked
  to persist.
- **`components/vector_store_backend.py`** -- `VectorStoreBackend`:
  chromadb when installed (one collection per agent), falling back to
  a small in-memory store scored via the same embedding/keyword-overlap
  helpers `context_manager`/`memory` already use when chromadb isn't
  available -- functionally equivalent at this project's scale, just
  not disk-persisted across restarts.
- **`components/storage_queue.py`** -- `StorageSyscallQueue`: the
  centralized FIFO queue "put into the storage queue by the scheduler"
  describes. Storage ops are effectively instantaneous local
  file/DB operations, not multi-second generations, so this only needs
  FIFO -- there's no equivalent need for the LLM scheduler's Round
  Robin/context-interrupt machinery here.
- **`pipeline.py`** -- `StorageManagerPipeline`: owns one
  `LocalFileBackend`, one `VectorStoreBackend`, and the queue. Direct
  calls (`read`/`write`/`delete`/`list_keys`/`vector_upsert`/
  `vector_query`) and the queue-driven path (`submit`/`tick`/`drain`,
  routing through `execute(StorageSyscall)`) both hit the same
  backends -- there's exactly one storage manager per process, not a
  separate implementation per caller.
- **`utils/sdk.py`** -- `StorageSDK`: what an agent calls. Builds a
  `StorageSyscall`, submits it to the shared queue, then drains that
  queue until its own request completes -- so the call looks
  synchronous to the agent even though it went through the same
  centralized queue every other storage request does.
- **`utils/memory_adapter.py`** -- `AgentScopedMemoryStorage`: what the
  memory manager calls. Presents the same narrow interface
  `pal.memory.components.storage_manager.StorageManager` exposes
  (`write`/`read`/`delete`/`exists`/`list_ids`/`read_all`, keyed by
  `MemoryNote.id`) but backed by the shared `StorageManagerPipeline`,
  scoped to one `agent_id`. `MemoryManager` doesn't know or care which
  storage implementation sits behind this interface, only that it has
  one -- so it works identically whether given this adapter or the
  standalone local store, which remains the default when no
  `agent_id`/`storage_manager` is supplied to `MemoryPipeline`.

`OrchestratorPipeline` owns one `StorageManagerPipeline`, shared by
every agent/pipeline it manages, and exposes `storage_sdk_for(agent_id)`
for agents plus `submit_storage_syscall()` / `storage_tick()` /
`drain_storage_syscalls()` for direct queue control -- the storage-side
counterpart to the LLM syscall scheduling it already centralizes.

## The Tool Manager: standardized loading + conflict resolution (AIOS 3.7)

AIOS 3.7 gives the tool manager two responsibilities: standardized tool
loading (a uniform interface across every tool, parameter validation
before execution to prevent tool crashes, dynamic loading by name that
initializes the tool's executable and verifies its dependencies), and
resolution of tool-call conflicts for tools with parallel-access limits
(a hashmap tracking real-time in-flight instance counts; when a queued
request would exceed a tool's limit, the system advances to the next
queued request rather than blocking on the conflicting one).

`pal/tool_manager/` implements both halves as separate,
composable pieces:

- **`components/base.py`** -- `ToolSpec` is the standardized interface
  every tool conforms to: a `name`, a `loader` (called at most once,
  the first time the tool is invoked -- this is the "dynamically loads
  the tool instance including executables initialization" part),
  declared `params` (`ToolParam`, with type and required/default),
  an optional `parallel_limit`, and `dependencies` (importable module
  names checked before the loader runs).
- **`components/param_validator.py`** -- `validate_params()` runs
  before a tool's underlying callable is ever invoked: checks every
  required param is present and correctly typed, fills in declared
  defaults for missing optional ones, and rejects unexpected params --
  a malformed call fails with a clear `ToolValidationError` instead of
  an arbitrary exception from inside the tool.
- **`components/tool_registry.py`** -- `ToolRegistry.get_callable()` is
  the dynamic loading path: verifies every declared dependency imports
  cleanly (raising a clear `ToolDependencyError` naming exactly which
  module is missing, not an opaque failure from inside the tool's own
  init code) *before* calling the tool's `loader`, then caches the
  result so later calls to the same tool skip loading entirely.
- **`components/conflict_resolver.py`** -- `ToolConflictResolver` is
  the hashmap: `tool_name -> current in-flight call count`.
  `can_acquire()` checks a tool's count against its `parallel_limit`;
  `acquire()`/`release()` update it around an actual call.
- **`components/tool_call_queue.py`** -- `ToolCallQueue.dispatch_next()`
  is the described skip-ahead behavior exactly: it scans the queue from
  the front for the first request whose tool currently has capacity,
  executes *that* one, and leaves any conflicting requests it skipped
  over in their original queue position (not requeued at the back --
  their place in line doesn't change, only which request runs next
  does) for a later `dispatch_next()` call once capacity frees up.
  Unlike the LLM syscall scheduler's FIFO/Round-Robin (section 3.3,
  strict arrival or time-sliced order), a conflict here does not block
  everything behind it.
- **`pipeline.py`** -- `ToolManagerPipeline.invoke()` is the
  synchronous convenience call an agent or the orchestrator actually
  uses: submits to the queue, then drains dispatch attempts until its
  own request is the conflict-free candidate that gets picked, and
  returns its `ToolCallResult`.

`ToolDevice` (the socket transport agents connect to, see the Tool
device section below) takes an optional `tool_manager=` -- when
supplied, every tool call over the socket routes through
`ToolManagerPipeline.invoke()` instead of a raw handler dict, so a
socket-connected agent gets the same validation and conflict resolution
as a direct in-process caller. `OrchestratorPipeline` owns one
`ToolManagerPipeline`, wired into its `ToolDevice` this way, and
exposes `register_tool()` / `invoke_tool()` directly.

## The Access Manager: privilege groups + user intervention (AIOS 3.8)

AIOS 3.8 gives the access manager two responsibilities: privilege-based
access control for cross-agent data access (each agent assigned to a
privilege group, enforced through a hashmap of agent ID -> group,
validated before any cross-agent request executes), and a user
intervention interface for irreversible operations (deletion,
overwrite, privilege modification) that requires explicit user
confirmation regardless of what the privilege check says.

`pal/access_manager/` implements both:

- **`components/base.py`** -- `OperationType` (`READ`, `WRITE`,
  `DELETE`, `OVERWRITE`, `PRIVILEGE_MODIFY`) and `DESTRUCTIVE_OPS`
  (`DELETE`, `OVERWRITE`, `PRIVILEGE_MODIFY` -- exactly the three the
  spec names). `AccessRequest` always carries both a
  `requesting_agent_id` and a `target_agent_id`; when they're equal,
  there's no cross-agent question to answer.
- **`components/privilege_registry.py`** -- `PrivilegeRegistry` is the
  hashmap itself: `agent_id -> group`. `same_group()` is what
  `AccessController` checks a cross-agent request against.
- **`components/access_control.py`** -- `AccessController.check()`:
  an agent acting on its own resource is always allowed (nothing
  cross-agent to regulate there); a cross-agent request is allowed only
  if both agents share a privilege group, and denied -- with a reason
  naming which group(s), or that one/both agents aren't assigned one at
  all -- otherwise. Every decision also flags `requires_confirmation`
  when the operation is destructive, independent of whether it's
  cross-agent.
- **`components/intervention.py`** -- `UserInterventionGate` wraps a
  pluggable `confirmation_handler` callback. The actual UI a real
  deployment would ask the user through (a CLI prompt, a web dialog,
  ...) is out of scope for this project; what's implemented is the gate
  itself, defaulting to deny-everything so a destructive operation is
  never silently approved just because no real confirmation UI has been
  wired up.
- **`pipeline.py`** -- `AccessManagerPipeline.authorize()` combines
  both: the privilege check runs first and short-circuits (a
  privilege-denied cross-agent request never reaches the confirmation
  gate at all, even with a handler that always approves); if that
  passes and the operation is destructive, the confirmation gate must
  also approve before the request is authorized.

Integration is real, not just parallel code: `StorageSyscall` carries
an optional `requesting_agent_id`, distinct from the `agent_id` whose
resource is targeted (`None` means "the owner acting on its own
data" -- the common case, never gated). When a `StorageManagerPipeline`
is given an `access_manager`, every `execute()` call -- the path
`StorageSDK` and the storage syscall queue use -- is authorized before
running: a `WRITE` to an existing key is treated as `OVERWRITE`
(destructive), a fresh key is not. `StorageSDK.read_from()` /
`write_to()` / `delete_from()` are the actual cross-agent path this
enables -- they set `requesting_agent_id` to the calling agent while
targeting a *different* agent's resource. The memory manager's own
swaps (`AgentScopedMemoryStorage`, via `StorageManagerPipeline`'s
direct `read`/`write`/`delete` calls, not `execute()`) intentionally
bypass this gating entirely -- those aren't the cross-agent case 3.8
regulates.

`OrchestratorPipeline` owns one `AccessManagerPipeline`, wired into its
`StorageManagerPipeline`, and exposes `assign_privilege_group()` (also
reachable via `submit_agent(..., privilege_group=...)`),
`set_confirmation_handler()`, and `check_access()` directly.

## The AIOS SDK: one interface for agent developers (AIOS 3.9)

AIOS 3.9 gives agent developers an SDK with three responsibilities:
tool integration (a broad catalog of tools across platforms and
input/output modalities), an interaction interface with the kernel
(API functions that abstract away system calls so developers focus on
the agent's own workflow, not on constructing syscalls), and framework
adapters (letting agents built with Autogen, Open Interpreter, or
MetaGPT run inside AIOS without modifying the agent's own code, by
locating and redirecting each framework's core LLM-call function).

`pal/sdk/` implements all three:

- **`pipeline.py`** -- `AgentSDK` is the interaction interface: one
  instance scoped to exactly one `agent_id`, exposing `chat()`,
  `remember()`/`recall()` (memory, 3.5), `read()`/`write()`/`delete()`/
  `add_knowledge()`/`search_knowledge()` (storage, 3.6),
  `read_from()`/`write_to()`/`delete_from()`/`request_access()`
  (cross-agent access, 3.8), and `call_tool()`/`list_tools()` (tools,
  3.7). Every method is a thin pass-through to the corresponding
  manager -- the abstraction is precisely that an agent developer never
  constructs an `LLMSyscall`, `StorageSyscall`, or `ToolCallRequest` by
  hand, or even needs to know those types exist. `chat()` still routes
  through the centralized `SyscallScheduler` (3.3) when one was
  supplied at construction, exactly like the top-level `Pipeline` does.
- **`components/tool_catalog.py`** -- `register_builtin_tools()`: a
  small, dependency-free starter set (a sandboxed calculator, current
  time, word count) spanning a couple of different input/output
  shapes, registered into a `ToolManagerPipeline`. Deliberately minimal
  -- AIOS's own SDK ships a much larger catalog sourced from real
  external platforms and APIs, which needs network access and
  credentials this project doesn't assume are available; a real
  deployment registers its own tools alongside or instead of these via
  `ToolManagerPipeline.register_tool()` directly.
- **`components/framework_adapter_base.py`** -- `FrameworkAdapter`, the
  common shape every framework adapter follows: `install()`
  monkeypatches that framework's core LLM-call function to redirect
  through an `AgentSDK.chat()`, `uninstall()` restores it.
- **`components/autogen_adapter.py`** / **`open_interpreter_adapter.py`**
  / **`metagpt_adapter.py`** -- one adapter per framework AIOS 3.9
  names. Each locates that framework's actual completion entry point
  (`OpenAIWrapper.create` for AutoGen, `llm.completions` for Open
  Interpreter, `BaseLLM.acompletion_text` for MetaGPT) and replaces it
  with a function that calls `sdk.chat()` and translates the reply back
  into whatever shape that framework's own code expects to read.
  Because none of these frameworks are dependencies of this project,
  `install()` raises a clear `FrameworkAdapterError` -- naming the
  package to `pip install`, or the specific attribute path that wasn't
  found if the framework's version has moved on from what the adapter
  targets -- rather than either crashing on an unrelated import error
  or silently leaving the framework's real API untouched (which would
  be worse: the agent would keep calling the real API instead of
  pal with no indication anything was wrong).

`OrchestratorPipeline.sdk_for(agent_id, core_config)` is the usual way
to get an `AgentSDK`: it builds a fresh `LLMWrapperPipeline` for that
agent (different agents may run different backends/models) and wires
everything else -- storage, tools, access control, the syscall
scheduler -- to the orchestrator's shared instances.
`OrchestratorPipeline.register_builtin_tools()` is an explicit opt-in
for the starter tool catalog; a fresh orchestrator still starts with no
tools registered, unchanged from before this existed.

## The Studio (`pal/studio/`, `web/templates/studio.html`, `web/static/studio.js`)

A page and a set of endpoints that turn "build me X" into a reviewed plan and a watched build. It is built on the components above; nothing in the kernel depends on it.

| Module | Does |
|---|---|
| `models.py` | reads Ollama's model list and picks the strongest chat model (largest by parameter count, from Ollama's details, else the name, else the file size; embedding models are never chosen) |
| `catalog.py` | the five roles (Designer, Coder, Tester, Docs writer, Reviewer), the tools a build may use, and the permissions, each with a plain description |
| `plan.py` | the plan, and **one normaliser that enforces every rule whoever changed the plan**: unknown agents, tools and permissions are dropped; locked permissions are always off; a risky one (going online) is on only if the person switched it on (`user_set`); a tool that needs a permission is dropped when it is off; the model must exist; steps follow the agents. It also describes what changed between two plans in words |
| `planner.py` | the bot: a local model asked for one JSON object (read tolerantly: fences, prose, trailing commas), one retry with the error fed back, then simple keyword rules. It can only *propose* |
| `policy.py`, `files.py`, `checks.py` | what a build may write (paths stay inside the project folder: no `..`, absolute paths, drive letters, hidden files, reserved Windows names; only text-like file types, never programs or scripts; size and count limits; names compared without regard to case), how a model's reply is read for files (many ways of labelling a block), and static checks (Python syntax, JSON, HTML tags, brace balance, and at the end of the build that every linked file exists) |
| `builder.py` | the loop: each step is a request, each reply is read, saved under the policy, checked, and sent back once for a fix if there are mistakes; every action is an event |
| `clients.py` | the two ways to talk to the model, counted the same way: `PlainClient` (resends the whole conversation, pastes reference files in full) and `PalClient` (the real chat service: input savers, paged and compressed history, document excerpts) |
| `runs.py` | background runs, one at a time, cancel, old runs deleted, **build twice and compare**, zip download |
| `routes.py` | `/studio` and `/api/studio/{catalog,plan,revise,validate,run,run/<id>,...}` |

**Safety.** The bot's answer is untrusted text: it passes through the normaliser before it can become an action. Generated code is **never run** (running it needs a
sandbox that this version does not have): files are read, checked and written. The key for an API model is used for the run and never stored, logged or put in a plan or
an event. The page is served with a content-security policy that forbids inline script, inline style and any other origin, and its script never builds HTML from model
text (`textContent` only; a test fails if `innerHTML` appears).

**The comparison.** With *Build it twice*, the same plan runs through the plain client and then through Pal with the same model, tools off in both so it is fair, and the
report gives main-model tokens, tokens read and written, Pal's helper tokens (an extra cost, shown), time, files and problems left. It is one run of each, the plain client
goes first, and the notes say so; when Pal used more tokens or time the page says that plainly and why (a short build has little to resend).

**Tested** with a scripted model that plays every role (so a whole build runs through the real Pal pipeline), against an Ollama-shaped server through the real LiteLLM, and
by driving the page in jsdom (type, click, toggle, build, switch tabs, an injection attempt). **Not tested:** a real Ollama model's quality, a real browser (nothing was laid out or
rendered; jsdom does not enforce the content-security policy, which is checked separately), or a long build with a real model.

## The orchestrator's syscall scheduler: centralized queueing for agents (AIOS 3.3)

AIOS 3.3 centralizes request management in the scheduler module rather
than letting each processing module manage its own queue, and implements
two classic algorithms over that queue: FIFO (arrival order, simple, can
starve later requests behind a long one) and Round Robin (bounded time
slices, cycling through requests, backed by a context-interrupt
mechanism for LLM inference).

`pal/orchestrator/components/syscall_scheduler.py` is that
centralization point:

- **`SyscallScheduler`** owns one queue (`collections.deque`) for every
  LLM inference request in the system. Any module -- the top-level
  `Pipeline`, an orchestrator `Agent`, anything else -- calls `submit()`
  with an `LLMWrapper` + `LLMSyscall` and gets a request id back; nothing
  submits work anywhere else. `Pipeline` itself has no queue of its own
  -- when constructed with a shared `syscall_scheduler`, its `run()`
  routes every chat call there instead of calling the core directly, so
  multiple `Pipeline`/`Agent` instances sharing one scheduler actually
  interleave through one ordering, not N private ones.
- **`run_fifo_once()`** pops the head of the queue and runs it straight
  through `core.chat()` to completion -- no interruption.
- **`run_round_robin_tick()`** runs the head of the queue for at most
  `rr_quantum` seconds via `core.stream_chat()`. If it finishes in time,
  it's done. If not, the partial output is captured and folded into a
  continuation request that goes to the *back* of the queue, so every
  other queued request gets its own slice before this one is revisited.
- **`ContextInterruptController`** (`components/context_interrupt.py`)
  is what makes that resume behave like a continuation rather than a
  restart, via two approaches depending on what the request's core can
  actually expose:
  - **Text-based** (`ContextSnapshot(mode="text")`) -- for cores with no
    logits access, which is every cloud API and every local server this
    project talks to. The only state available to snapshot is the
    decoded text generated so far; the partial text is appended back
    into the message list as a partial assistant turn (with a short
    "continue exactly where you left off" instruction), so the next
    slice picks up the thread instead of regenerating from scratch.
  - **Logits-based** (`ContextSnapshot(mode="logits")`) -- for cores
    with direct logits access, currently only `HuggingFaceLocalLLM`.
    Generation runs as manual beam-search stepping
    (`beam_search_step`, beam width 1 by default, matching the AIOS
    example), and the snapshot is the actual search-tree state --
    every beam's token sequence and cumulative log-probability score
    (`BeamSearchState`) -- not just its decoded text. Resuming steps
    that exact state forward, with no re-tokenization or risk of
    the model drifting the way a "please continue" text prompt can.
    (This intentionally excludes the model's KV cache/`past_key_values`
    from the snapshot, trading resume-time recompute cost for a
    simple, portable, inspectable snapshot object.)
  `ContextInterruptController.run_slice()` picks between the two
  automatically based on `core.capabilities().supports_logits_snapshot`.
  Either way, this is a request-level approximation -- there's no way to
  literally suspend an in-flight forward pass through any of the
  backend HTTP/local APIs this project talks to -- but for backends
  with real server-side prefix caching (llama.cpp) or automatic prefix
  reuse (Ollama), reprocessing the shared prefix on a text-based resume
  is cheap for the same reason paging benefits from it elsewhere in
  this project.
- Real incremental streaming (`stream_chat()`) -- required for RR to
  have anything meaningful to interrupt mid-generation -- is implemented
  through `LiteLLMWrapper.stream_chat` (LiteLLM's streaming, with the stream closed when the consumer stops early, see the LLM cores
  section). Cores
  without a streaming endpoint fall back to `LLMWrapper`'s default
  `stream_chat()`, which yields the whole response as one chunk: RR
  still works against them, it just can't interrupt partway through a
  single generation for that specific core.

`OrchestratorPipeline` exposes this as `submit_syscall()` /
`syscall_tick()` / `drain_syscalls()`, configurable at construction with
`syscall_strategy=SchedulingStrategy.FIFO` or `.ROUND_ROBIN` and an
`rr_quantum` in seconds.
