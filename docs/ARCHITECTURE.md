# Architecture

Pal is one Python package (`pal-code`, imported as `pal`) with two interfaces over one token-saving engine.

```
 pal cli  ─┐                                  ┌─ Ollama (local models, on a computer with a GPU)
           ├─ folder setup ─ session ─ engine ─┤
 pal web  ─┘   (map, graph,   (agents,  (Chat  └─ API models through LiteLLM (OpenAI, Anthropic, Gemini, OpenRouter ...)
               documents,      workflow) Service)
               local model)
```

## 1. Start-up: `pal cli` / `pal web`

`src/pal/terminal/cli.py` → `prepare_folder()`:

| Step | Code | What it does |
|---|---|---|
| Hardware | `hardware.py` | GPU and its memory (NVIDIA, AMD, Apple Silicon, Windows registry). No GPU means API models only. |
| Local model | `setup.py: ensure_local()` | Installs Ollama (asking first), starts `ollama serve`, downloads the models `local_models.recommend()` picks for the GPU (asking first). |
| Folder | `project.py: Project.prepare()` | Scans the folder (no dependencies, hidden folders, secrets, `.palignore`), converts documents to markdown, writes the project map, builds the code graph in a background thread. Kept in `~/.pal/projects/<id>/`; only changes are redone. |
| Session | `session.py: make_session()` | The engine client with the folder's graph and documents (`ProjectGraph`, `ChatService.preload_docs`) and the one-off project brief. |

## 2. The terminal session

| Module | Role |
|---|---|
| `ui.py` | prompt_toolkit screens (landing, input box, menus) and rich output, in Pal's black/red/white with the logo. |
| `session.py` | `PalSession`: one request. Builds the agent's message (Ask, Plan, Build), hides secrets, sends only changed files, parses proposed files, runs the security review and the **syntax check** (with up to two fix rounds). |
| `workflow.py` | Plans per folder (`Store`, with a **lock** per plan), branches, context, the sandbox of pending changes, versions, `apply()` with checks, rollback and auto-debug, git commits. |
| `flow_cli.py` | The workflow commands (`/load`, `/continue`, `/diff`, `/apply`, `/debug`, `/rewind` ...). |
| `syntax.py` | Parses a proposed file: Python, JSON, YAML, TOML, and tree-sitter grammars for 20 languages. Never runs code. |
| `security.py` | Secret detection and redaction, the code review rules, terminal-escape stripping, the folder audit. |
| `review.py` | `/review`: the pending changes as a static page in the browser (escaped, no scripts). |
| `bench.py` | `pal bench`: the same task as a plain chat and through Pal, in temporary copies, with the model's own token counts. |
| `local_models.py` | The catalogue of local models by the GPU memory they need, and the pick for this computer. |

## 3. The engine (`ChatService`, `src/pal/chat/service.py`)

One turn, in order (details in [ENGINE.md](ENGINE.md)):

1. **Direct answers** without a model (arithmetic, time).
2. **Routing**: easy turns to the small local model, the rest to the main model (off without a GPU, and for project work).
3. **Input saver**: the local helper trims the message; deterministic guards and a verifier undo any loss (off without a GPU).
4. **Context**, under one token budget (`context_manager/`): recent turns, older turns only when they match (paging), compressed old
   replies, memory notes, **document passages** that match the request (`documents/`), and **code-graph excerpts** (`knowledge_graph/`).
   Passages are looked up with the person's own words, not the agent instructions around them.
5. **The model call** through the LLM scheduler (`llm_scheduler/`: one line per API key; the local model is preemptive), with tools when
   allowed, and terse-output rules (`agent_constraints/`).
6. **Metrics**: tokens, savings by stage, latency, memory (`metrics/`, `token_tracker/`, `latency_tracker/`, `memory_tracker/`).

## 4. The web interface (`pal web`)

Flask (`web/app.py`) on 127.0.0.1, served by waitress. The chat page uses the same `ChatService`, with the folder's graph and documents.
The Studio (`studio/`) plans and builds small apps and has an editor with "Ask Pal". Strict content security policy, no external scripts.

## 5. Data

| Where | What |
|---|---|
| `~/.pal/pal.json` | provider, model, agent, helper model, setup done. Never a key. |
| `~/.pal/projects/<folder id>/` | plans (`*.plan.json`, `*.lock`), `project.json`, `project-map.md`, `markdown/`, `graph/` |
| `~/.pal/pal-audit.log` | one line per file Pal wrote |
| `~/.pal/bench/` | `pal bench` results |
| `~/.pal/ollama.log` | the log of an `ollama serve` Pal started |

`PAL_HOME` moves all of it.

## 6. Tests and checks

`tests/` has one folder per component, plus `tests/pal/` for the terminal (fake models, real files, real git). CI runs ruff, mypy and pytest
on Linux, macOS and Windows with Python 3.10 to 3.13, then builds the package and installs it in a clean environment. Publishing to PyPI
happens when a `v*` tag is pushed (`.github/workflows/publish.yml`, trusted publishing).
