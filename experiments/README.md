# Experiments: how many tokens and how much time does Pal save?

Not part of the Pal package or its architecture: nothing in `src/` imports them. There are two, and both run the same work twice with
the same local models, once as a **plain chat client** and once through **Pal**:

1. **The build experiment (real only):** the local model *writes code*, which is saved to files in the main directory and judged by the
   project's tests. `python run.py experiment build`. See "The build experiment" below.
2. **The session experiment:** a long question-and-answer session over a real project, with a clearly labelled simulation when Ollama is
   not available. `python run.py experiment`.

## The build experiment: a local model writes code, for real

```
python run.py experiment build                       # 4 functions; add --size big (9) or --size max (12)
python run.py experiment build --functions cosine,jaccard --attempts 3
python run.py experiment build --model qwen2.5-coder:7b     # a code-tuned model, if you have one pulled
```

**There is no simulation.** It needs Ollama running with the models pulled (`ollama pull llama3.2:3b`, `ollama pull llama3.2:1b`); without them it
stops with instructions and writes nothing.

**What the model is asked to make.** The real LocalRAG project (`experiments/projects/localrag`, with 100 tests) has some of its functions
replaced by stubs: signature and docstring kept, body removed (`reciprocal_rank`, `recall_at_k`, `precision_at_k`, `cosine`, `jaccard`,
`min_max`, `reciprocal_rank_fusion`, `weighted_fusion`, `average_precision`, `ndcg_at_k`, `centroid`, `mmr_select`). The model writes each
one again, one at a time, from the stub and the rest of the project. The plain client pastes the whole project into the chat and resends it every turn;
Pal attaches it once as a code graph and sends only what each request needs.

**How it is judged.** A function is *correct* only if LocalRAG's own test suite passes with that one function replaced by the model's version
(the other functions are restored to the original), so the project's tests decide, not opinion. Every function was checked to be a real test: removing it
makes the tests fail. If a reply is unusable (no code, a syntax error, a changed signature, disallowed code) or the tests fail, the model is told why and
tries again (`--attempts`, default 2). Both arms get the same treatment.

**Where the files go.** `experiment_output/<date-time>/` in the main directory (change it with `--out-dir`; it is ignored by git):

```
experiment_output/20261003-103000/
  report.md  report.json            tokens, time and correct functions for both arms
  baseline/localrag_project/        the project as the plain client's model left it (open src/localrag/eval/metrics.py)
  baseline/replies/                 every raw reply, one file each
  pal/localrag_project/          the same for Pal
  pal/replies/
experiment_output/LATEST.txt        the path of the newest run
```

Each arm folder also has `test_output.txt`, the full output of the whole-suite run on the model's code. If that run prints no result at all (a crash, nothing
collected) the report says **"did not run"** instead of reporting zeros, and that file shows why.

A function the model could not write still has its stub, so you can see exactly what it did and did not make.

**Pal's tools are off in this experiment** (`--tools` turns them on): writing a function from its stub does not need them, and a small model can waste calls on
them. The report shows the main-model calls actually made, so extra tool rounds would be visible. If Ollama reused a cached start of the plain client's prompt, the
report says so, because that makes the plain client's time much lower than its token count suggests.

**Read the result carefully.** It is one run, and small models differ from run to run. Fewer tokens for fewer correct functions is not a saving, which is
why the report puts correct-function counts beside the tokens and time. The model's code is run to test it, in a temporary copy of the project, with a time
limit and a filter against file, process and network access and against `eval`/`exec`; that reduces accidents but is not a sandbox. A 3B model may fail some of the
harder functions (`ndcg_at_k`, `mmr_select`); a larger code model does better.

## The RAG build experiment: build a RAG with and without Pal, and see the tokens saved

```
python run.py experiment rag                  # 7 files; add --size small (5) or --size max (9)
python run.py suite                          # 30 varied builds; CSV of the token-saving ones + 3 bar charts (--only 1,5 --repeats 3)
python run.py experiment rag --no-spec        # the honest control: no design document attached
```

A real local model (Ollama) builds a small RAG engine, one file at a time (`chunker.py`, `embedder.py`, `store.py`, `prompt.py`, `rag.py`, and for the larger sizes `cli.py`,
`README.md`, `test_rag.py`, `demo.py`), **once through a plain chat client and once through Pal**. The report leads with one line: *without Pal N tokens, with
Pal M tokens, saved K (p%)*, then a step-by-step table showing where the saving comes from. There is no simulation: without Ollama it stops with instructions.

The model is given the design documentation of LocalRAG (about 4,500 tokens) as its reference, the way a developer hands an assistant a design document. The plain client pastes it
into its first message and so resends it at every step; Pal attaches it once and sends only the parts that match each step. Pal runs with its default settings.

Files go to `experiment_output/<date-time>/baseline/rag/` and `.../pal/rag/` (read `rag.py`), with the raw replies, `check_output.txt`, and `report.md`. After writing, each engine is
run once on a test question (`Paris is the capital of France...`) and the report says whether it worked; that is a side result, because a small model may write an engine that does not run.

**Read it honestly.** The saving comes mostly from not resending the design document. With `--no-spec`, only the conversation history is left to save on, and a build this short saves
nothing: measured with a stand-in model that returns files of realistic length, Pal used about 4% to 6% *more* tokens, because it adds a fixed answer-style rule to every
request and the history is too short to pay that back. Pal pays off when there is a lot to resend (a design document, a project, a long conversation). It is one run, and the
engine runs the model's code once, after a filter and in a temporary copy, which is not a sandbox. By default the helper is the same model as the builder, so Ollama never has to
swap two models in and out of memory (`--helper-model` changes that).

## The session experiment

## Run it

```
python run.py experiment
```

That one command sets Pal up if needed, runs the experiment, prints a summary and writes a report to `experiments/reports/`
(`demo-<time>.md`, a `.json` with every number, and a `-turns.csv` with one row per turn). If Pal is already installed,
`python experiments/run_demo.py` does the same.

## What the session is (what to tell someone who asks)

A developer-assistant session over a **real software project**, **LocalRAG** (`experiments/projects/localrag`), written for this
experiment: a local-first retrieval-augmented generation (RAG) engine that is also a Model Context Protocol (MCP) server. It is pure
Python with no dependencies: about 3,400 lines in 81 modules (loaders and chunkers, an embedder, vector and BM25 indexes with fusion,
reranking, a document tree, a JSON-RPC and MCP server, an evaluation harness and a command line), with its own documentation and a test
suite of 100 tests. RAG and MCP are two of the most active areas in open-source AI at the moment.

The session attaches LocalRAG's documentation in turn 1 and its source part-way, then asks what a developer asks an assistant:

- *Which function is documented as "The document text of one section, cut to max_chars"?* (answer: `section_text`)
- *What parameters does `char_ngrams` take?* (`text`, `n`) and *which files call `centroid`?* (`ivf.py`)
- *What is the default value of `rate_limit_per_minute`?* (600), and the same question answered from the documentation instead
- *According to the MCP guide, which error code means "not initialized"?* (-32002)

mixed with easy questions (arithmetic, the time, small facts) and follow-ups, with every message written with greetings and thanks, as
people type. **Every question and its expected answer is generated from the real files** (with Python's `ast` module and the documentation
tables), so each answer can be checked, and a real run scores the models' answers against them.

The plain client resends the documentation, the source and the whole conversation on every turn. Pal indexes the source as a code
graph (graphify), retrieves only the matching excerpts, answers easy questions without the big model, trims what you type and compresses
the history.

| `--size` | turns | LocalRAG source attached | modules |
|---|---:|---:|---:|
| `small` (quick) | 16 | about 8,400 tokens (core, embed, tree) | 20 |
| `big` (default) | 36 | about 25,500 tokens (plus index, retrieve, ingest) | 59 |
| `max` | 48 | about 37,800 tokens (the whole project) | 81 |

The documentation is about 4,500 tokens at every size.

## Real run or simulated run

- **Real run.** With [Ollama](https://ollama.com) running and two models pulled (`ollama pull llama3.2:3b` and `ollama pull llama3.2:1b`)
  it uses them: the 3B model plays the main model (the one you would otherwise pay an API for) and the 1B model is Pal's local
  helper. Seconds are measured on your machine and token counts also come from the server. Each answer is also checked for the expected fact.
- **Simulated run.** If Ollama is not available it runs anyway, labelled **SIMULATED** in the report and the terminal: the prompts
  Pal builds, and so the tokens it sends, are real; the seconds are computed from an assumed speed (`SIM_SPEED` in
  `experiments/demo/models.py`), and the answers are scripted. It shows the mechanism, not your hardware. `--real` refuses to
  simulate; `--simulate` skips Ollama.

Options: `--model`, `--helper-model`, `--ollama-url`, `--num-ctx` (by default as large as the longest prompt needs, so the plain client's
prompt is never cut off), `--max-ctx` (the largest window a real run may use, 65,536 by default; `--size max` needs more), `--max-output`
(the same answer cap for both arms), `--no-code`, `--seed`, `--out`, `--quiet`. The code part needs graphify (`pip install graphifyy`); without
it that part is left out and the report says so. A real run's length depends on your hardware, and the plain client sends a lot of tokens
at `big` and `max`: try `--size small` first.

## How far to trust it

The report says this itself: the workload is a best case, the baseline is deliberately naive (no trimming, retrieval, tools or routing),
tokens are counted from the request text (characters divided by 4) the same way for both arms, Pal's helper tokens and time are
included, and Ollama may reuse a repeated prefix, which would make the plain client's measured time lower than without a cache and so the
saving shown cautious. A typical chat saves less than this session does. LocalRAG was written by an AI assistant during this project; it is
real, tested code, but it is not an existing open-source repository.
