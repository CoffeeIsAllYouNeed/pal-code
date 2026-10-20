# Tools

**What counts as a tool.** The Tool Manager holds **28 tools in the chat**: 8 core tools, 10 service tools, 7 for your documents and code
project (offered only when something is attached), 2 for the code graph and `file_to_markdown`. With the SDK's `word_count` and the 5 optional
OS tools there are **34 distinct tools**. The service tools are **off by default**: your query leaves this machine, most need a key, and
some cost money. Switch them on in the settings:

```yaml
tools:
  network: true                        # wikipedia and arxiv_search
  services: [google_search, currency_convert, moon_phase]     # or [all]
```

`pal tools` prints every tool by name with its live state, and `pal doctor` shows which service tools are on and have their key.

## All tools by name

This list is generated from `pal/tool_catalog.py`, and a test checks it against the tools that are really registered.

**Files and directories**

| Tool | Name | What it does | Who can use it |
|---|---|---|---|
| File Read Tool | `read_file` | Read a text file in the agent's workspace. | SDK agents, opt-in (register_os_tools) |
| File Writer Tool | `write_file` | Write text to a workspace file (an overwrite asks first). | SDK agents, opt-in (register_os_tools) |
| Directory Read Tool | `list_files` | List the files in the workspace. | SDK agents, opt-in (register_os_tools) |
| File Save Tool | `save_file` | Save a new file to the workspace for download. | chat, agents that list it |
| File to Markdown Converter Tool | `file_to_markdown` | Convert a PDF, Word, PowerPoint, Excel, EPUB, HTML or LaTeX file to markdown and save it. | chat, when a convertible file is in the workspace; SDK agents (opt-in) |

**Your documents (PDF, Word, text ...)**

| Tool | Name | What it does | Who can use it |
|---|---|---|---|
| Document Search Tool | `search_documents` | Find the parts of your attached documents that match a query. | chat, offered when a document is attached |
| Document Reader Tool | `read_document` | Read one part of an attached document. | chat, offered when a document is attached |
| Document Outline Tool | `document_outline` | List a document's parts with headings and sizes. | chat, offered when a document is attached |

**Your code project (graphify)**

| Tool | Name | What it does | Who can use it |
|---|---|---|---|
| Code Graph Builder Tool | `build_code_graph` | Index a source zip, or a folder of code, as a graph (graphify). | chat, when a zip or code is in the workspace |
| Code Graph Query Tool | `query_code_graph` | Ask the code graph: matching nodes and source excerpts. | chat, offered when a project is loaded |
| Source Reader Tool | `read_source` | Read lines of a source file in the project. | chat, offered when a project is loaded |
| Usage Finder Tool | `find_usages` | Where a function or class is defined and used. | chat, offered when a project is loaded |
| Source Search Tool | `search_source` | Find lines containing some text. | chat, offered when a project is loaded |
| Project File List Tool | `list_project_files` | List the project's source files. | chat, offered when a project is loaded |

**Calculation and text**

| Tool | Name | What it does | Who can use it |
|---|---|---|---|
| Calculator Tool | `calculator` | Exact arithmetic. | chat, SDK |
| Clock Tool | `current_time` | The current date and time. | chat, SDK |
| Word Count Tool | `word_count` | Count the words in a text. | SDK |
| Token Counter Tool | `count_tokens` | Estimate the tokens in a text (characters divided by 4). | chat |
| Cost Estimator Tool | `estimate_cost` | Dollars for some tokens at the model's price (LiteLLM's table). | chat |
| Text Summarizer Tool | `summarize_text` | A short summary of a text, by the local model. | chat |
| Wolfram|Alpha Tool | `wolfram_alpha` | Computed answers: maths, units, science, facts. | chat, off until named in tools.services |
| Currency Converter Tool | `currency_convert` | Convert money at central bank reference rates. | chat, off until named in tools.services |
| Moon Phase Tool | `moon_phase` | The moon's phase for a date, computed locally. | chat, off until named in tools.services |
| Words API Tool | `words_api` | Definitions, synonyms, antonyms, rhymes, examples. | chat, off until named in tools.services |

**Search and lookup**

| Tool | Name | What it does | Who can use it |
|---|---|---|---|
| Google Search Tool | `google_search` | Google results for a query (through SerpApi). | chat, off until named in tools.services |
| Wikipedia Tool | `wikipedia` | The summary of a Wikipedia article. | chat, off until tools.network is on |
| arXiv Search Tool | `arxiv_search` | Research papers on arXiv. | chat, off until tools.network is on |
| HTTP GET Tool | `http_get` | Fetch a URL through the guarded network interface. | SDK agents, opt-in (register_os_tools) |

**Images and audio**

| Tool | Name | What it does | Who can use it |
|---|---|---|---|
| Image Caption Tool | `image_caption` | Describe an uploaded image. | chat, off until named in tools.services, needs an uploaded image |
| Visual Question Answering Tool | `visual_question_answering` | Answer a question about an uploaded image. | chat, off until named in tools.services, needs an uploaded image |
| Text-to-Image Tool | `text_to_image` | Make an image from a description. | chat, off until named in tools.services |
| Text-to-Audio Tool | `text_to_audio` | Turn text into spoken audio. | chat, off until named in tools.services |
| Speech-to-Text Tool | `speech_to_text` | Transcribe an uploaded audio file. | chat, off until named in tools.services, needs an uploaded audio file |

**System**

| Tool | Name | What it does | Who can use it |
|---|---|---|---|
| System Info Tool | `system_info` | Operating system, memory and load. | SDK agents, opt-in (register_os_tools) |

## The AIOS paper's 17 tools: what became of each

The AIOS paper (Table 5) lists 17 tools. Pal keeps 12 and has removed 5 (Bing search, Google Places, IMDb rank, Shazam and
TripAdvisor).

| AIOS tool | Pal tool | Backed by | Key | Checked against the service's documentation? |
|---|---|---|---|---|
| Arxiv | `arxiv_search` | arXiv API | none (`tools.network`) | tested against fakes only |
| CurrencyConverter | `currency_convert` | **Frankfurter** reference rates (the paper used a RapidAPI listing) | none | yes (v2 `/rate/{base}/{quote}`) |
| GoogleSearch | `google_search` | **SerpApi, Google engine.** Google's Custom Search JSON API is closed to new customers and ends 1 Jan 2027 | `SERPAPI_API_KEY` | yes (`organic_results` with title, link, snippet). The paper lists this tool's output as an image; this one returns text results |
| ImageCaption | `image_caption` | Hugging Face, hf-inference route | `HF_TOKEN` | router address and bearer header yes; the request and answer shape are from Hugging Face's task format, not checked here |
| MoonPhaseSearch | `moon_phase` | **computed here**, no service | none | n/a. Tested against 7 real new and full moons: worst error 0.54 day |
| TextToAudio | `text_to_audio` | Hugging Face, hf-inference route | `HF_TOKEN` | as ImageCaption |
| TextToImage | `text_to_image` | Hugging Face, hf-inference route | `HF_TOKEN` | yes for the request body (`inputs`); the model's availability can change |
| VisualQuestionAnswering | `visual_question_answering` | Hugging Face, hf-inference route | `HF_TOKEN` | as ImageCaption |
| VoiceActivityRecognition | `speech_to_text` | Hugging Face, hf-inference route (speech recognition) | `HF_TOKEN` | as ImageCaption |
| Wikipedia | `wikipedia` | Wikipedia REST API | none (`tools.network`) | tested against fakes only |
| WolframAlpha | `wolfram_alpha` | Wolfram\|Alpha LLM API | `WOLFRAM_APP_ID` | yes (address, bearer header, `input`, `maxchars`) |
| WordsAPI | `words_api` | WordsAPI on RapidAPI (`wordsapiv1.p.rapidapi.com`) | `RAPIDAPI_KEY` | host, path and headers yes; the answer fields are from its documentation |

**Removed.** BingSearch (Bing's own Search API was shut down on 11 Aug 2025), GooglePlace, ImdbRank, Shazam and TripAdvisor.

**What "checked" means here.** The documentation of the service was read. It does not mean a real request was made: **no real service was
ever called from this project's tests.** Every request is built exactly as documented and sent in tests to a fake kernel or to a real
HTTP server on this machine (see `tests/chat/test_services.py` and `test_services_real_chain.py`). The first real call with your own key
is the first real test. The Hugging Face models are set in `tools.hf_models`, because which models the hf-inference route serves changes
over time.

## Images and audio

An uploaded image or audio file (png, jpg, webp, gif, wav, mp3, flac, ogg, m4a) is stored in the session's workspace under
`uploads/`; the model is told only its path and never sees the bytes. It is accepted only when a media tool that can use it is switched
on. Pictures and audio the tools make are saved under `generated/` and offered as download links under the reply. The size limit is
`os_kernel.max_binary_bytes` (20 MB).

## What keeps this safe

- **Off by default,** and a tool also needs its key in the environment before it is offered. Which tools are offered also depends on the
  message (a search tool for a search-like message) so the prompt does not grow by ten tool descriptions.
- **The model never holds a key.** Keys are read from the environment, sent in a header where the service allows it, and never put in
  a tool argument, a result, an error message or a log. Where a service insists on a key in the address (SerpApi), the OS kernel
  interface logs the address without its query string, urllib3's own debug log (which prints the whole address) is silenced during a
  request, and error text is scrubbed of anything shaped like a key. A test checks all of this with a debug-level log capture.
- **One guarded door.** Every request goes through the OS kernel interface: http and https only, no private addresses, a size cap, an
  allow-list of headers (a request cannot set `Host` or `Cookie`), a method limit (GET and POST), and a redirect is followed without
  the caller's headers, so a key is never sent to another address.
- **Fixed addresses.** Each tool builds its address from constants and checked input (currency codes, words, model names); there is
  still no "fetch any URL" tool, because a document could instruct the model to send your conversation to an address of its choosing.
- **Agents do not get these.** The coding and documentation agents list their tools explicitly, and none of these is on either list.
- **What leaves the machine** is shown for each tool in `ServiceSpec.sends`, for example "your query goes to SerpApi" or "the image goes
  to Hugging Face". The moon-phase tool sends nothing.
