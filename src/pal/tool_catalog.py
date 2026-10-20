"""Every tool in Pal, by its human name: the single list the docs and `pal tools` are built from.

A test compares this catalogue with what is really registered (the chat's tool box, the SDK, the optional OS tools), so it cannot
drift: a tool that is added without an entry here, or listed here without existing, fails the tests.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolEntry:
    name: str          # what the model and the Tool Manager call it
    title: str         # its human name, like "File Read Tool"
    group: str
    summary: str
    where: str         # who can use it, and what switches it on


FILES = "Files and directories"
DOCS = "Your documents (PDF, Word, text ...)"
CODE = "Your code project (graphify)"
CALC = "Calculation and text"
LOOKUP = "Search and lookup"
MEDIA = "Images and audio"
SYSTEM = "System"

_SVC = "chat, off until named in tools.services"
_NET = "chat, off until tools.network is on"
_DOC = "chat, offered when a document is attached"
_PRJ = "chat, offered when a project is loaded"
_OS = "SDK agents, opt-in (register_os_tools)"

CATALOG: tuple[ToolEntry, ...] = (
    ToolEntry("read_file", "File Read Tool", FILES, "Read a text file in the agent's workspace.", _OS),
    ToolEntry("write_file", "File Writer Tool", FILES, "Write text to a workspace file (an overwrite asks first).", _OS),
    ToolEntry("list_files", "Directory Read Tool", FILES, "List the files in the workspace.", _OS),
    ToolEntry("save_file", "File Save Tool", FILES, "Save a new file to the workspace for download.", "chat, agents that list it"),
    ToolEntry("file_to_markdown", "File to Markdown Converter Tool", FILES,
              "Convert a PDF, Word, PowerPoint, Excel, EPUB, HTML or LaTeX file to markdown and save it.",
              "chat, when a convertible file is in the workspace; SDK agents (opt-in)"),
    ToolEntry("search_documents", "Document Search Tool", DOCS, "Find the parts of your attached documents that match a query.", _DOC),
    ToolEntry("read_document", "Document Reader Tool", DOCS, "Read one part of an attached document.", _DOC),
    ToolEntry("document_outline", "Document Outline Tool", DOCS, "List a document's parts with headings and sizes.", _DOC),
    ToolEntry("build_code_graph", "Code Graph Builder Tool", CODE, "Index a source zip, or a folder of code, as a graph (graphify).",
              "chat, when a zip or code is in the workspace"),
    ToolEntry("query_code_graph", "Code Graph Query Tool", CODE, "Ask the code graph: matching nodes and source excerpts.", _PRJ),
    ToolEntry("read_source", "Source Reader Tool", CODE, "Read lines of a source file in the project.", _PRJ),
    ToolEntry("find_usages", "Usage Finder Tool", CODE, "Where a function or class is defined and used.", _PRJ),
    ToolEntry("search_source", "Source Search Tool", CODE, "Find lines containing some text.", _PRJ),
    ToolEntry("list_project_files", "Project File List Tool", CODE, "List the project's source files.", _PRJ),
    ToolEntry("calculator", "Calculator Tool", CALC, "Exact arithmetic.", "chat, SDK"),
    ToolEntry("current_time", "Clock Tool", CALC, "The current date and time.", "chat, SDK"),
    ToolEntry("word_count", "Word Count Tool", CALC, "Count the words in a text.", "SDK"),
    ToolEntry("count_tokens", "Token Counter Tool", CALC, "Estimate the tokens in a text (characters divided by 4).", "chat"),
    ToolEntry("estimate_cost", "Cost Estimator Tool", CALC, "Dollars for some tokens at the model's price (LiteLLM's table).", "chat"),
    ToolEntry("summarize_text", "Text Summarizer Tool", CALC, "A short summary of a text, by the local model.", "chat"),
    ToolEntry("wolfram_alpha", "Wolfram|Alpha Tool", CALC, "Computed answers: maths, units, science, facts.", _SVC),
    ToolEntry("currency_convert", "Currency Converter Tool", CALC, "Convert money at central bank reference rates.", _SVC),
    ToolEntry("moon_phase", "Moon Phase Tool", CALC, "The moon's phase for a date, computed locally.", _SVC),
    ToolEntry("words_api", "Words API Tool", CALC, "Definitions, synonyms, antonyms, rhymes, examples.", _SVC),
    ToolEntry("google_search", "Google Search Tool", LOOKUP, "Google results for a query (through SerpApi).", _SVC),
    ToolEntry("wikipedia", "Wikipedia Tool", LOOKUP, "The summary of a Wikipedia article.", _NET),
    ToolEntry("arxiv_search", "arXiv Search Tool", LOOKUP, "Research papers on arXiv.", _NET),
    ToolEntry("http_get", "HTTP GET Tool", LOOKUP, "Fetch a URL through the guarded network interface.", _OS),
    ToolEntry("image_caption", "Image Caption Tool", MEDIA, "Describe an uploaded image.", _SVC + ", needs an uploaded image"),
    ToolEntry("visual_question_answering", "Visual Question Answering Tool", MEDIA, "Answer a question about an uploaded image.",
              _SVC + ", needs an uploaded image"),
    ToolEntry("text_to_image", "Text-to-Image Tool", MEDIA, "Make an image from a description.", _SVC),
    ToolEntry("text_to_audio", "Text-to-Audio Tool", MEDIA, "Turn text into spoken audio.", _SVC),
    ToolEntry("speech_to_text", "Speech-to-Text Tool", MEDIA, "Transcribe an uploaded audio file.", _SVC + ", needs an uploaded audio file"),
    ToolEntry("system_info", "System Info Tool", SYSTEM, "Operating system, memory and load.", _OS),
)

GROUPS: tuple[str, ...] = (FILES, DOCS, CODE, CALC, LOOKUP, MEDIA, SYSTEM)
BY_NAME: dict[str, ToolEntry] = {e.name: e for e in CATALOG}

# which environment variable each service tool needs (the services' own table is the source of truth)
def key_needed(name: str) -> str | None:
    from pal.chat.services import BY_NAME as SERVICES
    spec = SERVICES.get(name)
    return spec.key_env if spec else None


def state(entry: ToolEntry, tools_cfg, env: Mapping[str, str]) -> str:
    """For a tool that has to be switched on: whether it is on and ready, from the settings and the environment."""
    from pal.chat.services import BY_NAME as SERVICES
    if entry.name in SERVICES:
        if "all" not in tools_cfg.services and entry.name not in tools_cfg.services:
            return "off"
        need = key_needed(entry.name)
        return f"needs {need}" if need and not env.get(need) else "ready"
    if entry.name in ("wikipedia", "arxiv_search"):
        return "ready" if tools_cfg.network else "off"
    return ""


def render(tools_cfg, env: Mapping[str, str]) -> str:
    lines = [f"Pal tools ({len(CATALOG)})", ""]
    for group in GROUPS:
        lines.append(group)
        for e in (x for x in CATALOG if x.group == group):
            st = state(e, tools_cfg, env)
            lines.append(f"  {e.title:<36} {e.name:<26} {e.summary}  [{e.where}{'; ' + st if st else ''}]")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
