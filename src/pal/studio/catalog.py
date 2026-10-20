"""What the bot can choose from: build roles, tools, and access permissions.

The bot proposes; the person decides. Every item here has a plain description so the page can show what was chosen and why.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pal import tool_catalog


@dataclass(frozen=True)
class Role:
    id: str
    name: str
    description: str
    prompt: str                # what this role is told it is, kept short: it is sent with every request


ROLES: dict[str, Role] = {r.id: r for r in (
    Role("designer", "Designer", "Plans the look and layout: pages, styling, user-interface structure.",
         "You are the designer. You decide the layout and write the styles and markup. Keep it clean, readable and responsive."),
    Role("coder", "Coder", "Writes the working code: the logic and the main files.",
         "You are the coder. You write complete, working code. No placeholders, no TODOs, no explanations inside code."),
    Role("tester", "Tester", "Writes tests and checks that the pieces fit together.",
         "You are the tester. You write small, clear tests for what exists and point out what does not fit together."),
    Role("docs", "Docs writer", "Writes the README and usage instructions.",
         "You are the documentation writer. You write a short README: what it is, how to run it, one example."),
    Role("reviewer", "Reviewer", "Reads everything that was written and fixes mistakes.",
         "You are the reviewer. You read all the files so far, find mistakes and mismatches, and give corrected files."),
)}
ROLE_ORDER = tuple(ROLES)


@dataclass(frozen=True)
class Permission:
    id: str
    title: str
    description: str
    default: bool
    risky: bool = False         # only the person can switch it on: the bot may suggest it but never enables it
    locked: bool = False        # cannot be changed in this version
    why_locked: str = ""


PERMISSIONS: tuple[Permission, ...] = (
    Permission("write_project", "Save files in the project folder", "The build writes the files it creates into its own project folder. Off: a dry run that shows the files but saves nothing.", True),
    Permission("modify_own_files", "Change files it already wrote", "Later steps (the reviewer, fixes) may rewrite a file an earlier step created in this build.", True),
    Permission("read_references", "Read the reference files you attach", "The bot may read documents you attach as reference (a brief, a design doc). They stay on this machine unless you use an API model.", True),
    Permission("network_access", "Go online", "Tools that reach the internet (Wikipedia, arXiv, web fetch, search). Off by default: it sends queries off this machine.", False, risky=True),
    Permission("run_code", "Run the code it writes", "Running generated code needs a proper sandbox, which this version does not have. The build checks files for syntax mistakes instead.", False, locked=True,
               why_locked="needs a sandbox (not in this version)"),
    Permission("outside_project", "Touch files outside the project folder", "Never. Every path must stay inside the project folder.", False, locked=True, why_locked="never allowed"),
    Permission("delete_files", "Delete files", "Never. The build only creates and rewrites files.", False, locked=True, why_locked="never allowed"),
)
PERMISSION_BY_ID = {p.id: p for p in PERMISSIONS}
DEFAULT_PERMISSIONS = {p.id: p.default for p in PERMISSIONS}

# Tools the builder's own file handling replaces, or that need a loaded project: not offered as choices.
_NOT_OFFERED = frozenset({"save_file", "read_file", "write_file", "list_files", "file_to_markdown", "build_code_graph", "query_code_graph", "read_source", "find_usages",
                          "search_source", "list_project_files", "system_info", "word_count"})
DOCUMENT_TOOLS = frozenset({"search_documents", "read_document", "document_outline"})
NETWORK_TOOLS = frozenset({"wikipedia", "arxiv_search", "http_get", "google_search", "wolfram_alpha", "currency_convert", "moon_phase", "words_api",
                           "image_caption", "visual_question_answering", "text_to_image", "text_to_audio", "speech_to_text"})
OFFERED_TOOLS = tuple(e.name for e in tool_catalog.CATALOG if e.name not in _NOT_OFFERED)


def tool_needs(name: str) -> str | None:
    """The permission a tool needs: going online, or reading the files you attached."""
    if name in NETWORK_TOOLS:
        return "network_access"
    if name in DOCUMENT_TOOLS:
        return "read_references"
    return None


def catalog(tools_cfg, env: Mapping[str, str]) -> dict:
    """Everything the page can offer, as plain data. `ready` is false for a tool that has to be switched on in the settings first."""
    tools = []
    for name in OFFERED_TOOLS:
        e = tool_catalog.BY_NAME[name]
        st = tool_catalog.state(e, tools_cfg, env)
        tools.append({"name": name, "title": e.title, "group": e.group, "summary": e.summary, "needs": tool_needs(name), "state": st,
                      "ready": st in ("", "ready"), "why_not": "" if st in ("", "ready") else (f"{st}: set it in the settings" if st != "off" else "off in the settings (tools.services / tools.network)")})
    return {
        "roles": [{"id": r.id, "name": r.name, "description": r.description} for r in ROLES.values()],
        "tools": tools,
        "permissions": [{"id": p.id, "title": p.title, "description": p.description, "default": p.default, "risky": p.risky, "locked": p.locked, "why_locked": p.why_locked} for p in PERMISSIONS],
    }
