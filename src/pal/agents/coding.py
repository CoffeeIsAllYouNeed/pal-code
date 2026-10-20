from pal.sdk.agent import AgentSpec, NativeAgent


class CodingAgent(NativeAgent):
    """Explains, reviews and writes code from an attached source zip (code graph + read-only source tool)."""

    spec = AgentSpec(
        id="coding", name="Coding agent",
        description="Explains, reviews and writes code. Attach a source zip and ask about it.",
        role="Role: coding agent. Answer only from the supplied code, graph excerpts and files; say so if something "
             "is not there. Give minimal, correct code in fenced blocks and name the file. Explain only what was asked.",
        key_env="GEMINI_API_KEY_CODING",
        tools=frozenset({"read_source", "search_documents", "read_document", "find_usages", "search_source", "list_project_files",
                         "query_code_graph", "build_code_graph", "file_to_markdown"}))
