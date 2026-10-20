from pal.sdk.agent import AgentSpec, NativeAgent


class DocumentationAgent(NativeAgent):
    """Writes README, API reference and docstrings strictly from the code or text it is given."""

    spec = AgentSpec(
        id="docs", name="Documentation agent",
        description="Writes README, API reference and docstrings from code or text you attach.",
        role="Role: documentation agent. Write accurate, concise markdown documentation (README, API reference, "
             "docstrings, changelog) strictly from the supplied code or text. Never invent behaviour: mark unknowns "
             "as TODO. Use headings and short sentences.",
        key_env="GEMINI_API_KEY_DOCS", max_tokens=1536,
        tools=frozenset({"read_source", "search_documents", "read_document", "document_outline", "list_project_files", "search_source",
                         "save_file", "query_code_graph", "build_code_graph", "file_to_markdown"}))
