"""Prompt templates a client can ask the server to fill in: answering a question from the index, and summarising a document."""
from __future__ import annotations

from typing import Any

from localrag.engine import LocalRAG
from localrag.mcp.registry import Prompt, PromptRegistry


def _user(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": {"type": "text", "text": text}}]


def build_prompts(engine: LocalRAG) -> PromptRegistry:
    registry = PromptRegistry()

    def answer(args: dict[str, str]) -> list[dict[str, Any]]:
        result = engine.search(args["question"], int(args.get("top_k", "5")))
        return _user("Answer the question using only the numbered passages below. Cite them as [1], [2]. "
                     "If the passages do not contain the answer, say so.\n\n" + (result.context or "(no passages found)")
                     + f"\n\nQuestion: {args['question']}")

    def summarize(args: dict[str, str]) -> list[dict[str, Any]]:
        doc = engine.get_document(args["name"])
        return _user(f"Summarise the document {doc.source} in five bullet points.\n\nOutline:\n{engine.outline(args['name'])}\n\n"
                     f"Start of the text:\n{doc.text[:1500]}")

    registry.register(Prompt("answer_with_citations", "Answer a question from the indexed documents, citing passages.",
                             [{"name": "question", "description": "What to answer", "required": True},
                              {"name": "top_k", "description": "How many passages to retrieve", "required": False}], answer))
    registry.register(Prompt("summarize_document", "Summarise one indexed document.",
                             [{"name": "name", "description": "Document id or source name", "required": True}], summarize))
    return registry
