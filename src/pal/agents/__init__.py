"""The agent application layer. Only agents live here.

Everything an agent needs (spec type, base class, ReAct and Reflexion patterns, framework adapters) is in
`pal.sdk`. Add an agent by adding a NativeAgent subclass with a spec and listing it below.
"""
from pal.agents.coding import CodingAgent
from pal.agents.docs import DocumentationAgent
from pal.sdk.agent import NativeAgent

AGENTS: dict[str, type[NativeAgent]] = {a.spec.id: a for a in (CodingAgent, DocumentationAgent) if a.spec}

__all__ = ["AGENTS", "CodingAgent", "DocumentationAgent"]
