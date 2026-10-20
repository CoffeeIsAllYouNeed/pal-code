"""Typed queries an agent sends through the SDK, and where each kind goes.

    LLMQuery                            -> Pal kernel   (model work: reasoning, planning, tool parsing, code)
    MemoryQuery, StorageQuery, ToolQuery -> Pal kernel   (its memory, storage and tool managers)
    OSQuery                             -> OS kernel       (file operations, network requests, system facts)

An OSQuery never reaches a model and an LLMQuery never touches the filesystem or the network: the router in
`AgentSDK.send_request` is the only place that decides, and it decides by query class alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

PAL_KERNEL = "pal_kernel"
OS_KERNEL = "os_kernel"


@dataclass
class Query:
    query_class: ClassVar[str] = ""


@dataclass
class LLMQuery(Query):
    query_class: ClassVar[str] = "llm"
    messages: list[dict] = field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass
class MemoryQuery(Query):
    """op: remember | recall"""
    query_class: ClassVar[str] = "memory"
    op: str = "recall"
    params: dict = field(default_factory=dict)


@dataclass
class StorageQuery(Query):
    """op: read | write | delete | list | add_knowledge | search_knowledge"""
    query_class: ClassVar[str] = "storage"
    op: str = "read"
    params: dict = field(default_factory=dict)


@dataclass
class ToolQuery(Query):
    query_class: ClassVar[str] = "tool"
    tool: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class OSQuery(Query):
    """op: file_read | file_write | file_list | file_delete | net_get | net_post | sys_info | keyboard | mouse"""
    query_class: ClassVar[str] = "os"
    op: str = "sys_info"
    params: dict = field(default_factory=dict)


@dataclass
class Response:
    ok: bool
    data: Any = None
    error: str = ""
    handled_by: str = ""      # PAL_KERNEL or OS_KERNEL
