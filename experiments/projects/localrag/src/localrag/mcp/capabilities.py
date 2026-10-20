"""Version negotiation and the capabilities the server announces during the initialize handshake."""
from __future__ import annotations

from typing import Any

from localrag.core.config import ServerConfig

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


def negotiate_version(requested: str | None, supported: tuple[str, ...] = SUPPORTED_VERSIONS) -> str:
    """The protocol version to use: the client's if the server speaks it, otherwise the server's newest."""
    return requested if requested in supported else supported[0]


def server_capabilities() -> dict[str, Any]:
    """What this server can do: tools, resources and prompts, all with static lists (no change notifications)."""
    return {"tools": {"listChanged": False}, "resources": {"listChanged": False, "subscribe": False}, "prompts": {"listChanged": False}}


def server_info(config: ServerConfig) -> dict[str, str]:
    return {"name": config.server_name, "version": config.server_version}
