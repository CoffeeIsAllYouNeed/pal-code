"""Third-party service tools, the AIOS paper's Table 5 tools beyond the ones Pal already had (Wikipedia, arXiv)."""
from __future__ import annotations

from collections.abc import Iterable

from pal.chat.services import data, media, web
from pal.chat.services.base import ServiceCtx, ServiceSpec
from pal.errors import ConfigError

ALL_SERVICES: tuple[ServiceSpec, ...] = (*web.SERVICES, *data.SERVICES, *media.SERVICES)
SERVICE_NAMES: tuple[str, ...] = tuple(s.name for s in ALL_SERVICES)
BY_NAME = {s.name: s for s in ALL_SERVICES}


def selected(names: Iterable[str]) -> list[ServiceSpec]:
    """The services a `tools.services` list switches on. "all" means every one. An unknown name is an error, not ignored."""
    wanted = [str(n).strip() for n in names]
    if "all" in wanted:
        return list(ALL_SERVICES)
    unknown = [n for n in wanted if n not in BY_NAME]
    if unknown:
        raise ConfigError(f"tools.services: unknown service(s) {unknown}; choose from {', '.join(SERVICE_NAMES)}, or all")
    return [BY_NAME[n] for n in dict.fromkeys(wanted)]


__all__ = ["ALL_SERVICES", "BY_NAME", "SERVICE_NAMES", "ServiceCtx", "ServiceSpec", "selected"]
