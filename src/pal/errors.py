"""Exception hierarchy. Everything Pal raises on purpose derives from PalError, so the
web layer can map it to an HTTP status without string matching and never leaks a traceback."""
from __future__ import annotations


class PalError(Exception):
    """Base class. `http_status` is what the web layer answers with; `public` says whether the
    message is safe to show the user as-is."""

    http_status = 500
    public = False


class UserInputError(PalError, ValueError):
    """The request was wrong: bad file, empty message, unknown option. Message is user-safe."""

    http_status = 400
    public = True


class ConfigError(PalError, ValueError):
    """Settings are invalid. Raised at startup, so a bad config fails fast and clearly."""

    http_status = 500
    public = True


class ProviderError(PalError):
    """The API model or the local model failed (network, auth, rate limit, bad response)."""

    http_status = 502
    public = True


class DependencyMissingError(UserInputError):
    """An optional dependency or system tool (pandoc, graphify) is not installed."""


class NotFoundError(UserInputError):
    http_status = 404
