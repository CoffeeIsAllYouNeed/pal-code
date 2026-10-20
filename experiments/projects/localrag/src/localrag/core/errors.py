"""Every error LocalRAG raises on purpose derives from LocalRAGError, so callers can catch one type."""
from __future__ import annotations


class LocalRAGError(Exception):
    """Base class for all errors raised deliberately by LocalRAG."""


class ConfigError(LocalRAGError):
    """A setting is missing, has the wrong type, or is outside its allowed range."""


class IngestError(LocalRAGError):
    """A file or text could not be read, decoded or split into chunks."""


class IndexingError(LocalRAGError):
    """The vector or keyword index was asked to do something it cannot do (wrong dimension, unknown id, bad snapshot)."""


class RetrievalError(LocalRAGError):
    """A query could not be answered, for example because it is empty or the filter is malformed."""


class ProtocolError(LocalRAGError):
    """A JSON-RPC or MCP message was malformed. Carries the JSON-RPC error code to send back."""

    def __init__(self, message: str, code: int = -32600, data: object = None):
        super().__init__(message)
        self.code = code
        self.data = data
