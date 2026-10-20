"""Source code files. The text is kept exactly as written; the loader only records the language and size."""
from __future__ import annotations

from pathlib import PurePosixPath

from localrag.ingest.loaders.base import LoadedText, Loader, decode_bytes

LANGUAGES = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust", ".java": "java", ".c": "c",
             ".h": "c", ".cpp": "cpp", ".rb": "ruby", ".sh": "shell", ".sql": "sql", ".kt": "kotlin", ".swift": "swift"}


class CodeLoader(Loader):
    extensions = tuple(LANGUAGES)

    def load(self, raw: bytes, name: str) -> LoadedText:
        text = decode_bytes(raw, name).replace("\r\n", "\n")
        lang = LANGUAGES.get(PurePosixPath(name).suffix.lower(), "text")
        return LoadedText(text.strip("\n"), {"format": "code", "language": lang, "title": name, "lines": text.count("\n") + 1})
