"""JSON files, flattened to one "path: value" line per leaf so that every fact can be found by its key path."""
from __future__ import annotations

import json
from typing import Any

from localrag.core.errors import IngestError
from localrag.ingest.loaders.base import LoadedText, Loader, decode_bytes


def flatten(value: Any, prefix: str = "") -> list[str]:
    """Lines of the form path: value for every scalar inside nested dicts and lists."""
    if isinstance(value, dict):
        return [line for k, v in value.items() for line in flatten(v, f"{prefix}.{k}" if prefix else str(k))]
    if isinstance(value, list):
        return [line for i, v in enumerate(value) for line in flatten(v, f"{prefix}[{i}]")]
    return [f"{prefix}: {value}"]


class JsonLoader(Loader):
    extensions = (".json", ".jsonl")

    def load(self, raw: bytes, name: str) -> LoadedText:
        text = decode_bytes(raw, name)
        try:
            if name.lower().endswith(".jsonl"):
                values = [json.loads(line) for line in text.splitlines() if line.strip()]
                lines = [ln for i, v in enumerate(values) for ln in flatten(v, f"record{i}")]
            else:
                lines = flatten(json.loads(text))
        except json.JSONDecodeError as e:
            raise IngestError(f"{name} is not valid JSON: {e}") from e
        return LoadedText("\n".join(lines), {"format": "json", "title": name, "leaves": len(lines)})
