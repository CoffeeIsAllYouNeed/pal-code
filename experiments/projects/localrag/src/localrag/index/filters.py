"""Metadata filters: a small query language over chunk metadata, written as plain dictionaries (so it works over JSON)."""
from __future__ import annotations

from typing import Any

from localrag.core.errors import RetrievalError

_OPS = {
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "gt": lambda a, b: a is not None and a > b, "gte": lambda a, b: a is not None and a >= b,
    "lt": lambda a, b: a is not None and a < b, "lte": lambda a, b: a is not None and a <= b, "in": lambda a, b: a in b,
    "contains": lambda a, b: isinstance(a, str) and str(b).lower() in a.lower(), "exists": lambda a, b: (a is not None) == bool(b),
}


def matches(metadata: dict[str, Any], spec: dict[str, Any] | None) -> bool:
    """Does a chunk's metadata satisfy the filter? Forms: {"k": v}, {"k": {"gte": 3}}, {"and": [..]}, {"or": [..]}, {"not": {..}}."""
    if not spec:
        return True
    if not isinstance(spec, dict):
        raise RetrievalError("a filter must be an object")
    for key, cond in spec.items():
        if key == "and":
            ok = all(matches(metadata, s) for s in _list(cond, key))
        elif key == "or":
            ok = any(matches(metadata, s) for s in _list(cond, key))
        elif key == "not":
            ok = not matches(metadata, cond)
        elif isinstance(cond, dict):
            ok = all(_apply(op, metadata.get(key), arg) for op, arg in cond.items())
        else:
            ok = metadata.get(key) == cond
        if not ok:
            return False
    return True


def _list(cond: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(cond, list) or not cond:
        raise RetrievalError(f"{key} needs a non-empty list of filters")
    return cond


def _apply(op: str, value: Any, arg: Any) -> bool:
    if op not in _OPS:
        raise RetrievalError(f"unknown filter operator {op!r}; known: {', '.join(sorted(_OPS))}")
    try:
        return bool(_OPS[op](value, arg))
    except TypeError:
        return False
