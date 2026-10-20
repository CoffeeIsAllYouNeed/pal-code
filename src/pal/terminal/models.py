"""Which local model is the strongest one on this machine?

"Strongest" here means the largest chat model Ollama has installed, judged by parameter count (from Ollama's own details, else from the name,
else from the file size). That is a proxy: a bigger model is usually better but needs more memory, so the page shows the choice and lets you
change it. Embedding models are never chosen: they cannot chat.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

EMBEDDING_HINTS = ("embed", "bge-", "minilm", "nomic", "e5-", "gte-", "snowflake-arctic", "mxbai", "paraphrase", "rerank")
_GB_PER_BILLION = 0.55          # a 4-bit model is roughly 0.55 GB per billion parameters: only used when nothing else says how big it is
_DETAIL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([BM])\s*$", re.I)
_TAG = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)b(?![a-z])", re.I)


@dataclass(frozen=True)
class LocalModel:
    name: str
    params_b: float | None       # billions of parameters, when known
    size_gb: float | None        # size on disk
    family: str
    quant: str
    embedding: bool

    @property
    def power(self) -> float:
        """The number models are ranked by: parameters in billions, else estimated from the file size, else 0."""
        if self.params_b is not None:
            return self.params_b
        return round(self.size_gb / _GB_PER_BILLION, 2) if self.size_gb else 0.0

    def label(self) -> str:
        bits = [self.name]
        if self.params_b is not None:
            bits.append(f"{self.params_b:g}B")
        if self.size_gb:
            bits.append(f"{self.size_gb:.1f} GB")
        return " | ".join(bits)


def _params_from(name: str, details: dict) -> float | None:
    m = _DETAIL.match(str(details.get("parameter_size") or ""))
    if m:
        value = float(m.group(1))
        return value / 1000 if m.group(2).upper() == "M" else value
    tag = name.split(":", 1)[1] if ":" in name else ""
    found = _TAG.search(tag) or _TAG.search(name)
    return float(found.group(1)) if found else None


def parse_tags(payload: Any) -> list[LocalModel]:
    """Ollama's /api/tags -> models. Tolerant: anything that is not a named model is skipped."""
    out: list[LocalModel] = []
    items = payload.get("models") if isinstance(payload, dict) else None
    for m in items if isinstance(items, list) else []:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or m.get("model") or "")
        if not name:
            continue
        details: dict = m["details"] if isinstance(m.get("details"), dict) else {}
        size = m.get("size")
        out.append(LocalModel(name=name, params_b=_params_from(name, details), size_gb=round(size / 1e9, 2) if isinstance(size, (int, float)) and size > 0 else None,
                              family=str(details.get("family") or ""), quant=str(details.get("quantization_level") or ""),
                              embedding=any(h in name.lower() or h in str(details.get("family") or "").lower() for h in EMBEDDING_HINTS)))
    return sorted(out, key=lambda x: x.name)


def strongest(models: list[LocalModel]) -> LocalModel | None:
    """The largest chat model (ties: the bigger file, then the name), or None when there is none."""
    chat = [m for m in models if not m.embedding]
    return max(chat, key=lambda m: (m.power, m.size_gb or 0.0, m.name)) if chat else None


def fetch_tags(url: str, get: Any = None, timeout: float = 3.0) -> Any:
    if get is None:
        import requests
        get = requests.get
    r = get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
    r.raise_for_status()
    return r.json()


def discover(url: str, get: Any = None) -> dict:
    """What the page needs: whether Ollama answers, the models it has, and the recommended one."""
    try:
        models = parse_tags(fetch_tags(url, get))
    except Exception as e:  # noqa: BLE001  (not running, not reachable, or not Ollama)
        return {"ok": False, "url": url, "models": [], "recommended": None, "error": f"Ollama is not reachable at {url} ({type(e).__name__}). Start it with `ollama serve`."}
    best = strongest(models)
    error = None
    if not models:
        error = "Ollama is running but has no models. Pull one, for example: ollama pull llama3.1:8b"
    elif best is None:
        error = "Ollama has only embedding models, which cannot chat. Pull a chat model, for example: ollama pull llama3.1:8b"
    return {"ok": best is not None, "url": url, "models": [{**asdict(m), "label": m.label()} for m in models], "recommended": best.name if best else None, "error": error}
