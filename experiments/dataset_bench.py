"""
Loads human-only prompts from
https://huggingface.co/datasets/cfahlgren1/llama-3.1-awesome-chatgpt-prompts
for use as a fixed benchmark set.

Uses Hugging Face's datasets-server "rows" JSON API (plain HTTP via
`requests`) rather than downloading and parsing the dataset's parquet
file with pandas/pyarrow. pyarrow ships compiled DLLs that some
locked-down Windows environments (Application Control / WDAC policies)
refuse to load; this avoids that dependency entirely.

The dataset has 47 rows. Each row's `messages` field is a list containing
one entry: {"role": "user", "content": "<prompt>"}. We deliberately pull
ONLY that user content — not the assistant_message column that's also in
the dataset — since the benchmark needs to run these prompts against
live backends (Ollama/Gemini) itself, not replay recorded answers.
"""
import json
import os

import requests

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_prompts_cache.json")

_DATASET = "cfahlgren1/llama-3.1-awesome-chatgpt-prompts"
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_PAGE_SIZE = 100  # datasets-server's max rows per request; this dataset has 47


def _fetch_from_hub() -> list[str]:
    prompts = []
    offset = 0

    while True:
        resp = requests.get(
            _ROWS_API,
            params={
                "dataset": _DATASET,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": _PAGE_SIZE,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("rows", [])
        if not rows:
            break

        for entry in rows:
            messages = entry.get("row", {}).get("messages", [])
            for m in messages:
                if m.get("role") == "user":
                    prompts.append(m.get("content", ""))
                    break

        offset += len(rows)
        total = data.get("num_rows_total")
        if (total is not None and offset >= total) or len(rows) < _PAGE_SIZE:
            break

    return prompts


def load_human_prompts(limit: int = None, refresh: bool = False) -> list[str]:
    """Returns a list of human/user prompt strings from the dataset.
    Caches to disk after the first successful fetch (network call), since
    the dataset is static (47 rows) and re-fetching every benchmark run is
    wasteful."""
    if not refresh and os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, encoding="utf-8") as f:
            prompts = json.load(f)
    else:
        prompts = _fetch_from_hub()
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(prompts, f)
    if limit is not None:
        return prompts[:limit]
    return prompts
