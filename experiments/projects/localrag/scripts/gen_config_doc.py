"""Writes docs/CONFIG.md from the settings classes in localrag/core/config.py, so the reference can never disagree with the code.

    python scripts/gen_config_doc.py            (rewrites docs/CONFIG.md)
"""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localrag.core import config as cfg  # noqa: E402

SECTIONS = [("chunk", cfg.ChunkConfig, "How documents are cut into chunks."), ("embed", cfg.EmbedConfig, "How text becomes vectors."),
            ("index", cfg.IndexConfig, "The vector index, the keyword index and how their results are fused."),
            ("retrieve", cfg.RetrieveConfig, "How a query is answered."), ("tree", cfg.TreeConfig, "The document tree and tree search."),
            ("server", cfg.ServerConfig, "The MCP server.")]


def comments() -> dict[str, str]:
    """The trailing comment of each setting line in config.py."""
    found = {}
    for line in (ROOT / "src/localrag/core/config.py").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+(\w+): [\w\[\], |]+ = .+?\s+# (.+)$", line)
        if m:
            found[m.group(1)] = m.group(2).strip()
    return found


def render() -> str:
    notes = comments()
    out = ["# Configuration reference", "",
           "Every setting has a default, so LocalRAG works with no configuration. Override settings in a JSON file (`--config file.json`), "
           "with nested objects named as below, or with the `LOCALRAG_*` environment variables listed at the end. Unknown names and values of "
           "the wrong type are errors. This file is generated from `src/localrag/core/config.py` by `scripts/gen_config_doc.py`.", "",
           "Example `localrag.json`:", "", "```json", '{"chunk": {"chunk_size": 800}, "retrieve": {"top_k": 8}, "index": {"vector_index": "ivf"}}', "```", ""]
    for name, klass, blurb in SECTIONS:
        out += [f"## `{name}`", "", blurb, "", "| Setting | Default | Meaning |", "|---|---|---|"]
        for f in dataclasses.fields(klass):
            default = f.default if f.default is not dataclasses.MISSING else ""
            out.append(f"| `{f.name}` | `{default!r}` | {notes.get(f.name, '')} |")
        out.append("")
    out += ["## Top level", "", "| Setting | Default | Meaning |", "|---|---|---|", f"| `data_dir` | `{cfg.Config().data_dir!r}` | Where the snapshot file is kept. |", "",
            "## Environment variables", "", "| Variable | Setting |", "|---|---|"]
    out += [f"| `{var}` | `{'.'.join(path)}` |" for var, path in cfg._ENV.items()]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    (ROOT / "docs" / "CONFIG.md").write_text(render(), encoding="utf-8")
    print("wrote docs/CONFIG.md")
