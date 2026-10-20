"""Which local model to run on which computer.

The catalogue lists well-known models from the Ollama library with their 4-bit download size (Ollama's default `q4_K_M` builds; sizes are
approximate). From the size Pal works out the video memory each needs, so it can group them by GPU, and pick for this computer:

- the **main model**, which writes and plans when you choose a local model: the strongest one that fits next to the helper;
- the **helper**, the small model Pal runs for its own jobs on every request (routing, trimming what you type, checking answers): a fast one.

Models that are already installed but not in the catalogue are judged the same way, from their size. Reasoning models (deepseek-r1, qwen3) print
their thinking before the answer, so they are listed but not picked automatically.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from pal.terminal.hardware import Hardware

CONTEXT_GB = 1.0          # room for the conversation (an 8k-token context) on top of the weights
OVERHEAD = 1.1            # runtime buffers on top of the file size
USABLE = 0.92             # the share of VRAM a model can really use (the desktop and drivers keep some)
TIERS = [0, 4, 6, 8, 12, 16, 24, 48, 80]     # GB of VRAM (0 = no GPU, runs on the CPU)


@dataclass(frozen=True)
class LocalModel:
    tag: str
    params_b: float
    size_gb: float
    use: str               # "code" | "general" | "reasoning" | "small"
    rank: int              # how well it builds and plans in Pal (higher is better)
    helper: bool = False   # fast and follows short instructions well: a good helper
    thinks: bool = False   # prints its reasoning first: not picked automatically
    note: str = ""

    @property
    def need_gb(self) -> float:
        """Memory to run it with a working context."""
        return round(self.size_gb * OVERHEAD + CONTEXT_GB, 1)

    @property
    def min_vram_gb(self) -> int:
        """The smallest GPU tier it fits on fully."""
        need = self.need_gb / USABLE
        return next((t for t in TIERS[1:] if t >= need), math.ceil(need))


CATALOGUE: list[LocalModel] = [
    # tiny: helpers, and machines without a GPU
    LocalModel("qwen2.5:0.5b", 0.5, 0.4, "small", 22, helper=True, note="tiny and fast; a helper for small machines"),
    LocalModel("qwen3:0.6b", 0.6, 0.5, "small", 25, thinks=True),
    LocalModel("gemma3:1b", 1.0, 0.8, "small", 34),
    LocalModel("llama3.2:1b", 1.2, 1.3, "small", 35, helper=True),
    LocalModel("qwen2.5:1.5b", 1.5, 1.0, "small", 42, helper=True, note="the default helper: fast, follows instructions"),
    LocalModel("qwen2.5-coder:1.5b", 1.5, 1.0, "code", 46, note="small coder for CPU-only machines"),
    LocalModel("qwen3:1.7b", 1.7, 1.4, "small", 45, thinks=True),
    LocalModel("llama3.2:3b", 3.2, 2.0, "small", 52, helper=True, note="the best helper when there is room"),
    LocalModel("qwen2.5:3b", 3.1, 1.9, "small", 53, helper=True),
    LocalModel("qwen2.5-coder:3b", 3.1, 1.9, "code", 56, note="coder for 4 GB GPUs and CPUs"),
    LocalModel("phi4-mini", 3.8, 2.5, "general", 57),
    LocalModel("qwen3:4b", 4.0, 2.6, "general", 60, thinks=True),
    LocalModel("gemma3:4b", 4.3, 3.3, "general", 58),
    # 7-9B: 6-8 GB GPUs
    LocalModel("mistral:7b", 7.2, 4.1, "general", 62),
    LocalModel("qwen2.5:7b", 7.6, 4.7, "general", 71),
    LocalModel("qwen2.5-coder:7b", 7.6, 4.7, "code", 75, note="best coder for 8 GB GPUs"),
    LocalModel("llama3.1:8b", 8.0, 4.9, "general", 70),
    LocalModel("qwen3:8b", 8.2, 5.2, "general", 73, thinks=True),
    LocalModel("deepseek-r1:8b", 8.0, 5.2, "reasoning", 69, thinks=True),
    LocalModel("gemma2:9b", 9.2, 5.4, "general", 68),
    # 12-16B: 12-16 GB GPUs
    LocalModel("mistral-nemo:12b", 12.2, 7.1, "general", 74),
    LocalModel("gemma3:12b", 12.0, 8.1, "general", 77),
    LocalModel("deepseek-coder-v2:16b", 16.0, 8.9, "code", 78, note="fast for its size (mixture of experts)"),
    LocalModel("qwen2.5:14b", 14.7, 9.0, "general", 80),
    LocalModel("qwen2.5-coder:14b", 14.7, 9.0, "code", 83, note="best coder for 12-16 GB GPUs"),
    LocalModel("deepseek-r1:14b", 14.8, 9.0, "reasoning", 79, thinks=True),
    LocalModel("phi4:14b", 14.7, 9.1, "general", 81),
    LocalModel("qwen3:14b", 14.8, 9.3, "general", 82, thinks=True),
    # 22-32B: 24 GB GPUs
    LocalModel("codestral:22b", 22.2, 13.0, "code", 85),
    LocalModel("mistral-small:24b", 23.6, 14.0, "general", 84),
    LocalModel("gemma3:27b", 27.4, 17.0, "general", 86),
    LocalModel("qwen3:30b", 30.5, 19.0, "general", 87, thinks=True, note="mixture of experts: fast for its size"),
    LocalModel("qwen2.5:32b", 32.8, 20.0, "general", 89),
    LocalModel("qwen2.5-coder:32b", 32.8, 20.0, "code", 95, note="best local coder for 24 GB GPUs"),
    LocalModel("qwen3:32b", 32.8, 20.0, "general", 91, thinks=True),
    LocalModel("deepseek-r1:32b", 32.8, 20.0, "reasoning", 88, thinks=True),
    # 70B+: 48 GB and more
    LocalModel("llama3.3:70b", 70.6, 43.0, "general", 92),
    LocalModel("deepseek-r1:70b", 70.6, 43.0, "reasoning", 90, thinks=True),
    LocalModel("qwen2.5:72b", 72.7, 47.0, "general", 94),
    # newer models (sizes are Ollama's default downloads, October 2026)
    LocalModel("qwen3.5:0.8b", 0.8, 1.3, "small", 30, thinks=True),
    LocalModel("qwen3.5:2b", 2.0, 2.7, "small", 50, thinks=True),
    LocalModel("qwen3.5:4b", 4.0, 3.4, "general", 64, thinks=True),
    LocalModel("gemma4:e2b", 2.0, 4.6, "general", 55, thinks=True, note="reads images too"),
    LocalModel("gemma4:e4b", 4.0, 6.6, "general", 66, thinks=True, note="reads images too"),
    LocalModel("qwen3.5:9b", 9.0, 6.6, "general", 79, thinks=True, note="reads images too"),
    LocalModel("gemma4:12b", 12.0, 8.0, "general", 82, thinks=True, note="reads images too"),
    LocalModel("gpt-oss:20b", 21.0, 14.0, "reasoning", 90, thinks=True, note="OpenAI's open model; reasons before answering"),
    LocalModel("devstral-small-2:24b", 24.0, 15.0, "code", 93, note="agentic coding: multi-file edits, explores a codebase"),
    LocalModel("qwen3.5:27b", 27.0, 17.0, "general", 92, thinks=True),
    LocalModel("qwen3.6:27b", 27.0, 18.0, "code", 96, thinks=True, note="strong agentic coder; thinks first"),
    LocalModel("gemma4:26b", 26.0, 18.0, "general", 89, thinks=True, note="mixture of experts: fast for its size"),
    LocalModel("qwen3-coder:30b", 30.5, 19.0, "code", 97, note="best local coder for 24 GB GPUs: mixture of experts, fast, 256K context"),
    LocalModel("gemma4:31b", 31.0, 20.0, "general", 91, thinks=True),
    LocalModel("qwen3.5:35b", 35.0, 22.0, "general", 93, thinks=True, note="mixture of experts: fast for its size"),
    LocalModel("qwen3.6:35b", 35.0, 24.0, "code", 97, thinks=True, note="mixture of experts; strong agentic coder; thinks first"),
]
BY_TAG = {m.tag: m for m in CATALOGUE}


def budget_gb(hw: Hardware) -> float:
    """Memory models may use: on the GPU (or Apple's shared memory) when there is one, else half the RAM for running on the CPU."""
    return round(hw.vram_gb * (1.0 if hw.apple_silicon else USABLE), 1) if hw.vram_gb else round(hw.ram_gb * 0.5, 1)


def fit(need_gb: float, hw: Hardware) -> str:
    """'gpu' (fits on the GPU), 'partial' (part on the GPU, part on the CPU: works, slower), 'no gpu' (Pal does not run local models without one)
    or 'too big'."""
    if hw.vram_gb:
        room = budget_gb(hw)
        if need_gb <= room:
            return "gpu"
        if not hw.apple_silicon and need_gb <= room + hw.ram_gb * 0.5:
            return "partial"
        return "too big"
    return "no gpu"


FIT_LABEL = {"gpu": "fits your GPU", "partial": "partly on the CPU (slower)", "no gpu": "needs a GPU", "too big": "too big for this computer"}


def tier_name(hw: Hardware) -> str:
    if not hw.vram_gb:
        return "no GPU (API models only)"
    if hw.apple_silicon:
        return f"{hw.ram_gb:g} GB unified memory"
    vram = hw.vram_gb
    t = max((x for x in TIERS[1:] if x <= vram + 0.5), default=4)
    return f"{t} GB {'unified memory' if hw.apple_silicon else 'GPU'}"


@dataclass
class Recommendation:
    main: LocalModel
    helper: LocalModel
    others: list[LocalModel]          # other good choices that fit, best first
    reason: str


def pick_helper(hw: Hardware) -> LocalModel:
    room = budget_gb(hw)
    if hw.vram_gb and room >= 14:
        return BY_TAG["llama3.2:3b"]
    if (hw.vram_gb and room >= 4.5) or (not hw.vram_gb and hw.ram_gb >= 12):
        return BY_TAG["qwen2.5:1.5b"]
    return BY_TAG["qwen2.5:0.5b"]


def api_only(hw: Hardware) -> bool:
    """No GPU (and not Apple Silicon): Pal offers no local model on this computer and uses API models only."""
    return not hw.vram_gb


CODE_BONUS = 8          # Pal writes software: a coding model beats a general one of a similar rank (qwen2.5-coder:3b over mistral:7b)


def score(m: LocalModel) -> int:
    return m.rank + (CODE_BONUS if m.use == "code" else 0)


def recommend(hw: Hardware) -> Recommendation | None:
    """The best main model that fits the GPU (coding models first: Pal builds software), and the helper. The helper goes on the GPU beside the
    main model when there is room, else on the CPU, where a small model is still quick. On a CPU only, models are kept small enough to answer
    in reasonable time."""
    if api_only(hw):
        return None
    helper = pick_helper(hw)
    room = budget_gb(hw)
    usable = [m for m in CATALOGUE if not m.thinks and m.use != "small" and m.need_gb <= room]
    if not usable:
        usable = [m for m in CATALOGUE if not m.thinks and m.need_gb <= max(room, 1.6)] or [BY_TAG["qwen2.5-coder:1.5b"]]
    ranked = sorted(usable, key=score, reverse=True)
    main = ranked[0]
    where = "on your GPU" if hw.vram_gb else "on your CPU"
    if main.need_gb + helper.need_gb - CONTEXT_GB <= room or not hw.vram_gb:
        beside = "beside it"
    else:
        beside = "when needed (Ollama swaps the two in shared memory)" if hw.apple_silicon else "on the CPU, where it is still quick"
    reason = f"the strongest model that runs {where} ({main.need_gb:g} GB of about {room:g} GB); the helper runs {beside}"
    return Recommendation(main, helper, ranked[1:6], reason)


def by_tier() -> list[tuple[int, list[LocalModel]]]:
    """The catalogue grouped by the smallest GPU each model fits on fully."""
    groups: dict[int, list[LocalModel]] = {}
    for m in CATALOGUE:
        groups.setdefault(m.min_vram_gb, []).append(m)
    return [(t, sorted(ms, key=lambda m: -m.rank)) for t, ms in sorted(groups.items())]


_BASE = re.compile(r"^([^:]+):(\d+(?:\.\d+)?[bm])\b", re.I)


def lookup(name: str) -> LocalModel | None:
    """The catalogue entry for an installed model, also for longer tags of the same model (llama3.1:8b-instruct-q4_K_M is llama3.1:8b)."""
    if name in BY_TAG or name.removesuffix(":latest") in BY_TAG:
        return BY_TAG.get(name) or BY_TAG[name.removesuffix(":latest")]
    m = _BASE.match(name)
    return BY_TAG.get(f"{m.group(1)}:{m.group(2).lower()}") if m else None


def guess_rank(params_b: float) -> int:
    """A rank for a model the catalogue does not know, from its size: bigger is usually better, with diminishing returns."""
    return int(min(90, 30 + 7 * math.log2(params_b))) if params_b >= 1 else 20


def installed_fit(models: list[dict], hw: Hardware) -> list[dict]:
    """Ollama's installed models with how each fits: from the catalogue when known, else from the file size Ollama reports."""
    out = []
    for m in models:
        if m.get("embedding"):
            continue
        known = lookup(m["name"])
        need = round((m.get("size_gb") or 0) * OVERHEAD + CONTEXT_GB, 1) if m.get("size_gb") else (known.need_gb if known else CONTEXT_GB)
        out.append({**m, "need_gb": need, "fit": fit(need, hw), "rank": known.rank if known else guess_rank(m.get("params_b") or 0)})
    return out


def best_installed(models: list[dict], hw: Hardware, exclude: str = "") -> str | None:
    """The strongest installed chat model that fits the GPU (or CPU); if none fits, the smallest one."""
    rows = [r for r in installed_fit(models, hw) if r["name"] != exclude]
    if not rows:
        return None
    if api_only(hw):
        return None
    good = [r for r in rows if r["fit"] == "gpu"]
    return (max(good, key=lambda r: (r["rank"], r["need_gb"])) if good else min(rows, key=lambda r: r["need_gb"]))["name"]


def suitable(hw: Hardware, installed: list[str] | None = None, limit: int = 6) -> list[tuple[str, str, bool]]:
    """Main-model choices for this GPU, best first: (tag, one line about it, installed?). The recommended model leads, then installed models that
    fit, then the next best downloads. Reasoning models are left out (they can still be chosen by name)."""
    rec = recommend(hw)
    if rec is None:
        return []
    have = set()
    for h in installed or []:
        known = lookup(h)
        have.add(known.tag if known else h)
    picks = [rec.main] + [m for m in rec.others if m.tag != rec.main.tag]
    for name in installed or []:
        known = lookup(name)
        if known and known not in picks and not known.thinks and known.use != "small" and fit(known.need_gb, hw) == "gpu":
            picks.append(known)
    picks.sort(key=lambda m: (m.tag != rec.main.tag, m.tag not in have, -score(m)))
    return [(m.tag, (f"installed · {m.note or m.use}" if m.tag in have else f"{m.size_gb:g} GB download · {m.note or m.use}"), m.tag in have)
            for m in picks[:limit]]
