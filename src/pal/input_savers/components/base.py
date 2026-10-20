from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Method:
    id: str
    name: str
    rule: str  # the one edit the local LLM is allowed to make
    system: str = ""  # optional full applier system prompt (used by the rules-file mode)


@dataclass
class StepResult:
    method_id: str
    name: str
    status: str               # "kept" | "reverted" | "skipped"
    reason: str
    tokens_before: int
    tokens_after: int         # equals tokens_before when reverted/skipped
    chunks_total: int = 0
    chunks_kept: int = 0
    chunk_notes: list = field(default_factory=list)  # per reverted chunk: why
    before: str = ""
    after: str = ""


@dataclass
class SaverResult:
    original: str
    reduced: str
    steps: list
    tokens_original: int
    tokens_reduced: int
    counter: str = ""

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_original - self.tokens_reduced)
