"""
Context interrupt mechanism for Round Robin scheduling (AIOS section
3.3 / detailed in their Appendix A.3, which this module is the
counterpart of here).

Long-running LLM inference is exactly the kind of system call that can
monopolize a shared resource -- one slow generation blocks everything
queued behind it unless something can pause it. FIFO scheduling doesn't
need that (it runs each request to completion by design); Round Robin
does, since giving every request a bounded time slice only works if a
request that isn't done yet can be suspended and picked back up later
without losing the work already done.

Two snapshot/restore approaches, matching what's actually available
per core:

- **Text-based** (`_run_slice_text` / `ContextSnapshot(mode="text")`):
  for cores with no logits access -- every cloud API and every local
  server this project talks to (Anthropic, Gemini, OpenAI-compatible,
  Ollama). None of them return raw per-step logits over HTTP, so the
  only state available to snapshot is the decoded text generated so
  far. Resumption re-injects that text as a partial assistant turn plus
  a short "continue exactly where you left off" instruction, and lets
  the model carry on from there.

- **Logits-based** (`_run_slice_logits` / `ContextSnapshot(mode="logits")`):
  for cores with direct logits access -- currently only
  HuggingFaceLocalLLM, the in-process local model. Generation is
  driven one token at a time via manual beam-search stepping
  (HuggingFaceLocalLLM.beam_search_step), and the snapshot is the
  actual search-tree state (every beam's token sequence + cumulative
  score, see BeamSearchState in huggingface_local.py) rather than
  just its decoded text. Resuming continues the search from that exact
  state -- no re-tokenization, no risk of the model repeating or
  drifting from where it left off the way a text-based "please
  continue" prompt can.

Both are request-level approximations of "pause an in-flight
computation" -- neither literally suspends a running matrix
multiplication. What differs between them is how much of the model's
own state survives the pause: text-based keeps only what came out the
other end; logits-based keeps the search state that produced it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from pal.llm_wrapper.components.base import LLMSyscall, LLMWrapper

_CONTINUE_INSTRUCTION = (
    "[Continue your previous reply exactly where it left off. Do not "
    "repeat any earlier part of it, do not restart, and do not "
    "acknowledge this instruction.]"
)


@dataclass
class ContextSnapshot:
    """The suspended state of one interrupted generation.

    mode="text": text_so_far is the only state -- decoding must
    restart from a continuation prompt built around it.
    mode="logits": beam_state carries the actual search-tree state
    (an opaque, core-specific object -- BeamSearchState for
    HuggingFaceLocalLLM) that resumption steps forward directly,
    with text_so_far kept alongside purely for human-readable
    inspection/logging, not used to drive resumption.
    """

    mode: str  # "text" | "logits"
    text_so_far: str = ""
    beam_state: Any = None


@dataclass
class InterruptResult:
    complete: bool
    text_so_far: str
    elapsed: float
    snapshot: ContextSnapshot
    # True when text_so_far is the FULL cumulative text generated across
    # every slice so far (logits mode -- beam decoding naturally tracks
    # whole sequences). False when it's only this slice's delta, to be
    # appended to whatever came before (text mode -- each streamed
    # continuation only yields its own new chunk).
    cumulative: bool = False
    chunks_yielded: int = 0


class ContextInterruptController:
    """Runs one LLMSyscall against one core for at most `quantum`
    seconds, choosing the text-based or logits-based approach based on
    the core's capabilities(), and returns either a complete result or
    a ContextSnapshot the scheduler can use to resume later."""

    def run_slice(
        self,
        core: LLMWrapper,
        syscall: LLMSyscall,
        quantum: float,
        resume_snapshot: Optional[ContextSnapshot] = None,
    ) -> InterruptResult:
        use_logits = (resume_snapshot is not None and resume_snapshot.mode == "logits") or (
            resume_snapshot is None
            and core.capabilities().supports_logits_snapshot
            and hasattr(core, "init_beam_search")
            and syscall.params.get("use_logits_snapshot", True)
        )
        if use_logits:
            return self._run_slice_logits(core, syscall, quantum, resume_snapshot)
        return self._run_slice_text(core, syscall, quantum)

    # -- text-based: closed-source / no-logits-access cores --------------------

    def _run_slice_text(self, core: LLMWrapper, syscall: LLMSyscall, quantum: float) -> InterruptResult:
        text_parts: list[str] = []
        t0 = time.time()
        chunks = 0

        for chunk in core.stream_chat(syscall):
            text_parts.append(chunk)
            chunks += 1
            elapsed = time.time() - t0
            if elapsed >= quantum:
                # Time slice exhausted mid-generation -- stop pulling
                # from the stream here. Cores without real incremental
                # streaming (LLMWrapper.stream_chat's default fallback,
                # which yields the whole response as one chunk) will
                # already have finished by the time the first chunk
                # arrives, so this branch only actually interrupts
                # something for cores with true token-level streaming.
                text_so_far = "".join(text_parts)
                return InterruptResult(
                    complete=False,
                    text_so_far=text_so_far,
                    elapsed=elapsed,
                    snapshot=ContextSnapshot(mode="text", text_so_far=text_so_far),
                    cumulative=False,
                    chunks_yielded=chunks,
                )

        text_so_far = "".join(text_parts)
        return InterruptResult(
            complete=True,
            text_so_far=text_so_far,
            elapsed=time.time() - t0,
            snapshot=ContextSnapshot(mode="text", text_so_far=text_so_far),
            cumulative=False,
            chunks_yielded=chunks,
        )

    def build_continuation(self, syscall: LLMSyscall, partial: InterruptResult) -> LLMSyscall:
        """Text-based resumption: folds the partial output back into the
        message list as prior context and returns a new LLMSyscall ready
        to be re-queued. Only meaningful for mode="text" results -- the
        scheduler doesn't call this for logits-mode interrupts, since
        those resume from `beam_state` directly instead of rebuilding
        the prompt."""
        continued_messages = list(syscall.messages) + [
            {"role": "assistant", "content": partial.text_so_far},
            {"role": "user", "content": _CONTINUE_INSTRUCTION},
        ]
        return LLMSyscall(
            messages=continued_messages,
            model=syscall.model,
            temperature=syscall.temperature,
            max_tokens=syscall.max_tokens,
            stream=syscall.stream,
            params=syscall.params,
        )

    # -- logits-based: direct logits access, search-tree state -----------------

    def _run_slice_logits(
        self,
        core: LLMWrapper,
        syscall: LLMSyscall,
        quantum: float,
        resume_snapshot: Optional[ContextSnapshot],
    ) -> InterruptResult:
        t0 = time.time()
        beam_width = syscall.params.get("beam_width", 1)
        max_new_tokens = syscall.max_tokens or 512

        state = resume_snapshot.beam_state if resume_snapshot is not None else core.init_beam_search(
            syscall, beam_width=beam_width
        )

        steps_this_slice = 0
        while not state.finished:
            state = core.beam_search_step(state, max_new_tokens=max_new_tokens)
            steps_this_slice += 1
            elapsed = time.time() - t0

            if elapsed >= quantum and not state.finished:
                # Suspended mid-generation -- snapshot the intermediate
                # search-tree state (not just its decoded text) so
                # resumption continues the beam search exactly where
                # this step left off.
                decoded = core.decode_beam_search(state)
                return InterruptResult(
                    complete=False,
                    text_so_far=decoded,
                    elapsed=elapsed,
                    snapshot=ContextSnapshot(mode="logits", text_so_far=decoded, beam_state=state),
                    cumulative=True,
                    chunks_yielded=steps_this_slice,
                )

        final_text = core.decode_beam_search(state)
        return InterruptResult(
            complete=True,
            text_so_far=final_text,
            elapsed=time.time() - t0,
            snapshot=ContextSnapshot(mode="logits", text_so_far=final_text, beam_state=state),
            cumulative=True,
            chunks_yielded=steps_this_slice,
        )
