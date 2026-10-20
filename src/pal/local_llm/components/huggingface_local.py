"""
Fully in-process local core: the model is loaded directly into this
Python process via transformers, no server (Ollama/llama.cpp/vLLM)
involved at all. This is the deployment option AIOS 3.2 calls out
explicitly -- "whether the LLM is hosted on cloud or on local device,
what hardware conditions the LLM requires" -- that a server-based local
core doesn't cover, since a server is still a separate process/host
with its own port.

Heavy deps (torch, transformers) are imported lazily inside __init__,
not at module import time, so importing pal.llm_wrapper doesn't
require them unless this specific core is actually built.

This is also the only core with direct logits access, which is what
lets it implement the beam-search step/snapshot/restore methods the
logits-based context interrupt mechanism needs (see
orchestrator/components/context_interrupt.py) -- no cloud or
local-server backend exposes raw per-step logits over its API, so
those cores are permanently limited to text-based snapshotting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pal.llm_wrapper.components.base import (
    DeploymentType,
    LLMResponse,
    LLMSyscall,
    LLMWrapper,
    WrapperCapabilities,
    WrapperConfig,
)


@dataclass
class BeamSearchState:
    """The "intermediate search tree structure" a logits-based snapshot
    preserves: every beam's full token sequence so far and its
    cumulative log-probability score. Restoring from this resumes
    decoding exactly where generation left off, one token at a time --
    unlike a text snapshot, nothing has to be re-tokenized or
    re-interpreted by the model to continue.

    Deliberately does NOT snapshot the model's KV cache (past_key_values)
    -- that would make resumption cheaper (no need to reprocess the
    prefix) but ties the snapshot to a single model instance's live
    memory, and isn't picklable/inspectable the way this plain
    tensor-of-token-ids state is. The tradeoff: resuming here recomputes
    the forward pass over the full sequence-so-far rather than just the
    new token, which is the honest cost of keeping snapshots simple and
    portable.
    """

    input_ids: Any             # torch.Tensor, shape (beam_width, seq_len)
    beam_scores: Any           # torch.Tensor, shape (beam_width,)
    prompt_len: int
    step: int = 0
    finished: bool = False


class HuggingFaceLocalLLM(LLMWrapper):
    name = "huggingface_local"

    def __init__(self, config: WrapperConfig):
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except (ImportError, OSError) as e:
            raise ImportError(
                "transformers and torch are required for the in-process local "
                "core. Run: pip install torch transformers"
            ) from e

        self._torch = torch
        self.model_name = config.model
        # "auto" lets accelerate/torch pick the best available device;
        # explicit device (e.g. "cuda", "cpu", "mps") pins it, matching
        # the "hardware conditions the LLM requires" concern from AIOS.
        device = config.device or "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype=config.extra.get("torch_dtype", "auto"),
        )

    def _render_prompt(self, messages: list[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # No chat template on this tokenizer/model -- fall back to a
        # plain role-prefixed transcript rather than failing outright.
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        prompt = self._render_prompt(syscall.messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.time()
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=syscall.max_tokens or 512,
                temperature=syscall.temperature if syscall.temperature is not None else 1.0,
                do_sample=syscall.temperature not in (None, 0),
                **syscall.params,
            )
        latency = time.time() - t0

        prompt_tokens = inputs["input_ids"].shape[-1]
        completion_tokens = output_ids.shape[-1] - prompt_tokens
        text = self.tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)

        return LLMResponse(
            text=text,
            raw=output_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            meta={"latency": latency, "device": str(self.model.device)},
        )

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # -- logits-based beam search: step / snapshot / restore -------------------
    #
    # Not part of LLMWrapper's abstract contract (see base.py's LLMWrapper
    # docstring) -- only reachable via capabilities().supports_logits_
    # snapshot, and only ever called by ContextInterruptController.

    def init_beam_search(self, syscall: LLMSyscall, beam_width: int = 1) -> BeamSearchState:
        prompt = self._render_prompt(syscall.messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"].repeat(beam_width, 1)
        beam_scores = self._torch.zeros(beam_width, device=self.model.device)
        return BeamSearchState(input_ids=input_ids, beam_scores=beam_scores, prompt_len=input_ids.shape[-1])

    def beam_search_step(self, state: BeamSearchState, max_new_tokens: int = 512) -> BeamSearchState:
        """Advances every beam by exactly one token. This single-token
        granularity is the actual snapshot boundary: the context
        interrupt mechanism can pause between ANY two calls to this
        method, not just between whole streamed chunks the way the
        text-based path is limited to."""
        if state.finished:
            return state

        torch = self._torch
        with torch.no_grad():
            outputs = self.model(state.input_ids)
            next_token_logits = outputs.logits[:, -1, :]  # (beam_width, vocab)
            next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
            beam_width = state.input_ids.shape[0]

            if beam_width == 1:
                # Beam width 1 (this project's default, matching the
                # AIOS example): greedy is equivalent to beam search
                # with one beam, and avoids the topk/reshape machinery
                # below for the common case.
                next_token_id = next_token_logprobs.argmax(dim=-1, keepdim=True)
                new_scores = state.beam_scores + next_token_logprobs.gather(-1, next_token_id).squeeze(-1)
                new_input_ids = torch.cat([state.input_ids, next_token_id], dim=-1)
            else:
                # True beam search: expand every beam by its full
                # vocabulary of candidates, then keep only the top
                # `beam_width` sequences overall (not top-1-per-beam --
                # that's what lets a single strong beam claim more than
                # one slot next step, same as standard beam search).
                vocab_size = next_token_logprobs.shape[-1]
                candidate_scores = (state.beam_scores.unsqueeze(-1) + next_token_logprobs).view(-1)
                top_scores, top_indices = candidate_scores.topk(beam_width)
                beam_indices = top_indices // vocab_size
                token_indices = (top_indices % vocab_size).unsqueeze(-1)
                new_input_ids = torch.cat([state.input_ids[beam_indices], token_indices], dim=-1)
                new_scores = top_scores

        generated_len = new_input_ids.shape[-1] - state.prompt_len
        eos_id = self.tokenizer.eos_token_id
        finished = generated_len >= max_new_tokens or (
            eos_id is not None and bool((new_input_ids[:, -1] == eos_id).all())
        )

        return BeamSearchState(
            input_ids=new_input_ids,
            beam_scores=new_scores,
            prompt_len=state.prompt_len,
            step=state.step + 1,
            finished=finished,
        )

    def decode_beam_search(self, state: BeamSearchState) -> str:
        """Decodes the current best beam's generated tokens. Called both
        mid-generation (to snapshot a human-readable partial answer
        alongside the raw state) and at completion."""
        best_idx = int(state.beam_scores.argmax())
        generated = state.input_ids[best_idx, state.prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def capabilities(self) -> WrapperCapabilities:
        return WrapperCapabilities(
            deployment_type=DeploymentType.LOCAL_IN_PROCESS,
            # The KV cache lives in this same process's memory for the
            # lifetime of self.model -- there's no cross-request reuse
            # concept to expose since nothing is reprocessed for a
            # repeated turn within one generate() call.
            supports_kv_cache=False,
            supports_embeddings=False,
            supports_streaming=False,
            billed_per_token=False,
            # Direct logits access via self.model(...) -- the only core
            # in this project that can offer this.
            supports_logits_snapshot=True,
        )
