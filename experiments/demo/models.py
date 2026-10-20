"""The models the experiment talks to, and the recorder that counts every call.

REAL mode: the local models run in Ollama and are reached through LiteLLM, like everything else in Pal. The seconds are measured
and the token counts come from the server.

SIMULATED mode (used when Ollama is not available): the models are scripted stand-ins with a stated speed. The prompts that Pal
builds, and so the tokens it sends, are still real; the seconds are computed from the speed below and are NOT a measurement of any
machine; the answers are scripted, so answer quality is not measured at all.
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from experiments.demo.scenario import GREETINGS, SIGNOFFS
from pal.llm_wrapper.components.base import WrapperCapabilities, DeploymentType, LLMWrapper, LLMResponse, LLMSyscall
from pal.tokens import estimate_tokens

# Simulated speed, tokens per second: (reading the prompt, writing the answer). Typical for a laptop: about a 3B model for the main
# one and a 1B model for the helper. These are assumptions, shown in the report. Change them here to model other hardware.
SIM_SPEED = {"main": (450.0, 22.0), "helper": (1400.0, 70.0)}
SIM_CALL_OVERHEAD = 0.05            # seconds per model call
DEFAULT_MAIN, DEFAULT_HELPER = "llama3.2:3b", "llama3.2:1b"


@dataclass
class Call:
    turn: int
    role: str                        # "main" or "helper"
    model: str
    prompt_tokens: int               # counted from the request text, the same way for every call and every arm
    completion_tokens: int           # counted from the reply text
    seconds: float                   # measured (real) or computed from SIM_SPEED (simulated)
    wall: float = 0.0                # the real time the call took on this machine (tiny for a simulated model)
    server_prompt: int | None = None       # as reported by the server, when there is one (it may leave out a cached prefix)
    server_completion: int | None = None
    truncated: bool = False          # the prompt filled the model's context window: the answer was made from a cut-off prompt


@dataclass
class Recorder:
    calls: list = field(default_factory=list)
    turn: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, call: Call) -> None:
        with self._lock:
            self.calls.append(call)

    def of_turn(self, turn: int, role: str | None = None) -> list:
        return [c for c in self.calls if c.turn == turn and (role is None or c.role == role)]


def count_messages(messages: list) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


class RecordingCore(LLMWrapper):
    """Wraps a core and records every call: tokens, seconds, and what the server said."""

    name = "recording"

    def __init__(self, inner: LLMWrapper, role: str, model: str, recorder: Recorder, num_ctx: int = 0, max_output: int = 0):
        super().__init__(None)
        self.inner, self.role, self.model, self.rec, self.num_ctx = inner, role, model, recorder, num_ctx
        self.max_output = max_output           # the same cap for both arms, so one arm cannot run longer just because it was allowed to

    def _record(self, messages, text, wall, resp: LLMResponse | None) -> None:
        simulated = isinstance(self.inner, SimModel)
        sp = None if simulated else (getattr(resp, "prompt_tokens", 0) or None)
        sc = None if simulated else (getattr(resp, "completion_tokens", 0) or None)
        seconds = self.inner.last_seconds if simulated else wall
        self.rec.add(Call(self.rec.turn, self.role, self.model, count_messages(messages), estimate_tokens(text), seconds, wall, sp, sc,
                          bool(self.num_ctx and sp and sp >= 0.98 * self.num_ctx)))

    def _cap(self, syscall: LLMSyscall) -> None:
        if self.max_output and self.role == "main":
            syscall.max_tokens = min(syscall.max_tokens or self.max_output, self.max_output)

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        self._cap(syscall)
        t0 = time.perf_counter()
        resp = self.inner.chat(syscall)
        self._record(syscall.messages, resp.text, time.perf_counter() - t0, resp)
        return resp

    def stream_chat(self, syscall: LLMSyscall) -> Iterator[str]:
        self._cap(syscall)
        t0, pieces = time.perf_counter(), []
        try:
            for piece in self.inner.stream_chat(syscall):
                pieces.append(piece)
                yield piece
        finally:
            if pieces:
                self._record(syscall.messages, "".join(pieces), time.perf_counter() - t0, None)

    def token_count(self, text: str) -> int:
        return self.inner.token_count(text)

    def capabilities(self) -> WrapperCapabilities:
        return self.inner.capabilities()

    def price_per_token(self):
        return None


# -- simulated models --------------------------------------------------------------------------------------------------------------------
@dataclass
class SimScript:
    """What the simulated models answer for the turn being run. Set by the runner before each turn, for both arms."""
    answer: str = ""
    short_answer: str = ""
    route: str = "api"


def trim_filler(text: str) -> str:
    """What the simulated helper does when it rewrites a message: drops the greeting and the thanks. The real helper is a language model."""
    for g in GREETINGS:
        if text.startswith(g):
            text = text[len(g):]
            text = text[:1].upper() + text[1:]
    for s in SIGNOFFS:
        if text.endswith(s):
            text = text[:-len(s)]
    return text.strip()


class SimModel(LLMWrapper):
    """A scripted model with a stated speed. It reads the prompt like the real helper's prompts are read: it routes, answers easy
    questions, rewrites a message and audits the rewrite; any other request is a question for the main model."""

    name = "simulated"

    def __init__(self, role: str, script: SimScript):
        super().__init__(None)
        self.role, self.script, self.last_seconds = role, script, 0.0

    def _reply(self, messages: list) -> str:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"]
        if system.startswith("You route"):
            return f'{{"route": "{self.script.route}", "reason": "scripted"}}'
        if system.startswith("You audit"):
            return '{"ok": true, "lost": []}'
        if system.startswith("You answer"):
            return self.script.short_answer or self.script.answer
        m = re.search(r"<text>\n(.*)\n</text>", user, re.S)
        if m and self.role == "helper":
            return f"<text>\n{trim_filler(m.group(1))}\n</text>"
        return self.script.answer

    def _seconds(self, prompt_tokens: int, completion_tokens: int) -> float:
        prefill, decode = SIM_SPEED[self.role]
        return SIM_CALL_OVERHEAD + prompt_tokens / prefill + completion_tokens / decode

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        text = self._reply(syscall.messages)
        p, c = count_messages(syscall.messages), estimate_tokens(text)
        self.last_seconds = self._seconds(p, c)
        return LLMResponse(text=text, prompt_tokens=p, completion_tokens=c, meta={"sim_seconds": self.last_seconds})

    def stream_chat(self, syscall: LLMSyscall) -> Iterator[str]:
        text = self._reply(syscall.messages)
        self.last_seconds = self._seconds(count_messages(syscall.messages), estimate_tokens(text))
        yield from re.findall(r"\S+\s*", text)

    def token_count(self, text: str) -> int:
        return estimate_tokens(text)

    def capabilities(self) -> WrapperCapabilities:
        return WrapperCapabilities(deployment_type=DeploymentType.LOCAL_SERVER)


# -- real models (Ollama through LiteLLM) ---------------------------------------------------------------------------------------------------
@dataclass
class Probe:
    ok: bool
    missing: list = field(default_factory=list)
    error: str = ""
    available: list = field(default_factory=list)       # the models this Ollama really has, as it names them


def probe_ollama(url: str, wanted: list, timeout: float = 3.0) -> Probe:
    """Is Ollama answering at `url`, and are these models pulled?"""
    import requests
    try:
        r = requests.get(url.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
        have = {m["name"] for m in r.json().get("models", [])}
    except requests.ConnectionError:
        return Probe(False, error="could not connect, is `ollama serve` running?")
    except requests.Timeout:
        return Probe(False, error="timed out")
    except Exception as e:  # noqa: BLE001  (anything else that goes wrong also means: not usable)
        return Probe(False, error=f"{type(e).__name__}: {str(e)[:100]}")
    installed = sorted(have)
    have |= {n.split(":")[0] for n in have if n.endswith(":latest")}
    missing = [m for m in wanted if m not in have and f"{m}:latest" not in have]
    return Probe(not missing, missing, available=installed)


def missing_model_message(probe: Probe, model: str, command: str) -> str:
    """What to tell someone whose Ollama does not have the model: what it does have, and the two ways forward."""
    have = ", ".join(probe.available) if probe.available else "none yet"
    use = (f"or use one you already have, for example:\n  {command} --model {probe.available[0]}\n" if probe.available else "")
    return (f"The model {model} is not installed in your Ollama.\nInstalled models: {have}\n"
            f"To use the default, pull it:\n  ollama pull {model}\n{use}"
            "(`ollama list` shows what you have. The helper model is optional: if it is missing, the main model does its work too.)")


def build_real_core(model: str, url: str, num_ctx: int, seed: int):
    from pal.llm_wrapper import WrapperConfig, build_wrapper
    return build_wrapper(WrapperConfig(backend="ollama", model=model, host=url, extra={
        "timeout": 900, "litellm_params": {"num_ctx": num_ctx, "keep_alive": "30m", "seed": seed}}))


def warm_up(core: LLMWrapper) -> None:
    """Load the model before anything is timed, so loading it is not charged to whichever arm runs first."""
    core.chat(LLMSyscall(messages=[{"role": "user", "content": "Reply with the single word: ready"}], temperature=0, max_tokens=4))
