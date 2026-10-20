"""The two ways a build talks to the model: a plain chat client, and Pal. Both count every token and second the same way.

PlainClient   a chat that resends the whole conversation every step (what any simple chat client does).
PalClient  the real Pal pipeline: input savers, paged and compressed history, document excerpts, output rules.
Tokens are the wrapper's own counts when the model reports them, else an estimate (characters / 4), the same for both.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, LLMWrapper, WrapperCapabilities
from pal.tokens import estimate_tokens


@dataclass
class Reply:
    text: str
    prompt_tokens: int = 0         # read by the main model
    completion_tokens: int = 0     # written by the main model
    helper_tokens: int = 0         # Pal's small jobs (trimming, checking): an extra cost, shown separately
    seconds: float = 0.0
    calls: int = 0
    saved_by: dict = field(default_factory=dict)    # where Pal saved tokens on this request: stage -> tokens (estimates)
    route: str = "api"                               # "api" (the main model answered), "local" (the small local model did) or "tool"
    cost_usd: float = 0.0                            # what the request cost, when the provider's price is known (0 for local models)


@dataclass
class Counts:
    prompt: int = 0
    completion: int = 0
    calls: int = 0
    seconds: float = 0.0

    def since(self, before: Counts) -> Counts:
        return Counts(self.prompt - before.prompt, self.completion - before.completion, self.calls - before.calls, self.seconds - before.seconds)


class CountingWrapper(LLMWrapper):
    """Wraps a model and counts what goes through it. Everything else is passed on untouched."""
    name = "counting"

    def __init__(self, inner: LLMWrapper, shown: bool = False):
        super().__init__(inner.config)
        self.inner = inner
        self.shown = shown          # the main model's reply streams to the screen; the helper's never does
        self.prompt = self.completion = self.calls = 0
        self.estimated = 0          # calls whose token counts were estimated (the model did not report them)
        self.seconds = 0.0

    def chat(self, syscall: LLMSyscall) -> LLMResponse:
        from pal import steps
        if self.shown and steps.wants_tokens():        # the screen shows the main model's reply as it is written
            out: list[str] = []
            t0 = time.perf_counter()
            steps.check_stop()
            for piece in self.inner.stream_chat(syscall):
                steps.check_stop()
                out.append(piece)
                steps.token(piece)
            text = "".join(out)
            self.seconds += time.perf_counter() - t0
            p, c = sum(estimate_tokens(m.get("content", "")) for m in syscall.messages), estimate_tokens(text)
            self.prompt, self.completion, self.calls, self.estimated = self.prompt + p, self.completion + c, self.calls + 1, self.estimated + 1
            return LLMResponse(text=text, prompt_tokens=p, completion_tokens=c, raw=None, meta={"streamed": True})
        t0 = time.perf_counter()
        steps.check_stop()
        r = self.inner.chat(syscall)
        steps.check_stop()
        self.seconds += time.perf_counter() - t0
        self.prompt += r.prompt_tokens or sum(estimate_tokens(m.get("content", "")) for m in syscall.messages)
        self.completion += r.completion_tokens or estimate_tokens(r.text)
        self.calls += 1
        self.estimated += 0 if (r.prompt_tokens and r.completion_tokens) else 1
        return r

    def stream_chat(self, syscall: LLMSyscall):
        t0 = time.perf_counter()
        out = []
        for piece in self.inner.stream_chat(syscall):
            out.append(piece)
            yield piece
        self.seconds += time.perf_counter() - t0
        self.prompt += sum(estimate_tokens(m.get("content", "")) for m in syscall.messages)
        self.completion += estimate_tokens("".join(out))
        self.calls += 1

    def token_count(self, text: str) -> int:
        return self.inner.token_count(text)

    def capabilities(self) -> WrapperCapabilities:
        return self.inner.capabilities()

    def snapshot(self) -> Counts:
        return Counts(self.prompt, self.completion, self.calls, self.seconds)


class PlainClient:
    label = "Plain chat client"

    def __init__(self, core: LLMWrapper, system: str, max_tokens: int = 3000):
        self.core, self.max_tokens = CountingWrapper(core, shown=True), max_tokens
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def ask(self, message: str, references: list[dict] | None = None, query: str | None = None) -> Reply:
        if references:      # a plain client has no document index: it pastes the files into the message, and so resends them every step
            message += "\n\nReference files from the person:\n" + "\n\n".join(f"### {r['name']}\n{r['text']}" for r in references)
        self.messages.append({"role": "user", "content": message})
        before = self.core.snapshot()
        text = self.core.chat(LLMSyscall(messages=list(self.messages), temperature=0.0, max_tokens=self.max_tokens)).text
        self.messages.append({"role": "assistant", "content": text})
        d = self.core.snapshot().since(before)
        return Reply(text, d.prompt, d.completion, 0, d.seconds, d.calls)

    def close(self) -> None:
        return None


class PalClient:
    label = "Pal"

    def __init__(self, core: LLMWrapper, helper: LLMWrapper | None = None, allow_tools: frozenset | None = frozenset(), network: bool = False,
                 hot_turns: int | None = None, home: Path | None = None, model_name: str = "model", max_tokens: int = 3000, local: bool = True, project=None):
        """`local=False` (a computer without a GPU): no local model at all, so routing, input trimming and checking are off and the message
        goes to the API model as typed; paging and document excerpts, which need no model, still save tokens."""
        from pal.chat import ChatService
        from pal.config import load_settings
        from pal.knowledge_graph import KnowledgeGraph

        self.main, self.helper = CountingWrapper(core, shown=True), CountingWrapper(helper or core)
        self.home = Path(home) if home else Path(tempfile.mkdtemp(prefix="pal-"))
        self.allow, self.model_name, self.max_tokens = allow_tools, model_name, max_tokens
        context = {"hot_turns": hot_turns} if hot_turns else {}
        settings = load_settings(env={}, overrides={"storage_dir": str(self.home), "knowledge_graph": {"dir": str(self.home / "graph")}, "memory": {"enabled": False},
                                                     "tools": {"enabled": True, "network": bool(network)}, "context": context, "server": {"session_ttl_minutes": 240},
                                                     **({} if local else {"local_llm": {"enabled": False}})})
        graph = project.graph if project is not None else KnowledgeGraph(settings.knowledge_graph)
        self.svc = ChatService(settings, local_core=self.helper, graph=graph)
        if project is not None:                     # the folder's documents and project map: passages that match each request are added
            self.svc.preload_docs = project.documents()
        self.role_cores: dict[str, CountingWrapper] = {}     # agent -> its own model (/model plan <model>); the others use the main model
        self.role: str | None = None                          # the agent making the current request
        self.svc.api_core = lambda provider, model, api_key=None: self.core_for(self.role)  # type: ignore[method-assign]
        self.sid = "pal-" + self.home.name[-8:].replace("_", "-").lower()

    def core_for(self, role: str | None) -> CountingWrapper:
        return self.role_cores.get(role or "", self.main)

    def set_role_core(self, role: str, core: LLMWrapper | None) -> None:
        """Give one agent its own model (None: back to the main model). The conversation stays the same."""
        if core is None:
            self.role_cores.pop(role, None)
        else:
            self.role_cores[role] = CountingWrapper(core, shown=True)

    def ask(self, message: str, references: list[dict] | None = None, query: str | None = None) -> Reply:
        core = self.core_for(self.role)
        before_main, before_helper = core.snapshot(), self.helper.snapshot()
        attachments = [{"name": r["name"], "text": r["text"], "kind": "document"} for r in (references or [])]
        res = self.svc.send(self.sid, "local", self.model_name, None, message, attachments, route="api", allow_tools=self.allow, query=query)
        m, h = core.snapshot().since(before_main), self.helper.snapshot().since(before_helper)
        return Reply(res.reply, m.prompt, m.completion, h.prompt + h.completion, m.seconds + h.seconds, m.calls, self._saved_by(), res.handled_by,
                     float(getattr(res, "cost_usd", 0.0) or 0.0))

    def add_documents(self, docs: list[tuple[str, str]]) -> None:
        """Index documents into this conversation now (and into any later one)."""
        self.svc.preload_docs = list(docs)
        sess = self.svc.sessions.get_or_create(self.sid)
        if sess.docs is not None:
            for name, text in docs:
                sess.docs.add(name, text)

    def _saved_by(self) -> dict:
        """The savings the engine recorded for the turn that just ran, by stage. Every number is an estimate (characters / 4)."""
        _, turns = self.svc.metrics.collector.snapshot()
        if not turns:
            return {}
        t = turns[-1]
        compressed = min(max(t.compression_saved, 0), max(t.saved_paging, 0))      # history savings already include what compression saved
        found = {"input": t.saved_savers, "history": t.saved_paging - compressed, "compression": compressed, "documents": t.saved_documents, "code": t.saved_graph}
        return {k: int(v) for k, v in found.items() if v > 0}

    def close(self) -> None:
        self.svc.close()
