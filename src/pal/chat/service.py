"""One chat turn, end to end.

    message -> [direct answer: arithmetic / time, no model]
            -> [route: local LLM answers easy tasks itself]
            -> API path: reduce input (local LLM) -> attach documents / project graph excerpts / memory
               -> page history (last turns + relevant older ones) -> API model (+ read-only tool loop)
            -> record history, memory, metrics

Everything the chat touches goes through the AgentSDK (chat, memory, storage, tools), which sits on
the storage, memory and tool managers. Extra context (documents, graph excerpts, memory) is added to the
request but never stored in the history, so it is not resent on later turns.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from pal.access_manager.pipeline import AccessManagerPipeline
from pal.chat.providers import build_api_core
from pal.chat.services import selected
from pal.chat.sessions import Session, SessionStore, validate_session_id
from pal.chat.tools import ToolBox, direct_answer, strip_tool_tags
from pal.config import Settings, load_settings
from pal.context_manager import Turn, TurnPager
from pal.context_manager.chat_context import ChatContext
from pal.documents import DocumentIndex
from pal.errors import DependencyMissingError, PalError, ProviderError, UserInputError
from pal.input_savers.components.base import SaverResult
from pal.input_savers.pipeline import InputSaverPipeline
from pal.knowledge_graph import GRAPH_NOTICE, KnowledgeGraph
from pal.llm_scheduler import LLMScheduler, TaskClass, TurnStats, current_stats, is_rate_limit
from pal.llm_wrapper.components.base import LLMResponse, LLMSyscall, UnavailableWrapper, WrapperConfig
from pal.llm_wrapper.components.openrouter_wrapper import fetch_models, search_models
from pal.llm_wrapper.utils.registry import build_wrapper
from pal.memory.pipeline import MemoryPipeline
from pal.memory_tracker import dir_bytes
from pal.metrics import MetricsPipeline
from pal.metrics.components.collector import TurnMetric
from pal.notes_manager import NotesManager
from pal.os_kernel import OSKernel
from pal.paths import data_dir, safe_id
from pal.redact import redact
from pal.sdk.agent import AgentSpec, NativeAgent, agent_key, get_spec
from pal.sdk.pipeline import AgentSDK
from pal.storage_manager.pipeline import StorageManagerPipeline
from pal.task_router import API_NOTICE, LOCAL_TAG, RouteDecision, TaskRouter
from pal.tokens import estimate_tokens, make_counter
from pal.tool_manager.pipeline import ToolManagerPipeline

log = logging.getLogger(__name__)

DOCS_NOTICE = "Attached files are indexed. A request may end with <documents> holding the parts relevant to it."
PAGING_NOTICE = "Older turns may be omitted; the relevant ones are restored."
COMPRESS_NOTICE = ("Some older assistant replies are shown compressed ([compressed]): exact numbers, names, paths and code signatures "
                   "are kept; old code bodies and replaced drafts are left out. A <facts> block, when present, lists exact details "
                   "from older turns that are not shown.")
_PROJECT_KIND = "project"


@dataclass
class TurnResult:
    reply: str
    handled_by: str                     # "api" | "local" | "tool"
    route_reason: str = ""
    sent: str = ""                      # the last user message the API model received
    input_tokens: int = 0
    output_tokens: int = 0
    saver: object = None
    graph: Optional[dict] = None
    documents: Optional[dict] = None
    memory: Optional[dict] = None
    context: Optional[dict] = None
    tools: list = field(default_factory=list)
    fallback: Optional[dict] = None     # {"provider", "model", "reason"} when the fallback model answered
    cost_usd: float = 0.0
    files: list = field(default_factory=list)       # files an agent saved to its workspace this turn
    local_tokens: Optional[dict] = None
    latency: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def is_fallback_error(e: BaseException) -> bool:
    """Rate limits, quota, timeouts, connection failures and server errors: the kinds a second model can fix.
    A bad key, a bad request or an unknown model would fail the same way on any provider, so they are not retried elsewhere."""
    if is_rate_limit(e) or isinstance(e, (TimeoutError, ConnectionError)):
        return True
    for attr in ("status_code", "code"):
        code = getattr(e, attr, None)
        if isinstance(code, int) and code >= 500:
            return True
    return any(k in type(e).__name__.lower() for k in ("timeout", "connection", "serviceunavailable", "internalserver"))


class ChatService:
    def __init__(self, settings: Optional[Settings] = None, local_core=None, verifier_core=None, graph=None,
                 storage: Optional[StorageManagerPipeline] = None, metrics: Optional[MetricsPipeline] = None,
                 agents: Optional[dict[str, type[NativeAgent]]] = None):
        self.settings = settings or load_settings()
        if agents is None:
            from pal.agents import AGENTS  # the application layer is plugged in here, at composition time
            agents = AGENTS
        self.agents = agents
        s = self.settings
        self.scheduler = LLMScheduler(s.scheduler)   # API: first come first served per key. Local: preemptive priority.
        self._local = self.scheduler.wrap_local(local_core)
        self._verifier = self.scheduler.wrap_local(verifier_core)
        self.metrics = metrics or MetricsPipeline(max_records=s.metrics.max_records, enabled=s.metrics.enabled)
        self.graph = graph or KnowledgeGraph(s.knowledge_graph)
        self.preload_docs: list[tuple[str, str]] = []    # (name, markdown) indexed into every new session: the project folder's documents and map
        home = Path(s.storage_dir).expanduser() if s.storage_dir else data_dir()
        self.storage = storage or StorageManagerPipeline(root=str(home / "storage"))
        self.access = AccessManagerPipeline()
        self.os_kernel = OSKernel(s.os_kernel, access=self.access, root=home / "workspaces")
        selected(s.tools.services)                 # an unknown service name stops the start-up, with the list to choose from
        self._memory_pipeline = MemoryPipeline(
            store_path=None, block_limit=s.memory.block_limit, agent_id=f"profile-{safe_id(s.memory.profile)}",
            storage_manager=self.storage)
        self._memory_sdk = self._new_sdk(f"profile-{safe_id(s.memory.profile)}", ToolManagerPipeline())
        self.notes = NotesManager(self._memory_sdk, s.memory, scheduler=self.scheduler)
        self.pager = TurnPager(s.context.hot_turns, s.context.recall_turns, s.context.recall_threshold,
                               s.context.max_history_tokens)
        self.context = ChatContext(s.context, self.pager, self.notes, self.graph, tokens=self.metrics.tokens, memory_tracker=self.metrics.memory)
        self.sessions = SessionStore(self._new_session, s.server.max_sessions, s.server.session_ttl_minutes * 60)
        mt = self.metrics.memory                       # what the Memory Tracker can report on
        mt.register("notes", self.notes.stats)
        mt.register("sessions", lambda: {"open": len(self.sessions), "max": s.server.max_sessions})
        mt.register("knowledge_graph", self.graph.stats)
        mt.register("storage", lambda: {"bytes": dir_bytes(home / "storage")})
        try:  # projects left behind by a crash or a failed request
            self.graph.prune_stale(s.server.session_ttl_minutes * 60)
            self.os_kernel.prune_stale("chat-", s.server.session_ttl_minutes * 60)
        except Exception:  # noqa: BLE001  (housekeeping must never stop startup)
            log.exception("could not prune stale graph projects")

    # -- construction helpers -----------------------------------------------------------------
    def _new_sdk(self, agent_id: str, tools) -> AgentSDK:
        return AgentSDK(agent_id=agent_id, llm_wrapper=None, storage_manager=self.storage, tool_manager=tools,
                        access_manager=self.access, memory=self._memory_pipeline, os_kernel=self.os_kernel)

    def _new_session(self, session_id: str) -> Session:
        s = self.settings
        agent_id = f"chat-{safe_id(session_id)}"
        box = ToolBox(agent_id, s.tools, estimate_tokens, session_id=session_id, os_kernel=self.os_kernel, env=os.environ)
        sdk = self._new_sdk(agent_id, box.manager)
        docs = DocumentIndex(sdk, s.documents, estimate_tokens) if s.documents.enabled else None
        if docs is not None:
            for name, text in self.preload_docs:
                try:
                    docs.add(name, text)
                except Exception:  # noqa: BLE001  (one unreadable document never stops a session)
                    log.exception("could not index %s", name)
        box.bind(docs=docs, graph=self.graph if self.graph.enabled else None)
        box.code_run = self.scheduler.code.run          # converting and graph building run in the code slots, beside the LLM lines
        sess = Session(id=session_id, agent_id=agent_id, sdk=sdk, docs=docs, toolbox=box)
        sess.on_close = self._close_session
        return sess

    def _close_session(self, sess: Session) -> None:
        sess.sdk.close()
        self.graph.reset(sess.id)
        self.os_kernel.drop_workspace(sess.agent_id)       # uploaded and generated files do not outlive the chat

    # -- cores --------------------------------------------------------------------------------------
    @property
    def local_off(self) -> bool:
        """True on a computer without a GPU (local_llm.enabled: false): no local model at all, so routing, input trimming, checking and
        summaries by the local model are off and every request goes to the API model as typed. Savers that need no model (paging,
        document and code excerpts, memory) still work."""
        return self.settings.local_llm.get("enabled", True) is False

    def _build_local(self, cfg: dict):
        cfg = {k: v for k, v in cfg.items() if k != "enabled"}
        try:
            return self.scheduler.wrap_local(build_wrapper(WrapperConfig(**cfg)))
        except DependencyMissingError as e:                 # fail safe: everything that asks the local model falls back to the API
            log.warning("the local model cannot be set up: %s", e)
            return self.scheduler.wrap_local(UnavailableWrapper(str(e)))

    def local_core(self):
        if self._local is None:
            self._local = self._build_local(self.settings.local_llm)
        return self._local

    def verifier_core(self):
        if self._verifier is None:
            v = self.settings.verifier_llm
            self._verifier = self._build_local(v) if v else self.local_core()
        return self._verifier

    def api_core(self, provider: str, model: str, api_key: Optional[str] = None):
        return build_api_core(provider, model, api_key)

    def model_catalogue(self, text: str = "", free_only: bool = False, limit: int = 50) -> list[dict]:
        """OpenRouter's models for the Explore mode: search by words in the name, optionally only free ones. `self.catalogue_fetch` replaces the
        network read (for tests)."""
        try:
            models = fetch_models(getattr(self, "catalogue_fetch", None))
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"could not read OpenRouter's model list ({type(e).__name__}); you can still type a model name yourself") from e
        return [{**asdict(m), "label": m.label()} for m in search_models(models, text, free_only, limit)]

    def router(self) -> TaskRouter:
        cfg = self.settings.task_routing
        if self.local_off:                                  # no local model: nothing is answered locally, everything goes to the API model
            import copy
            off = copy.copy(TaskRouter(None, cfg).cfg)
            off.enabled = False
            return TaskRouter(None, off)
        return TaskRouter(self.local_core() if cfg.enabled else None, cfg)

    # -- prompt pieces ---------------------------------------------------------------------------------
    def output_config(self) -> str:
        return self.settings.read_text_file(self.settings.output_config)

    def system_prompt(self, with_output_config: bool = True, local_turns: bool = False, graph_project: bool = False,
                      documents: bool = False, paged: bool = False, tools: str = "", agent_role: str = "",
                      compressed: bool = False) -> str:
        prompt = self.settings.base_system_prompt
        if agent_role:
            prompt += f"\n\n{agent_role}"
        cfg = self.output_config() if with_output_config else ""
        if cfg:
            prompt += f"\n\n{cfg}"
        for on, text in ((local_turns, API_NOTICE), (graph_project, GRAPH_NOTICE), (documents, DOCS_NOTICE),
                         (paged, PAGING_NOTICE), (compressed, COMPRESS_NOTICE), (bool(tools), tools)):
            if on:
                prompt += f"\n\n{text}"
        return prompt

    def saver(self, counter) -> InputSaverPipeline:
        rules = self.settings.read_text_file(self.settings.input_rules) or None
        return InputSaverPipeline(self.local_core(), self.verifier_core(), counter, rules_text=rules)

    def reduce(self, counter, text: str, mode: Optional[str], methods: Optional[list], pasted: list):
        """Every chatbox input goes through the local LLM here, before any API call."""
        if self.local_off:                                  # no local model: the message goes as typed
            n = counter(text) if callable(counter) else estimate_tokens(text)
            return SaverResult(text, text, [], n, n, "off")
        mode = mode or self.settings.input_mode
        saver = self.saver(counter)
        if mode == "steps":
            return saver.run(text, self.enabled_methods(methods), pasted)
        return saver.run_rules(text, pasted)

    def enabled_methods(self, requested: Optional[list]) -> list:
        return [m for m in self.settings.input_methods if requested is None or m in requested]

    # -- one turn ----------------------------------------------------------------------------------------
    def send(self, session_id: str, provider: str, model: str, api_key: Optional[str], message: str,
             attachments: list[dict], methods: Optional[list] = None, mode: Optional[str] = None,
             pasted: Optional[list] = None, route: Optional[str] = None, ephemeral: bool = False,
             agent: Optional[str] = None, allow_tools: Optional[frozenset] = None, query: Optional[str] = None) -> TurnResult:
        """`ephemeral` turns (the demo) read and write no memory and are not added to the metrics.
        `agent` runs the turn as that agent: Gemini with the agent's own key, API route, its role and tools.
        `allow_tools` limits the turn to those tools (an empty set means none); None leaves every enabled tool available.
        `query` is what to look up memory, document passages and code excerpts with, when the message carries more than the person's own words
        (the terminal adds the folder listing and the agent's instructions); None uses the message."""
        spec = self._resolve_agent(agent)
        if spec is not None:
            provider, model, route = "gemini", self.settings.agents.model, "api"
            api_key = agent_key(spec)
        sid = validate_session_id(session_id or uuid.uuid4().hex)
        if not (message or "").strip() and not attachments:
            raise UserInputError("empty message")
        sess = self.sessions.get_or_create(sid)
        m = TurnMetric(route="api", session=sid[:8], provider=provider, model=model)
        start = time.perf_counter()
        cls = TaskClass.AGENT if spec is not None else TaskClass.CHAT
        with sess.lock, self.scheduler.task(cls), self.scheduler.track() as stats:
            try:
                res = self._turn(sess, provider, model, api_key, message, attachments, methods, mode, pasted or [],
                                 route, m, ephemeral, spec, allow_tools, query)
            except Exception as e:
                m.error = type(e).__name__
                m.latency_total = time.perf_counter() - start
                self.metrics.latency.finish(m)
                self._apply_stats(m, stats)
                if not ephemeral:
                    self.metrics.record_turn(m)
                raise
        self._apply_stats(m, stats)
        self.notes.after_turn(ephemeral)                     # background: merging duplicate notes (OTHER class)
        m.route = res.handled_by
        m.latency_total = time.perf_counter() - start
        self.metrics.latency.finish(m)
        res.latency = {"total": round(m.latency_total, 3), "local": round(m.latency_local, 3),
                       "api": round(m.latency_api, 3), "tools": round(m.latency_tools, 3)}
        if not ephemeral:
            self.metrics.record_turn(m)
        return res

    def _summarize_text(self, text: str) -> str:
        """For the summarize_text tool: a short summary of `text` by the local model."""
        core = self.local_core()
        system = "Summarise the text in at most 150 words. Plain text, no preface."
        return core.chat(LLMSyscall(messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
                                    temperature=0.0, max_tokens=300)).text

    @staticmethod
    def _apply_stats(m: TurnMetric, stats: TurnStats) -> None:
        m.wait_local, m.wait_api, m.local_pauses = round(stats.wait_local, 4), round(stats.wait_api, 4), stats.local_pauses

    def _resolve_agent(self, agent: Optional[str]) -> Optional[AgentSpec]:
        if not agent:
            return None
        if not self.settings.agents.enabled:
            raise UserInputError("agents are turned off in the settings")
        spec = get_spec(self.agents, agent)
        if not agent_key(spec):
            raise UserInputError(f"the {spec.name} needs a Gemini API key: set {spec.key_env} in your .env file")
        return spec

    def _remember(self, sid: str, message: str, ephemeral: bool = False) -> None:
        if ephemeral:
            return
        self.notes.remember(sid, message)

    def _turn(self, sess: Session, provider, model, api_key, message, attachments, methods, mode, pasted, route,
              m: TurnMetric, ephemeral: bool = False, spec: Optional[AgentSpec] = None,
              allow_tools: Optional[frozenset] = None, query: Optional[str] = None) -> TurnResult:
        s = self.settings
        sid = sess.id
        sess.toolbox.allow = spec.tools if spec else allow_tools
        sess.toolbox.saved_files = []
        sess.toolbox.summarizer = None if self.local_off else self._summarize_text      # the local model, in this turn's priority class
        history_msgs = [x for t in sess.turns for x in (
            {"role": "user", "content": t.user},
            {"role": "assistant", "content": (f"{LOCAL_TAG} " if t.local else "") + t.assistant})]
        project = self.graph.has_project(sid)

        # 1. no model needed
        if s.tools.enabled and s.tools.direct_answers and not attachments and route != "api":
            with self.metrics.latency.stage(m, "direct"):
                direct = direct_answer(message)
            if direct is not None:
                sess.turns.append(Turn(message, direct, local=True))
                return TurnResult(reply=direct, handled_by="tool", route_reason="answered without a model")

        # 2. routing
        t0 = time.perf_counter()
        router = self.router()
        if project and route in (None, "", "auto") and router.enabled:
            decision = RouteDecision("api", "a code project is loaded", "gate")
        else:
            with self.metrics.latency.stage(m, "routing"):
                decision = router.decide(message, attachments, route, history_msgs)

        if decision.route == "local":
            with self.metrics.latency.stage(m, "local_answer"), self.scheduler.task(TaskClass.AGENT if spec is not None else TaskClass.CHAT, preemptible=True):
                ans = router.answer(message, history_msgs, attachments if decision.source == "override" else None)
            m.latency_local += time.perf_counter() - t0
            if ans.text is not None:
                sess.turns.append(Turn(message, ans.text, local=True))
                self._remember(sid, message, ephemeral)
                m.local_tokens = ans.prompt_tokens + ans.completion_tokens
                return TurnResult(reply=ans.text, handled_by="local", route_reason=decision.reason,
                                  local_tokens={"input": ans.prompt_tokens, "output": ans.completion_tokens})
            if decision.source == "override":
                raise UserInputError(f"local model could not answer ({ans.reason}); switch routing to auto or API")
            decision = RouteDecision("api", ans.reason, "escalated")
            m.escalated = True
        else:
            m.latency_local += time.perf_counter() - t0

        # 3. API path
        api = self.api_core(provider, model, api_key)
        counter = self.metrics.tokens.counter_for(s.token_counter, api)
        t0 = time.perf_counter()
        with self.metrics.latency.stage(m, "input_saver"):
            result = self.reduce(counter, message, mode, methods, pasted)
            from pal import steps
            steps.detail(f"{result.tokens_saved:,} tokens less" if result.tokens_saved > 0 else "nothing to trim")
        m.latency_local += time.perf_counter() - t0
        self.metrics.tokens.credit(m, "input_savers", result.tokens_saved)

        # attachments: project summaries and small notes go inline; documents are indexed
        file_blocks, stubs, new_docs = [], [], set()
        for a in attachments:
            note = f" | {a['note']}" if a.get("note") else ""
            if a.get("kind") in (_PROJECT_KIND, "media") or sess.docs is None:
                file_blocks.append(f"[file: {a['name']}{note}]\n{a['text']}")
                stubs.append(f"[file: {a['name']}{note}]")
                continue
            sess.docs.add(a["name"], a["text"])
            new_docs.add(a["name"])
            file_blocks.append(f"[file: {a['name']}{note}]\n{sess.docs.upload_view(a['name'], a['text'], bool(s.tools.enabled))}")
            stubs.append(sess.docs.stub(a["name"]))
        file_block = "".join(f"\n\n{b}" for b in file_blocks)
        history_user = result.reduced + "".join(f"\n{x}" for x in stubs)

        # everything besides the message itself: history (compressed by age), memory, document and code excerpts, under one budget
        with self.metrics.latency.stage(m, "context"):
            bundle = self.context.build(sess, sid, (query or "").strip() or message, new_docs, project, counter, ephemeral, m)
            from pal import steps
            found = []
            if bundle.doc_info and bundle.doc_info.get("chunks"):
                found.append(f"{bundle.doc_info['chunks']} document passage{'s' if bundle.doc_info['chunks'] != 1 else ''}")
            if bundle.graph_info and bundle.graph_info.get("tokens") and bundle.graph_info["tokens"] < bundle.graph_info.get("project_tokens", 0):
                found.append(f"code: {bundle.graph_info['tokens']:,} of {bundle.graph_info.get('project_tokens', 0):,} tokens")
            steps.detail(", ".join(found))
        turns, paged, recall = bundle.turns, bundle.paged, bundle.recall
        doc_info, graph_info = bundle.doc_info, bundle.graph_info
        extras = bundle.texts

        request_content = result.reduced + file_block + "".join(f"\n\n{x}" for x in extras)
        system = self.system_prompt(
            local_turns=any(t.local for t in turns), graph_project=project,
            documents=bool(sess.docs and sess.docs.docs), paged=bool(paged and paged.dropped_turns),
            compressed=bundle.uses_compression,
            tools=sess.toolbox.prompt(message), agent_role=spec.role if spec else "")
        messages: list[dict] = [{"role": "system", "content": system}]
        for t in turns:
            messages += [{"role": "user", "content": t.user},
                         {"role": "assistant", "content": (f"{LOCAL_TAG} " if t.local else "") + t.assistant}]
        messages.append({"role": "user", "content": request_content})

        sess.toolbox.price_fn = lambda a=api: getattr(a, "price_per_token", lambda: None)()
        key = api_key or getattr(getattr(api, "config", None), "api_key", None) or ""
        line = (f"{provider}:{hashlib.sha1(key.encode()).hexdigest()[:8]}", self.scheduler.interval_for(provider, model))  # noqa: S324
        from pal import steps
        with steps.step("model", "Thinking") as st:
            reply, in_tok, out_tok, tool_log = self._call_with_tools(sess, api, messages, m, message, spec, line)
            st.detail = f"{in_tok:,} tokens in, {out_tok:,} out" if in_tok or out_tok else ""
        sess.turns.append(Turn(history_user, reply, files=tuple(sess.toolbox.saved_files)))
        self._remember(sid, message, ephemeral)
        m.input_tokens, m.output_tokens = in_tok, out_tok
        return TurnResult(
            reply=reply, handled_by="api", route_reason=decision.reason, sent=request_content,
            input_tokens=in_tok, output_tokens=out_tok, saver=result, graph=graph_info, documents=doc_info,
            memory={"notes": recall.hits, "tokens": recall.tokens} if recall else None,
            context=self.context.describe(sess, bundle, m),
            tools=tool_log,
            fallback=({"provider": self.settings.fallback.provider, "model": self.settings.fallback.model,
                       "reason": m.fallback_reason} if m.fallback else None),
            cost_usd=m.cost_usd, files=list(sess.toolbox.saved_files))

    # -- one API call through its line, with the optional fallback -----------------------------------------------------
    def _api_once(self, sess: Session, api, messages: list[dict], spec: Optional[AgentSpec], line: tuple[str, float]):
        def api_call(msgs: list[dict] = messages) -> LLMResponse:
            return sess.sdk.chat_response(msgs, core=api, max_tokens=spec.max_tokens if spec else 2048)

        ticket = self.scheduler.api.submit(line[0], api_call, line[1], label="api call")
        return ticket.result(), ticket      # waits its turn in the line (strictly first come, first served)

    @staticmethod
    def _provider_error(api, e: BaseException) -> ProviderError:
        text = redact(str(e), getattr(getattr(api, "config", None), "api_key", None))
        if any(k in text.lower() for k in ("429", "resource_exhausted", "quota", "rate limit")):
            return ProviderError("rate limit or quota reached for this API key (free tiers allow only a few "
                                 "requests per minute and per day). Wait a minute and try again.")
        return ProviderError(f"{type(e).__name__}: {text}")

    def _fallback_for(self, e: BaseException, spec: Optional[AgentSpec]):
        """(core, line) for the fallback model if this failure is the kind it is for, else None."""
        fb = self.settings.fallback
        if not (fb.enabled and fb.model) or (spec is not None and not fb.for_agents) or not is_fallback_error(e):
            return None
        try:
            core = self.api_core(fb.provider, fb.model, None)
        except (UserInputError, PalError) as why:
            log.warning("fallback to %s/%s is not available: %s", fb.provider, fb.model, why)
            return None
        key = getattr(getattr(core, "config", None), "api_key", None) or ""
        return core, (f"{fb.provider}:{hashlib.sha1(key.encode()).hexdigest()[:8]}",  # noqa: S324
                      self.scheduler.interval_for(fb.provider, fb.model))

    def _call_with_tools(self, sess: Session, api, messages: list[dict], m: TurnMetric, message: str,
                         spec: Optional[AgentSpec] = None, line: tuple[str, float] = ("api", 0.0)):
        """API call, then up to tools.max_rounds read-only tool calls. Returns (reply, tokens in, out, log)."""
        cfg = self.settings.tools
        in_tok = out_tok = 0
        log_: list[dict] = []
        rounds = cfg.max_rounds if (cfg.enabled and sess.toolbox.available(message)) else 0
        for i in range(rounds + 1):
            try:
                resp, ticket = self._api_once(sess, api, messages, spec, line)
            except UserInputError:
                raise
            except Exception as e:
                alt = None if m.fallback else self._fallback_for(e, spec)
                if alt is None:
                    raise self._provider_error(api, e) from e
                fb = self.settings.fallback
                log.warning("answering with the fallback model %s/%s after %s", fb.provider, fb.model, type(e).__name__)
                api, line = alt
                m.fallback, m.fallback_reason = f"{fb.provider}/{fb.model}", type(e).__name__
                try:
                    resp, ticket = self._api_once(sess, api, messages, spec, line)
                except UserInputError:
                    raise
                except Exception as e2:
                    raise self._provider_error(api, e) from e2      # report the original failure, not the fallback's
            m.latency_api += ticket.run_seconds
            m.cost_usd += float((resp.meta or {}).get("cost_usd") or 0.0)
            stats = current_stats()
            if stats is not None:
                stats.add_api(ticket)
            m.api_calls += 1
            in_tok += resp.prompt_tokens or 0
            out_tok += resp.completion_tokens or 0
            call = sess.toolbox.parse(resp.text or "") if i < rounds else None
            if call is None:
                price = getattr(api, "price_per_token", lambda: None)()
                if price:                                              # dollars the savings were worth at this model's input price
                    m.saved_cost_usd = round(m.saved_total * price[0], 6)
                return strip_tool_tags(resp.text or ""), in_tok, out_tok, log_
            outcome = sess.toolbox.run(call)
            m.tool_calls += 1
            m.latency_tools += outcome.seconds
            log_.append({"name": outcome.name, "ok": outcome.ok, "seconds": round(outcome.seconds, 3)})
            messages = messages + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": f'<tool_result name="{outcome.name}">\n{outcome.text}\n</tool_result>'}]
        return "", in_tok, out_tok, log_  # unreachable: the last round never parses a call

    def local_status(self) -> dict:
        """Is the local LLM reachable and does it have the model? {"ok": True|False|None, "detail": str}"""
        local = self.settings.local_llm
        if self.local_off:
            return {"ok": None, "detail": "off: no GPU on this computer, so Pal uses API models only"}
        if local.get("backend") != "ollama":
            return {"ok": None, "detail": f"{local.get('backend')} (not checked)"}
        import requests
        host = (local.get("host") or "http://localhost:11434").rstrip("/")
        try:
            tags = requests.get(host + "/api/tags", timeout=3).json()
            names = [x["name"] for x in tags.get("models", [])]
            have = any(n == local["model"] or n.startswith(local["model"] + ":") for n in names)
            return {"ok": have, "detail": f"{local['model']} " + (
                "ready" if have else f"not pulled (ollama pull {local['model']})")}
        except Exception as e:  # noqa: BLE001  (any failure means "not reachable")
            return {"ok": False, "detail": f"Ollama unreachable: {e.__class__.__name__}"}

    # -- housekeeping ----------------------------------------------------------------------------------------
    def reset(self, session_id: str) -> None:
        self.sessions.remove(session_id)
        try:
            sid = validate_session_id(session_id)
            self.graph.reset(sid)                              # also covers a project uploaded before any turn
            self.os_kernel.drop_workspace(f"chat-{safe_id(sid)}")   # and an image or audio file uploaded before any turn
        except UserInputError:
            pass

    def clear_memory(self) -> int:
        return self.notes.clear()

    def memory_count(self) -> int:
        return self.notes.count_notes()

    def warm_up_local(self):
        """A background (OTHER class) job: load the local model so the next real job starts fast. Returns its ticket."""
        if self.local_off:
            return None
        core = self.local_core()
        inner = getattr(core, "core", core)
        def ping() -> LLMResponse:
            return inner.chat(LLMSyscall(messages=[{"role": "user", "content": "hi"}], max_tokens=1))

        if self.scheduler.local is None:
            return ping()
        return self.scheduler.local.submit(ping, TaskClass.OTHER, "warm up the local model")

    def close(self) -> None:
        self.sessions.close_all()
        self.scheduler.close()

    # -- experiment ---------------------------------------------------------------------------------------------
    def experiment(self, text: str, methods: Optional[list], provider: str = "", model: str = "",
                   api_key: Optional[str] = None, call_api: bool = False, mode: Optional[str] = None,
                   pasted: Optional[list] = None) -> dict:
        api = None
        if provider and model:
            try:
                api = self.api_core(provider, model, api_key)
            except UserInputError:
                if call_api:
                    raise
        if call_api and api is None:
            raise UserInputError("choose a provider and model to compare API replies")
        counter = make_counter(self.settings.token_counter, api)
        result = self.reduce(counter, text, mode, methods, pasted or [])
        exact = api is not None and self.settings.token_counter == "provider"  # noqa: S105
        label = f"{provider} token counter" if exact else "heuristic (chars/4)"
        out = {"saver": result, "counter": label, "api": None}
        if call_api and api is not None:
            raw = api.chat(LLMSyscall(messages=[
                {"role": "system", "content": self.system_prompt(with_output_config=False)},
                {"role": "user", "content": text}], max_tokens=2048))
            opt = api.chat(LLMSyscall(messages=[
                {"role": "system", "content": self.system_prompt(with_output_config=True)},
                {"role": "user", "content": result.reduced}], max_tokens=2048))
            out["api"] = {
                "raw": {"input_tokens": raw.prompt_tokens, "output_tokens": raw.completion_tokens, "reply": raw.text},
                "optimized": {"input_tokens": opt.prompt_tokens, "output_tokens": opt.completion_tokens, "reply": opt.text},
            }
        return out
