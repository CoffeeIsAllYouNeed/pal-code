"""Typed, validated settings.

Sources, lowest to highest priority:
  1. the packaged defaults (config/default.yaml)
  2. a user file (--config path, or the PAL_CONFIG env var)
  3. environment variables: PAL__<SECTION>__<KEY>=value  (e.g. PAL__SERVER__PORT=8080)
  4. explicit overrides passed in code

Unknown keys and bad values raise ConfigError at startup, so a typo fails loudly instead of being
silently ignored.
"""
from __future__ import annotations

import copy
import os
from dataclasses import MISSING, dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

import yaml

from pal.errors import ConfigError
from pal.paths import data_dir

_ENV_PREFIX = "PAL__"


def _fail(section: str, key: str, msg: str) -> None:
    raise ConfigError(f"config {section}.{key}: {msg}")


@dataclass
class _Section:
    """Base for one settings section: builds from a dict, rejects unknown keys, checks types."""

    name: ClassVar[str] = ""

    @classmethod
    def coerce(cls, value: Any):
        """Accept an instance, a dict, or None (= all defaults)."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ConfigError(f"config {cls.name}: expected a mapping, got {type(value).__name__}")
        known = {f.name: f for f in fields(cls)}
        for k in value:
            if k not in known:
                raise ConfigError(f"config {cls.name}.{k}: unknown key (valid: {', '.join(sorted(known))})")
        kwargs = {}
        for k, v in value.items():
            default = known[k].default if known[k].default is not MISSING else None
            kwargs[k] = _convert(cls.name, k, v, default)
        obj = cls(**kwargs)
        obj.validate()
        return obj

    def validate(self) -> None:  # overridden
        return None


def _convert(section: str, key: str, value: Any, default: Any) -> Any:
    if default is None or value is None:
        return value
    want = type(default)
    if want is bool:
        if isinstance(value, bool):
            return value
        _fail(section, key, f"expected true/false, got {value!r}")
    if want in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(section, key, f"expected a number, got {value!r}")
        return want(value)
    if want is str:
        if not isinstance(value, str):
            _fail(section, key, f"expected text, got {value!r}")
        return value
    return value


def _check(section: str, key: str, ok: bool, msg: str) -> None:
    if not ok:
        _fail(section, key, msg)


@dataclass
class TaskRoutingConfig(_Section):
    name: ClassVar[str] = "task_routing"
    enabled: bool = True
    default: str = "auto"
    max_local_chars: int = 400
    max_local_lines: int = 6
    history_messages: int = 6

    def validate(self) -> None:
        _check(self.name, "default", self.default in ("auto", "local", "api"), "must be auto, local or api")
        _check(self.name, "max_local_chars", self.max_local_chars > 0, "must be > 0")
        _check(self.name, "max_local_lines", self.max_local_lines > 0, "must be > 0")
        _check(self.name, "history_messages", self.history_messages >= 0, "must be >= 0")


@dataclass
class KnowledgeGraphConfig(_Section):
    name: ClassVar[str] = "knowledge_graph"
    enabled: bool = True
    dir: str | None = None
    budget_tokens: int = 1500
    max_slice_lines: int = 40
    min_code_file_chars: int = 8000
    max_zip_files: int = 3000
    max_zip_mb: int = 50

    def validate(self) -> None:
        for k in ("budget_tokens", "max_slice_lines", "min_code_file_chars", "max_zip_files", "max_zip_mb"):
            _check(self.name, k, getattr(self, k) > 0, "must be > 0")

    def resolved_dir(self) -> Path:
        return Path(self.dir).expanduser() if self.dir else data_dir("graphs")


@dataclass
class ContextConfig(_Section):
    """Context paging: keep the last few turns, recall older turns only when relevant."""

    name: ClassVar[str] = "context"
    enabled: bool = True
    hot_turns: int = 6
    recall_turns: int = 2
    recall_threshold: float = 0.35
    max_history_tokens: int = 6000
    total_budget_tokens: int = 10000          # one budget for history, facts, memory, document and code excerpts together
    compress: bool = True                     # compress OLD assistant replies (no model involved)
    compress_turns: int = 12                  # how many turns older than the hot window are shown in compressed form
    compress_min_words: int = 60              # a reply shorter than this is left alone
    compress_budget_tokens: int = 400         # most tokens the compressed older turns may take together
    facts_tokens: int = 250                   # budget for the exact details of turns too old to show at all
    facts_max_ratio: float = 0.3              # a facts block is only added if it is this much smaller than the turns it stands for

    def validate(self) -> None:
        _check(self.name, "total_budget_tokens", self.total_budget_tokens >= 1000, "must be >= 1000")
        _check(self.name, "compress_turns", self.compress_turns >= 0, "must be >= 0")
        _check(self.name, "compress_min_words", self.compress_min_words >= 1, "must be >= 1")
        _check(self.name, "facts_tokens", self.facts_tokens >= 0, "must be >= 0")
        _check(self.name, "compress_budget_tokens", self.compress_budget_tokens >= 0, "must be >= 0")
        _check(self.name, "facts_max_ratio", 0 <= self.facts_max_ratio <= 1, "must be between 0 and 1")
        _check(self.name, "hot_turns", self.hot_turns >= 1, "must be >= 1")
        _check(self.name, "recall_turns", self.recall_turns >= 0, "must be >= 0")
        _check(self.name, "recall_threshold", 0 <= self.recall_threshold <= 1, "must be between 0 and 1")
        _check(self.name, "max_history_tokens", self.max_history_tokens >= 500, "must be >= 500")


@dataclass
class MemoryConfig(_Section):
    """Cross-session memory. Stores what you type on this machine; see README for what that means."""

    name: ClassVar[str] = "memory"
    enabled: bool = True
    dir: str | None = None
    profile: str = "default"
    recall_notes: int = 3
    min_score: float = 0.3
    budget_tokens: int = 300
    min_chars: int = 25
    block_limit: int = 200
    tidy_every: int = 50                      # after this many new notes, merge duplicates in the background (0 = never)

    def validate(self) -> None:
        _check(self.name, "recall_notes", self.recall_notes >= 0, "must be >= 0")
        _check(self.name, "min_score", 0 <= self.min_score <= 1, "must be between 0 and 1")
        _check(self.name, "budget_tokens", self.budget_tokens > 0, "must be > 0")
        _check(self.name, "block_limit", self.block_limit >= 5, "must be >= 5")


@dataclass
class DocumentsConfig(_Section):
    """Attached documents are indexed once; later turns get only the relevant chunks."""

    name: ClassVar[str] = "documents"
    enabled: bool = True
    inline_max_tokens: int = 2500
    preview_tokens: int = 800
    chunk_tokens: int = 350
    retrieve_chunks: int = 3
    budget_tokens: int = 1200
    min_score: float = 0.15
    max_file_chars: int = 500000

    def validate(self) -> None:
        for k in ("inline_max_tokens", "preview_tokens", "chunk_tokens", "retrieve_chunks", "budget_tokens",
                  "max_file_chars"):
            _check(self.name, k, getattr(self, k) > 0, "must be > 0")
        _check(self.name, "min_score", 0 <= self.min_score <= 1, "must be between 0 and 1")


@dataclass
class ToolsConfig(_Section):
    name: ClassVar[str] = "tools"
    enabled: bool = True
    max_rounds: int = 3
    max_result_tokens: int = 800
    direct_answers: bool = True
    network: bool = False                     # wikipedia and arxiv lookups: what the model asks for leaves this machine, so it is opt-in
    convert_max_chars: int = 500_000          # file_to_markdown: the largest markdown it will produce
    save_files: bool = True                   # agents that list save_file may write NEW files to their workspace (overwrites ask first)
    services: list = field(default_factory=list)      # third-party service tools to offer, by name, or ["all"]. Off by default: a
                                                      # query leaves this machine, most need a key, and some cost money
    hf_models: dict = field(default_factory=dict)     # Hugging Face model for a media tool, e.g. {text_to_image: "owner/model"}
    rapidapi_hosts: dict = field(default_factory=dict)  # RapidAPI host for a tool, if your subscription shows a different one

    def validate(self) -> None:
        _check(self.name, "max_rounds", 0 <= self.max_rounds <= 8, "must be between 0 and 8")
        _check(self.name, "max_result_tokens", self.max_result_tokens >= 50, "must be >= 50")
        _check(self.name, "convert_max_chars", self.convert_max_chars >= 1000, "must be >= 1000")
        _check(self.name, "services", isinstance(self.services, list) and all(isinstance(x, str) for x in self.services),
               "must be a list of service names (or [all])")
        for k in ("hf_models", "rapidapi_hosts"):
            v = getattr(self, k)
            _check(self.name, k, isinstance(v, dict) and all(isinstance(a, str) and isinstance(b, str) for a, b in v.items()),
                   "must map names to text")


@dataclass
class AgentsConfig(_Section):
    """Agent application layer (coding and documentation agents on the Gemini API)."""

    name: ClassVar[str] = "agents"
    enabled: bool = True
    model: str = "gemini-3.5-flash"

    def validate(self) -> None:
        _check(self.name, "model", bool(self.model.strip()), "must not be empty")


@dataclass
class OsKernelConfig(_Section):
    """Non-LLM queries (files, network, system facts) go to the OS kernel through this guarded interface."""

    name: ClassVar[str] = "os_kernel"
    enabled: bool = True
    workspace_dir: str | None = None
    max_read_bytes: int = 1_000_000
    max_write_bytes: int = 5_000_000
    allow_private_network: bool = False       # localhost, home network and cloud metadata addresses stay blocked
    net_timeout_seconds: float = 10.0
    net_max_bytes: int = 1_000_000
    max_binary_bytes: int = 20_000_000        # images and audio, in the workspace and over the network
    allowed_hosts: list = field(default_factory=list)   # empty = any public host

    def validate(self) -> None:
        for k in ("max_read_bytes", "max_write_bytes", "net_max_bytes", "max_binary_bytes"):
            _check(self.name, k, getattr(self, k) > 0, "must be > 0")
        _check(self.name, "net_timeout_seconds", self.net_timeout_seconds > 0, "must be > 0")
        _check(self.name, "allowed_hosts", isinstance(self.allowed_hosts, list)
               and all(isinstance(h, str) for h in self.allowed_hosts), "must be a list of host names")


@dataclass
class SchedulerConfig(_Section):
    """API model: first come, first served per key. Local model: preemptive priority (agent, chat, other)."""

    name: ClassVar[str] = "scheduler"
    enabled: bool = True
    local_max_pauses: int = 2                  # a local job is paused at most this many times, then it runs to the end
    other_stale_seconds: float = 600.0         # a background job that waited this long is dropped
    code_workers: int = 2                      # file conversions and code-graph builds running at once
    max_queue: int = 100
    api_min_interval: dict = field(default_factory=lambda: {"gemini": 4.0})   # seconds between call starts, per provider
    api_retry_429: int = 2
    api_backoff_seconds: float = 5.0
    free_model_interval: float = 3.0           # seconds between calls to an OpenRouter ":free" model (about 20 per minute)

    def validate(self) -> None:
        for k in ("local_max_pauses", "code_workers", "api_retry_429"):
            _check(self.name, k, getattr(self, k) >= 0, "must be >= 0")
        _check(self.name, "max_queue", self.max_queue >= 1, "must be >= 1")
        _check(self.name, "other_stale_seconds", self.other_stale_seconds > 0, "must be > 0")
        _check(self.name, "api_backoff_seconds", self.api_backoff_seconds >= 0, "must be >= 0")
        _check(self.name, "free_model_interval", self.free_model_interval >= 0, "must be >= 0")
        ok = isinstance(self.api_min_interval, dict) and all(
            isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
            for k, v in self.api_min_interval.items())
        _check(self.name, "api_min_interval", ok, "must map provider names to seconds (>= 0)")


@dataclass
class FallbackConfig(_Section):
    """When your own provider is rate-limited or down, answer with a second model (by default a free OpenRouter model)."""

    name: ClassVar[str] = "fallback"
    enabled: bool = False
    provider: str = "openrouter"
    model: str = ""                            # e.g. an OpenRouter model ending in ":free"; empty = fallback is off
    for_agents: bool = True

    def validate(self) -> None:
        _check(self.name, "provider", bool(self.provider.strip()), "must not be empty")


@dataclass
class MetricsConfig(_Section):
    name: ClassVar[str] = "metrics"
    enabled: bool = True
    max_records: int = 1000

    def validate(self) -> None:
        _check(self.name, "max_records", self.max_records >= 10, "must be >= 10")


@dataclass
class ServerConfig(_Section):
    name: ClassVar[str] = "server"
    host: str = "127.0.0.1"
    port: int = 5000
    threads: int = 8
    max_upload_mb: int = 20
    max_sessions: int = 200
    session_ttl_minutes: int = 240
    log_level: str = "INFO"

    def validate(self) -> None:
        _check(self.name, "port", 0 < self.port < 65536, "must be 1-65535")
        _check(self.name, "threads", self.threads >= 1, "must be >= 1")
        _check(self.name, "max_upload_mb", self.max_upload_mb >= 1, "must be >= 1")
        _check(self.name, "max_sessions", self.max_sessions >= 1, "must be >= 1")
        _check(self.name, "session_ttl_minutes", self.session_ttl_minutes >= 1, "must be >= 1")
        _check(self.name, "log_level", self.log_level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR"), "bad level")


@dataclass
class StudioConfig(_Section):
    """The Studio: a bot plans a product and builds it. The context window matters: Ollama's own default is small and it silently CUTS a longer prompt, so a
    build would lose its earlier files. The values below are sent with every local model call. A bigger window needs more memory."""

    name: ClassVar[str] = "studio"
    num_ctx: int = 16384                      # the builder's context window, in tokens
    planner_num_ctx: int = 8192               # the bot's (its prompt is the list of agents, tools and permissions)
    max_runs: int = 6                         # finished builds kept on disk (older ones are deleted)

    def validate(self) -> None:
        _check(self.name, "num_ctx", self.num_ctx >= 2048, "must be >= 2048")
        _check(self.name, "planner_num_ctx", self.planner_num_ctx >= 2048, "must be >= 2048")
        _check(self.name, "max_runs", self.max_runs >= 1, "must be >= 1")


_SECTIONS: dict[str, type[_Section]] = {
    "task_routing": TaskRoutingConfig, "knowledge_graph": KnowledgeGraphConfig, "context": ContextConfig,
    "memory": MemoryConfig, "documents": DocumentsConfig, "tools": ToolsConfig,
    "agents": AgentsConfig, "studio": StudioConfig, "scheduler": SchedulerConfig, "fallback": FallbackConfig, "os_kernel": OsKernelConfig, "metrics": MetricsConfig, "server": ServerConfig,
}
_TOP_LEVEL = {"local_llm", "verifier_llm", "input_mode", "input_rules", "input_methods", "output_config",
              "base_system_prompt", "token_counter", "storage_dir", *_SECTIONS}


@dataclass
class Settings:
    local_llm: dict = field(default_factory=lambda: {"backend": "ollama", "model": "llama3"})
    verifier_llm: dict | None = None
    input_mode: str = "rules"
    input_rules: str = "input_savers/INPUT_RULES.md"
    input_methods: list = field(default_factory=list)
    output_config: str = "agent_constraints/components/AGENT_OUTPUT.md"
    base_system_prompt: str = "You are a helpful assistant."
    token_counter: str = "heuristic"  # noqa: S105
    storage_dir: str | None = None
    task_routing: TaskRoutingConfig = field(default_factory=TaskRoutingConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    documents: DocumentsConfig = field(default_factory=DocumentsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    studio: StudioConfig = field(default_factory=StudioConfig)
    os_kernel: OsKernelConfig = field(default_factory=OsKernelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    config_dir: Path | None = None  # folder of the user config file, for resolving relative paths

    def validate(self) -> None:
        if self.input_mode not in ("rules", "steps"):
            raise ConfigError("config input_mode: must be 'rules' or 'steps'")
        if self.token_counter not in ("heuristic", "provider"):
            raise ConfigError("config token_counter: must be 'heuristic' or 'provider'")
        if not isinstance(self.local_llm, dict) or "backend" not in self.local_llm:
            raise ConfigError("config local_llm: needs at least a 'backend'")
        from pal.llm_wrapper.components.base import WrapperConfig
        allowed = {f.name for f in fields(WrapperConfig)}
        if not isinstance(self.local_llm.get("enabled", True), bool):
            raise ConfigError("config local_llm.enabled: must be true or false")
        for label, block in (("local_llm", self.local_llm), ("verifier_llm", self.verifier_llm)):
            for k in (block or {}):
                if k not in allowed and not (label == "local_llm" and k == "enabled"):
                    raise ConfigError(f"config {label}.{k}: unknown key (valid: {', '.join(sorted(allowed))})")

    def resolve_text_file(self, path: str) -> Path | None:
        """Find a packaged or user-relative text file (rules, output config). Absolute paths win, then the
        user config folder, then the package."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p if p.is_file() else None
        for base in (self.config_dir, Path(str(resources.files("pal")))):
            if base is not None and (base / p).is_file():
                return base / p
        return None

    def read_text_file(self, path: str) -> str:
        p = self.resolve_text_file(path)
        return p.read_text(encoding="utf-8").strip() if p else ""

    def to_dict(self) -> dict:
        """Effective settings as plain data (no secrets live here; API keys are per-request)."""
        out: dict = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = {k.name: getattr(v, k.name) for k in fields(v)} if isinstance(v, _Section) else (
                str(v) if isinstance(v, Path) else v)
        return out


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml(text: str, origin: str) -> dict:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{origin}: invalid YAML ({e})") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{origin}: top level must be a mapping")
    return data


def _env_overrides(env) -> dict:
    out: dict = {}
    for name, raw in env.items():
        if not name.startswith(_ENV_PREFIX):
            continue
        parts = [p.lower() for p in name[len(_ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return out


# Settings sections that were renamed with their component. An old settings file or environment variable that still uses the old name keeps working:
# its values are moved to the new section (and win over nothing: a value under the new name beats one under the old name).
RENAMED_SECTIONS = {"graph_context": "knowledge_graph"}


def migrate_renamed(raw: dict) -> dict:
    """The settings with every old section name moved to its new name. Values under the new name win."""
    out = dict(raw)
    for old, new in RENAMED_SECTIONS.items():
        if old in out:
            moved = out.pop(old)
            out[new] = _deep_merge(moved if isinstance(moved, dict) else {}, out.get(new) or {}) if isinstance(moved, dict) else out.get(new, moved)
    return out


def load_settings(path: str | os.PathLike | None = None, env=None, overrides: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    default_text = (resources.files("pal") / "config" / "default.yaml").read_text(encoding="utf-8")
    raw = _load_yaml(default_text, "default.yaml")
    user_path = path or env.get("PAL_CONFIG")
    config_dir = None
    if user_path:
        p = Path(user_path).expanduser()
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        raw = _deep_merge(raw, migrate_renamed(_load_yaml(p.read_text(encoding="utf-8"), str(p))))
        config_dir = p.resolve().parent
    raw = _deep_merge(raw, migrate_renamed(_env_overrides(env)))
    if overrides:
        raw = _deep_merge(raw, migrate_renamed(overrides))
    return settings_from_dict(raw, config_dir)


def settings_from_dict(raw: dict, config_dir: Path | None = None) -> Settings:
    unknown = set(raw) - _TOP_LEVEL - {"config_dir"}
    if unknown:
        raise ConfigError(f"config: unknown key(s): {', '.join(sorted(unknown))}")
    kwargs: dict = {}
    for k, v in raw.items():
        if k in _SECTIONS:
            kwargs[k] = _SECTIONS[k].coerce(v)
        elif k != "config_dir":
            kwargs[k] = v
    s = Settings(config_dir=config_dir, **kwargs)
    s.validate()
    return s
