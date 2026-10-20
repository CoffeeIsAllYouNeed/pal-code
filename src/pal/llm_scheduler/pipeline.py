"""The scheduler as one object: the local model's preemptive priority scheduler, the API model's first-come-first-served
lines, and the slots for code work."""
from __future__ import annotations

from pal.config import SchedulerConfig
from pal.llm_scheduler import core as _core
from pal.llm_scheduler.api import ApiScheduler
from pal.llm_scheduler.code import CodeSlots
from pal.llm_scheduler.core import ScheduledWrapper
from pal.llm_scheduler.local import LocalScheduler
from pal.llm_wrapper.components.base import LLMWrapper


class LLMScheduler:
    def __init__(self, cfg: SchedulerConfig | dict | None = None):
        self.cfg = SchedulerConfig.coerce(cfg)
        c = self.cfg
        self.local = LocalScheduler(c.local_max_pauses, c.other_stale_seconds, c.max_queue) if c.enabled else None
        self.api = ApiScheduler(c.api_retry_429, c.api_backoff_seconds, c.max_queue, enabled=c.enabled)
        self.code = CodeSlots(c.code_workers if c.enabled else 0)
        self.task = _core.task
        self.track = _core.track

    def wrap_local(self, core: LLMWrapper | None) -> LLMWrapper | None:
        if core is None or isinstance(core, ScheduledWrapper):
            return core
        return ScheduledWrapper(core, self.local)

    def interval_for(self, provider: str, model: str = "") -> float:
        """Seconds between call starts on this provider's line. OpenRouter free models are limited to about 20 a minute."""
        if provider == "openrouter" and model.endswith(":free"):
            return float(self.cfg.free_model_interval)
        return float(self.cfg.api_min_interval.get(provider, 0.0))

    def snapshot(self) -> dict:
        return {"enabled": self.cfg.enabled, "local_waiting": self.local.pending() if self.local else {},
                "api_waiting": self.api.pending()}

    def close(self) -> None:
        if self.local:
            self.local.close()
        self.api.close()
