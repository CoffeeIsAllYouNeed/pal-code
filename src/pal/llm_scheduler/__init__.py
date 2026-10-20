"""Scheduling. API model: first come, first served, one line per API key. Local model: preemptive priority (agent, then chat,
then other), first come first served inside each class. Code work: a small pool of slots beside both."""
from pal.llm_scheduler.api import ApiLine, ApiScheduler, is_rate_limit
from pal.llm_scheduler.code import CodeSlots
from pal.llm_scheduler.core import ScheduledWrapper, current_stats, task, track
from pal.llm_scheduler.local import LocalScheduler
from pal.llm_scheduler.pipeline import LLMScheduler
from pal.llm_scheduler.types import ExpiredError, QueueFullError, SchedulerError, TaskClass, Ticket, TurnStats

__all__ = ["ApiLine", "ApiScheduler", "CodeSlots", "ExpiredError", "LocalScheduler", "QueueFullError", "ScheduledWrapper",
           "LLMScheduler", "SchedulerError", "TaskClass", "Ticket", "TurnStats", "current_stats", "is_rate_limit", "task", "track"]
