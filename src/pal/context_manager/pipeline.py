from pal.context_manager.components.context_cache import ContextCache
from pal.context_manager.components.kv_bridge import KVBridge
from pal.context_manager.components.pager import ContextPager, Segment
from pal.llm_wrapper.components.base import LLMWrapper


class ContextManagerPipeline:
    """Ties system-prompt caching, hot/cold history paging, and the
    backend's native KV cache together behind one small interface."""

    def __init__(self, core: LLMWrapper, cache_ttl: float = 3600, hot_window: int = 6, cold_threshold: float = 0.35):
        self.cache = ContextCache(ttl=cache_ttl)
        self.pager = ContextPager(hot_window=hot_window, cold_threshold=cold_threshold)
        self.kv_bridge = KVBridge(core)

    def cache_system_prompt(self, system_prompt: str) -> tuple[bool, str]:
        return self.cache.get_or_store(system_prompt)

    def page_history(self, history: list[Segment], query: str) -> list[Segment]:
        """Returns hot + relevant paged-in cold segments, in order, ready
        to append to a request's message list."""
        hot, cold_refs = self.pager.page_out(history)
        recalled = self.pager.page_in(query, cold_refs)
        return recalled + hot

    def size_bytes(self) -> int:
        return self.cache.size_bytes()
