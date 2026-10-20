from pal.context_manager.components.context_cache import ContextCache


def test_cache_hit_on_repeat():
    cache = ContextCache()
    hit1, k1 = cache.get_or_store("system prompt text")
    hit2, k2 = cache.get_or_store("system prompt text")
    assert hit1 is False
    assert hit2 is True
    assert k1 == k2


def test_cache_miss_on_different_content():
    cache = ContextCache()
    _, k1 = cache.get_or_store("a")
    _, k2 = cache.get_or_store("b")
    assert k1 != k2
