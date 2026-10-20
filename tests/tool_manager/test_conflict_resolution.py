from pal.tool_manager.components.base import ToolSpec
from pal.tool_manager.components.conflict_resolver import ToolConflictResolver
from pal.tool_manager.components.tool_call_queue import ToolCallQueue
from pal.tool_manager.components.tool_registry import ToolRegistry
from pal.tool_manager.pipeline import ToolManagerPipeline


def test_conflict_resolver_tracks_in_flight_counts():
    resolver = ToolConflictResolver()
    assert resolver.can_acquire("search", parallel_limit=1)

    resolver.acquire("search")
    assert resolver.in_flight_count("search") == 1
    assert not resolver.can_acquire("search", parallel_limit=1)  # at capacity

    resolver.release("search")
    assert resolver.in_flight_count("search") == 0
    assert resolver.can_acquire("search", parallel_limit=1)


def test_unlimited_tool_never_conflicts():
    resolver = ToolConflictResolver()
    resolver.acquire("free_tool")
    resolver.acquire("free_tool")
    resolver.acquire("free_tool")
    assert resolver.can_acquire("free_tool", parallel_limit=None)


def test_queue_skips_conflicting_request_for_next_runnable_one():
    """The exact behavior AIOS 3.7 describes: 'upon detecting conflicts,
    the system advances to subsequent queue requests until identifying
    a conflict-free candidate.'"""
    registry = ToolRegistry()
    registry.register(ToolSpec(name="search", loader=lambda: (lambda: "search result"), parallel_limit=1))
    registry.register(ToolSpec(name="calc", loader=lambda: (lambda: "calc result"), parallel_limit=None))

    resolver = ToolConflictResolver()
    queue = ToolCallQueue(registry, resolver)

    # Simulate "search" already being in-flight elsewhere (e.g. another
    # agent's still-running call holding the only slot).
    resolver.acquire("search")

    id_search = queue.submit("agent-a", "search", {})
    id_calc = queue.submit("agent-b", "calc", {})

    # dispatch_next() must skip the conflicting "search" request (queue
    # position unchanged) and run "calc" instead.
    dispatched = queue.dispatch_next()
    assert dispatched.id == id_calc
    assert dispatched.status == "done"
    assert dispatched.result.result == "calc result"

    # "search" is still queued, untouched.
    search_entry = queue.get_result(id_search)
    assert search_entry.status == "queued"
    assert queue.pending_count() == 1

    # free the slot -- now "search" becomes runnable.
    resolver.release("search")
    dispatched2 = queue.dispatch_next()
    assert dispatched2.id == id_search
    assert dispatched2.status == "done"
    assert dispatched2.result.result == "search result"
    assert queue.pending_count() == 0


def test_dispatch_next_returns_none_when_everything_conflicts():
    registry = ToolRegistry()
    registry.register(ToolSpec(name="search", loader=lambda: (lambda: "x"), parallel_limit=1))

    resolver = ToolConflictResolver()
    queue = ToolCallQueue(registry, resolver)
    resolver.acquire("search")  # occupy the only slot

    queue.submit("agent-a", "search", {})
    assert queue.dispatch_next() is None
    assert queue.pending_count() == 1  # left in place, not dropped


def test_pipeline_invoke_end_to_end():
    tm = ToolManagerPipeline()
    from pal.tool_manager.components.base import ToolParam
    tm.register_tool("greet", lambda: (lambda name: f"hello {name}"), params=[ToolParam("name", str)])

    result = tm.invoke("agent-a", "greet", {"name": "world"})
    assert result.ok
    assert result.result == "hello world"


def test_pipeline_invoke_returns_validation_error_not_exception():
    tm = ToolManagerPipeline()
    from pal.tool_manager.components.base import ToolParam
    tm.register_tool("greet", lambda: (lambda name: f"hello {name}"), params=[ToolParam("name", str)])

    result = tm.invoke("agent-a", "greet", {})  # missing required "name"
    assert not result.ok
    assert "missing required parameter" in result.error
