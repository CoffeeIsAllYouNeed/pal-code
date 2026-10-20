from pal.memory.pipeline import MemoryPipeline
from pal.storage_manager.pipeline import StorageManagerPipeline


def test_two_agents_share_one_storage_manager_but_stay_isolated(tmp_path):
    """Matches AIOS 3.6: 'the storage manager reads and writes data
    based on the agent ID passed from the memory manager.' Two
    MemoryPipelines for two different agents, backed by the SAME
    StorageManagerPipeline, must not see each other's notes."""
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"))

    mem_a = MemoryPipeline(agent_id="agent-a", storage_manager=sm)
    mem_b = MemoryPipeline(agent_id="agent-b", storage_manager=sm)

    mem_a.add("agent a's private note")
    mem_b.add("agent b's private note")

    assert len(mem_a) == 1
    assert len(mem_b) == 1

    a_notes = [n.content for n in mem_a.storage.read_all()]
    b_notes = [n.content for n in mem_b.storage.read_all()]
    assert a_notes == ["agent a's private note"]
    assert b_notes == ["agent b's private note"]


def test_agent_scoped_memory_triggers_lru_k_eviction_through_shared_storage(tmp_path):
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"))
    mem = MemoryPipeline(agent_id="agent-a", storage_manager=sm, block_limit=3, high_water_mark=0.9, lru_k=2)

    mem.add("note one")
    mem.add("note two")
    mem.add("note three")  # ram hits block_limit -> eviction

    stats = mem.stats()
    assert stats["disk_count"] == 3
    assert stats["ram_count"] == 2

    # the evicted note is still readable via the shared storage manager
    # directly, scoped to this agent -- proving it really landed on disk,
    # not lost.
    keys = sm.list_keys("agent-a", prefix="memory/")
    assert len(keys) == 3
