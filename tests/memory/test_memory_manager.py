import math

from pal.memory.components.lru_k import LRUKTracker
from pal.memory.components.manager import MemoryManager
from pal.memory.components.note import MemoryNote
from pal.memory.components.storage_manager import StorageManager
from pal.memory.pipeline import MemoryPipeline


def test_lru_k_infinite_distance_below_k_accesses():
    tracker = LRUKTracker(k=2)
    tracker.record_access("a", when=100.0)
    assert tracker.backward_k_distance("a", now=200.0) == math.inf  # only 1 access, k=2


def test_lru_k_finite_distance_once_k_reached():
    tracker = LRUKTracker(k=2)
    tracker.record_access("a", when=100.0)
    tracker.record_access("a", when=150.0)
    assert tracker.backward_k_distance("a", now=200.0) == 100.0  # 200 - oldest-of-last-2 (100)


def test_lru_k_eviction_order_prefers_evicting_items_below_k():
    tracker = LRUKTracker(k=2)
    tracker.record_access("hot", when=100.0)
    tracker.record_access("hot", when=150.0)  # reached k=2
    tracker.record_access("cold", when=190.0)  # only 1 access -- infinite distance
    order = tracker.eviction_order(["hot", "cold"], now=200.0)
    assert order[0] == "cold"  # evict cold first despite being accessed more recently


def test_storage_manager_round_trip(tmp_path):
    storage = StorageManager(tmp_path / "notes")
    note = MemoryNote(content="hello")
    storage.write(note)

    assert storage.exists(note.id)
    reloaded = storage.read(note.id)
    assert reloaded.content == "hello"

    storage.delete(note.id)
    assert not storage.exists(note.id)
    assert storage.read(note.id) is None


def test_manager_retains_frequently_accessed_item_over_cold_ones(tmp_path):
    storage = StorageManager(tmp_path / "notes")
    manager = MemoryManager(storage, block_limit=4, high_water_mark=0.8, k=2)

    hot = MemoryNote(content="hot item")
    cold = MemoryNote(content="cold item")
    manager.write(hot)
    manager.write(cold)
    manager.read(hot.id)  # second access on hot -> reaches k=2, finite backward distance

    manager.write(MemoryNote(content="filler 1"))
    manager.write(MemoryNote(content="filler 2"))  # ram hits block_limit -> triggers eviction

    assert hot.id in manager.ram
    assert cold.id not in manager.ram  # evicted -- only 1 access, infinite distance

    # nothing is lost -- it's already durable on disk
    assert storage.exists(cold.id)

    # swap back in via read()
    reloaded = manager.read(cold.id)
    assert reloaded is not None
    assert reloaded.content == "cold item"
    assert cold.id in manager.ram  # re-admitted to RAM


def test_pipeline_reports_ram_disk_split_after_eviction(tmp_path):
    mem = MemoryPipeline(store_path=str(tmp_path / "mem"), block_limit=3, high_water_mark=0.9, lru_k=2)
    mem.add("note one")
    mem.add("note two")
    mem.add("note three")  # ram hits block_limit -> one eviction

    stats = mem.stats()
    assert stats["disk_count"] == 3   # every note durably persisted regardless of RAM residency
    assert stats["ram_count"] == 2    # RAM cache bounded below block_limit after eviction
    assert len(mem) == 3              # __len__ reflects total, not just RAM-resident, notes
