from pal.storage_manager.components.base import StorageOp, StorageSyscall
from pal.storage_manager.pipeline import StorageManagerPipeline
from pal.storage_manager.utils.sdk import StorageSDK


def test_queue_processes_fifo(tmp_path):
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"))

    id1 = sm.submit(StorageSyscall(agent_id="a", op=StorageOp.WRITE, key="k1.json", data="first"))
    id2 = sm.submit(StorageSyscall(agent_id="a", op=StorageOp.WRITE, key="k2.json", data="second"))

    first = sm.tick()
    assert first.id == id1
    assert first.status.value == "done"
    assert sm.read("a", "k1.json") == "first"

    second = sm.tick()
    assert second.id == id2
    assert sm.read("a", "k2.json") == "second"


def test_drain_processes_everything(tmp_path):
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"))
    for i in range(5):
        sm.submit(StorageSyscall(agent_id="a", op=StorageOp.WRITE, key=f"k{i}.json", data=i))

    finished = sm.drain()
    assert len(finished) == 5
    assert all(r.status.value == "done" for r in finished)
    assert sorted(sm.list_keys("a")) == [f"k{i}.json" for i in range(5)]


def test_unknown_key_read_returns_none_not_error(tmp_path):
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"))
    sm.submit(StorageSyscall(agent_id="a", op=StorageOp.READ, key="missing.json"))
    result = sm.tick()
    assert result.status.value == "done"
    assert result.response.data is None


def test_sdk_read_write_round_trip(tmp_path):
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"))
    sdk = StorageSDK("agent-a", sm)

    sdk.write("prefs.json", {"theme": "dark"})
    assert sdk.read("prefs.json") == {"theme": "dark"}
    assert "prefs.json" in sdk.list_keys()

    sdk.delete("prefs.json")
    assert sdk.read("prefs.json") is None


def test_sdk_vector_operations(tmp_path):
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"))
    sdk = StorageSDK("agent-a", sm)

    sdk.vector_upsert("doc1", "the weather in paris is sunny")
    results = sdk.vector_query("weather paris", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"


def test_sdk_raises_on_backend_error(tmp_path, monkeypatch=None):
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"))
    sdk = StorageSDK("agent-a", sm)

    # Force an error path via an unknown op value bypassing the enum,
    # simulating a backend failure surfaced through execute()'s
    # try/except -- SDK should raise rather than silently return None.
    bad_syscall = StorageSyscall(agent_id="agent-a", op="not_a_real_op", key="x")
    try:
        sdk._submit_and_wait(bad_syscall)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "storage syscall failed" in str(e)
