import os

import pytest

from pal.storage_manager.components.local_file_backend import InvalidStorageKey, LocalFileBackend
from pal.storage_manager.pipeline import StorageManagerPipeline


@pytest.fixture
def be(tmp_path):
    return LocalFileBackend(tmp_path / "root")


@pytest.mark.parametrize("key", ["../x", "a/../../x", "/etc/passwd", "C:\\x", "c:x", "a\\b", "con", "a/nul.txt", "",
                                 "a:b", "x\x00y", "trail."])
def test_keys_that_escape_or_break_windows_are_rejected(be, key, tmp_path):
    for op in (lambda: be.write("agent", key, 1), lambda: be.read("agent", key), lambda: be.delete("agent", key),
               lambda: be.exists("agent", key)):
        with pytest.raises(InvalidStorageKey):
            op()
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize("agent", ["../evil", "a/b", "", "a b", "C:", ".."])
def test_agent_ids_cannot_escape_the_root(be, agent):
    with pytest.raises(InvalidStorageKey):
        be.write(agent, "k.json", 1)


def test_symlink_escape_is_blocked(be, tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("no symlinks")
    be.write("agent", "ok.json", 1)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, tmp_path / "root" / "agent" / "link")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    with pytest.raises(InvalidStorageKey):
        be.write("agent", "link/x.json", 1)


def test_round_trip_types_and_posix_key_listing(be):
    be.write("a", "docs/one.json", {"k": [1, 2]})
    be.write("a", "docs/two.bin", b"\xff\x00\xfe")
    be.write("a", "top.json", "é")
    assert be.read("a", "docs/one.json") == {"k": [1, 2]} and be.read("a", "docs/two.bin") == b"\xff\x00\xfe"
    assert be.read("a", "top.json") == "é" and be.read("a", "missing.json") is None
    assert be.list_keys("a") == ["docs/one.json", "docs/two.bin", "top.json"]
    assert be.list_keys("a", prefix="docs/") == ["docs/one.json", "docs/two.bin"]


def test_writes_are_atomic_and_leave_no_temp_files(be, tmp_path):
    for i in range(5):
        be.write("a", "k.json", {"i": i})
    assert be.read("a", "k.json") == {"i": 4}
    assert [p.name for p in (tmp_path / "root" / "a").iterdir()] == ["k.json"]


def test_agents_are_isolated_and_purge_removes_only_one(be):
    be.write("a", "k.json", 1)
    be.write("b", "k.json", 2)
    be.purge("a")
    assert be.read("a", "k.json") is None and be.read("b", "k.json") == 2


def test_pipeline_purge_agent_clears_files_and_vectors(tmp_path):
    sm = StorageManagerPipeline(root=str(tmp_path / "s"))
    sm.write("a", "f.json", 1)
    sm.vector_upsert("a", "d1", "alpha beta")
    sm.vector_upsert("b", "d1", "alpha beta")
    sm.purge_agent("a")
    assert sm.list_keys("a") == [] and sm.vector_query("a", "alpha") == [] and len(sm.vector_query("b", "alpha")) == 1


def test_queue_does_not_leak_finished_requests(tmp_path):
    from pal.storage_manager.utils.sdk import StorageSDK
    sm = StorageManagerPipeline(root=str(tmp_path / "s"))
    sdk = StorageSDK("a", sm)
    for i in range(50):
        sdk.write(f"k{i}.json", i)
        sdk.read(f"k{i}.json")
    assert sm.queue.requests == {} and len(sm.queue.queue) == 0


def test_invalid_key_through_the_sdk_is_an_error_not_a_write(tmp_path):
    from pal.storage_manager.utils.sdk import StorageSDK
    sdk = StorageSDK("a", StorageManagerPipeline(root=str(tmp_path / "s")))
    with pytest.raises(RuntimeError, match="invalid storage key"):
        sdk.write("../escape.json", 1)
    assert not (tmp_path / "escape.json").exists()
