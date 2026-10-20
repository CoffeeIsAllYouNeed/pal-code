from pal.storage_manager.components.local_file_backend import LocalFileBackend


def test_write_read_round_trip(tmp_path):
    backend = LocalFileBackend(tmp_path / "files")
    backend.write("agent-a", "notes/hello.json", {"x": 1})
    assert backend.read("agent-a", "notes/hello.json") == {"x": 1}


def test_agents_are_isolated(tmp_path):
    backend = LocalFileBackend(tmp_path / "files")
    backend.write("agent-a", "shared.json", "a's data")
    backend.write("agent-b", "shared.json", "b's data")

    assert backend.read("agent-a", "shared.json") == "a's data"
    assert backend.read("agent-b", "shared.json") == "b's data"
    assert backend.list_keys("agent-a") == ["shared.json"]
    assert backend.list_keys("agent-b") == ["shared.json"]


def test_delete_and_exists(tmp_path):
    backend = LocalFileBackend(tmp_path / "files")
    backend.write("agent-a", "k.json", 42)
    assert backend.exists("agent-a", "k.json")
    backend.delete("agent-a", "k.json")
    assert not backend.exists("agent-a", "k.json")
    assert backend.read("agent-a", "k.json") is None


def test_list_keys_prefix_filter(tmp_path):
    backend = LocalFileBackend(tmp_path / "files")
    backend.write("agent-a", "memory/note1.json", "n1")
    backend.write("agent-a", "memory/note2.json", "n2")
    backend.write("agent-a", "kb/doc1.json", "d1")

    memory_keys = backend.list_keys("agent-a", prefix="memory/")
    assert sorted(memory_keys) == ["memory/note1.json", "memory/note2.json"]
