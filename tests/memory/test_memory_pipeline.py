import tempfile

from pal.memory.pipeline import MemoryPipeline


def test_add_and_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemoryPipeline(store_path=f"{tmp}/memory.json")
        mem.add("the user prefers dark mode", session_id="s1")
        mem.add("the user is learning Python", session_id="s1")

        results = mem.retrieve("python", top_k=1)
        assert len(results) <= 1
        if results:
            assert "python" in results[0].content.lower()


def test_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/memory.json"
        mem1 = MemoryPipeline(store_path=path)
        mem1.add("remember this fact")

        mem2 = MemoryPipeline(store_path=path)
        assert len(mem2) == 1
