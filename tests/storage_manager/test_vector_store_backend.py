from pal.storage_manager.components.vector_store_backend import VectorStoreBackend


def test_upsert_and_query_returns_relevant_doc():
    store = VectorStoreBackend()  # no persist_directory -> ephemeral/naive
    store.upsert("agent-a", "doc1", "the cat sat on the mat")
    store.upsert("agent-a", "doc2", "paris is the capital of france")

    results = store.query("agent-a", "cat mat", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"


def test_query_empty_collection_returns_empty_list():
    store = VectorStoreBackend()
    assert store.query("agent-with-nothing", "anything", top_k=3) == []


def test_agents_have_isolated_collections():
    store = VectorStoreBackend()
    store.upsert("agent-a", "doc1", "hello from agent a")
    store.upsert("agent-b", "doc1", "hello from agent b")  # same doc_id, different agent

    a_results = store.query("agent-a", "hello", top_k=5)
    b_results = store.query("agent-b", "hello", top_k=5)
    assert len(a_results) == 1 and a_results[0]["text"] == "hello from agent a"
    assert len(b_results) == 1 and b_results[0]["text"] == "hello from agent b"


def test_delete_removes_doc():
    store = VectorStoreBackend()
    store.upsert("agent-a", "doc1", "some text")
    store.delete("agent-a", "doc1")
    assert store.query("agent-a", "some text", top_k=5) == []
