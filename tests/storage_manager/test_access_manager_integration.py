from pal.access_manager.pipeline import AccessManagerPipeline
from pal.storage_manager.pipeline import StorageManagerPipeline
from pal.storage_manager.utils.sdk import StorageSDK


def test_own_read_write_never_gated_even_without_privilege_group(tmp_path):
    am = AccessManagerPipeline()  # default confirmation handler denies
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)
    sdk = StorageSDK("agent-a", sm)

    # plain write() to a NEW key -- not destructive, not cross-agent --
    # must succeed even though agent-a has no assigned privilege group
    # and the confirmation handler denies everything.
    sdk.write("notes.json", {"x": 1})
    assert sdk.read("notes.json") == {"x": 1}


def test_overwrite_of_own_key_requires_confirmation(tmp_path):
    am = AccessManagerPipeline()  # default handler denies
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)
    sdk = StorageSDK("agent-a", sm)

    sdk.write("notes.json", {"x": 1})  # first write -- fresh key, allowed
    try:
        sdk.write("notes.json", {"x": 2})  # second write -- overwrite, denied by default handler
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "declined confirmation" in str(e)

    # approve overwrites -- now it succeeds
    am.set_confirmation_handler(lambda req: True)
    sdk.write("notes.json", {"x": 2})
    assert sdk.read("notes.json") == {"x": 2}


def test_delete_requires_confirmation(tmp_path):
    am = AccessManagerPipeline()
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)
    sdk = StorageSDK("agent-a", sm)
    sdk.write("notes.json", "hello")

    try:
        sdk.delete("notes.json")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert sdk.read("notes.json") == "hello"  # untouched

    am.set_confirmation_handler(lambda req: True)
    sdk.delete("notes.json")
    assert sdk.read("notes.json") is None


def test_cross_agent_read_denied_without_shared_privilege_group(tmp_path):
    am = AccessManagerPipeline()
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)
    sdk_a = StorageSDK("agent-a", sm)
    sdk_b = StorageSDK("agent-b", sm)

    sdk_a.write("secret.json", "a's secret")

    try:
        sdk_b.read_from("agent-a", "secret.json")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "access denied" in str(e)


def test_cross_agent_read_allowed_within_shared_privilege_group(tmp_path):
    am = AccessManagerPipeline()
    am.assign_privilege_group("agent-a", "team-x")
    am.assign_privilege_group("agent-b", "team-x")
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)
    sdk_a = StorageSDK("agent-a", sm)
    sdk_b = StorageSDK("agent-b", sm)

    sdk_a.write("shared.json", "team data")
    assert sdk_b.read_from("agent-a", "shared.json") == "team data"


def test_direct_calls_bypass_access_manager_entirely(tmp_path):
    """The memory manager's swaps (AgentScopedMemoryStorage) use the
    direct read/write/etc. calls, not execute() -- those are
    intentionally never gated, even with an access_manager wired in,
    since they're not the cross-agent case AIOS 3.8 regulates."""
    am = AccessManagerPipeline()  # would deny everything destructive
    sm = StorageManagerPipeline(root=str(tmp_path / "storage"), access_manager=am)

    sm.write("agent-a", "k.json", "v1")
    sm.write("agent-a", "k.json", "v2")  # a direct overwrite -- not gated
    assert sm.read("agent-a", "k.json") == "v2"
    sm.delete("agent-a", "k.json")  # a direct delete -- not gated
    assert sm.read("agent-a", "k.json") is None
