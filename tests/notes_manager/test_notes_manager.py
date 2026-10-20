import pytest

from pal.access_manager.pipeline import AccessManagerPipeline
from pal.memory.pipeline import MemoryPipeline
from pal.notes_manager import NotesManager, looks_secret
from pal.sdk.pipeline import AgentSDK
from pal.storage_manager.pipeline import StorageManagerPipeline
from pal.tool_manager.pipeline import ToolManagerPipeline


def make(tmp_path, **cfg):
    sm = StorageManagerPipeline(root=str(tmp_path / "s"))
    mem = MemoryPipeline(agent_id="profile", storage_manager=sm)
    sdk = AgentSDK("profile", None, sm, ToolManagerPipeline(), AccessManagerPipeline(), memory=mem)
    scheduler = cfg.pop("scheduler", None)
    return NotesManager(sdk, cfg, scheduler=scheduler)


@pytest.mark.parametrize("text", [
    "my key is sk-abcdefghijklmnopqrstuvwxyz", "AKIAABCDEFGHIJKLMNOP is the id", "password: hunter2hunter2",
    "token = abc123xyz", "-----BEGIN RSA PRIVATE KEY-----", "ghp_" + "a" * 30, "e" * 45 + " hash",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop"])
def test_secrets_are_detected(text):
    assert looks_secret(text)


@pytest.mark.parametrize("text", ["how do I cook rice for four people", "what is the max tokens for a request",
                                  "my dog is called Biscuit"])
def test_normal_text_is_not_a_secret(text):
    assert not looks_secret(text)


def test_only_long_non_secret_messages_are_stored(tmp_path):
    m = make(tmp_path)
    assert not m.remember("s1", "hi there")
    assert not m.remember("s1", "my password: correcthorsebattery")
    assert m.remember("s1", "I am planning a trip to Lisbon in March with my sister")
    assert m.count_notes() == 1


def test_recall_skips_the_current_session_and_respects_limits(tmp_path):
    m = make(tmp_path, recall_notes=2, budget_tokens=60)
    for i in range(4):
        m.remember("old", f"planning a trip to Lisbon number {i} in spring with my sister and friends")
    m.remember("cur", "planning a trip to Lisbon right now with my sister")
    r = m.recall("cur", "trip to Lisbon with my sister")
    assert r is not None and r.hits <= 2 and r.tokens <= 60
    assert r.text.startswith("<memory>") and "right now" not in r.text


def test_recall_returns_nothing_when_nothing_matches_or_disabled(tmp_path):
    m = make(tmp_path)
    m.remember("old", "planning a trip to Lisbon in spring with my sister")
    assert m.recall("new", "quantum chromodynamics lecture notes") is None
    assert m.recall("new", "   ") is None
    off = make(tmp_path / "off", enabled=False)
    assert not off.remember("s", "planning a trip to Lisbon in spring with my sister") and off.recall("n", "trip") is None


def test_clear_and_count(tmp_path):
    m = make(tmp_path)
    m.remember("a", "planning a trip to Lisbon in spring with my sister")
    m.remember("a", "my favourite programming language is Kotlin these days")
    assert m.clear() == 2 and m.count_notes() == 0


def test_notes_are_pruned_when_over_the_cap(tmp_path):
    m = make(tmp_path)
    m.max_notes = 5
    for i in range(12):
        m.remember("a", f"distinct message number {i} about topic {i} for pruning")
    assert m.count_notes() <= 5 + 1


def test_memory_failures_never_raise(tmp_path):
    m = make(tmp_path)
    m.sdk.remember = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    assert m.remember("s", "a long enough message about my plans this weekend") is False
    m.sdk.memory.retrieve = lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone"))
    assert m.recall("s", "plans weekend") is None


# -- the Notes Manager as a component of its own ---------------------------------------------------------------------------------------------
class _Local:
    def __init__(self):
        self.jobs = []

    def submit_steps(self, steps, klass, label):
        self.jobs.append((steps, klass, label))


class _Scheduler:
    def __init__(self):
        self.local = _Local()


def test_forget_session_deletes_only_that_sessions_notes(tmp_path):
    m = make(tmp_path)
    m.remember("s1", "my project is called Papertrail and the budget is five hundred dollars")
    m.remember("s1", "I am planning a trip to Lisbon in March with my sister")
    m.remember("s2", "my dog is called Biscuit and he is afraid of thunder storms")
    assert m.forget_session("s1") == 2 and m.count_notes() == 1 and len(m) == 1
    assert [n.session_id for n in m.sdk.memory.storage.read_all()] == ["s2"] and m.forget_session("nobody") == 0


def test_stats_report_the_notes_the_ram_and_disk_split_and_the_notes_since_the_last_tidy(tmp_path):
    m = make(tmp_path)
    m.remember("s1", "my project is called Papertrail and the budget is five hundred dollars")
    st = m.stats()
    assert st["enabled"] and st["notes"] == 1 and st["max_notes"] == 1000 and st["since_tidy"] == 0
    assert {"ram_count", "disk_count", "usage_ratio", "block_limit"} <= set(st) and st["disk_count"] == 1
    assert make(tmp_path / "off", enabled=False).stats()["enabled"] is False


def test_after_turn_hands_the_tidy_to_the_scheduler_every_n_turns_as_a_background_job(tmp_path):
    from pal.llm_scheduler.types import TaskClass
    m = make(tmp_path, tidy_every=3)
    m.attach_scheduler(sched := _Scheduler())
    for _ in range(2):
        m.after_turn()
    assert sched.local.jobs == [] and m.stats()["since_tidy"] == 2
    m.after_turn()
    (steps, klass, label), = sched.local.jobs
    assert klass is TaskClass.OTHER and label == "tidy memory" and hasattr(steps, "send") and m.stats()["since_tidy"] == 0
    for _ in range(3):
        m.after_turn()
    assert len(sched.local.jobs) == 2                                         # and again, three turns later


def test_after_turn_does_nothing_for_demo_turns_when_off_when_tidy_is_off_or_without_a_scheduler(tmp_path):
    sched = _Scheduler()
    m = make(tmp_path, tidy_every=1)
    m.attach_scheduler(sched)
    m.after_turn(ephemeral=True)
    assert sched.local.jobs == []
    assert make(tmp_path / "a", tidy_every=1, enabled=False, scheduler=None).after_turn() is None
    off = make(tmp_path / "b", tidy_every=0)
    off.attach_scheduler(sched)
    off.after_turn()
    off.after_turn()
    assert sched.local.jobs == []
    make(tmp_path / "c", tidy_every=1).after_turn()                            # no scheduler attached: nothing to run on, nothing raised


def test_a_failing_scheduler_never_fails_the_chat(tmp_path):
    class Broken:
        class local:  # noqa: N801
            @staticmethod
            def submit_steps(*a, **k):
                raise RuntimeError("queue closed")
    m = make(tmp_path, tidy_every=1)
    m.attach_scheduler(Broken())
    m.after_turn()                                                            # logged, not raised


def test_the_manager_takes_its_own_lock_so_callers_do_not_have_to(tmp_path):
    import threading
    m = make(tmp_path)
    errors = []

    def work(i):
        try:
            for j in range(5):
                m.remember(f"s{i}", f"thread {i} note {j} about a completely different subject number {i * 10 + j} alpha{i}{j}")
                m.recall("other", "subject number")
                m.count_notes()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and m.count_notes() == 30
    m.clear()
    assert m.stats()["since_tidy"] == 0 and m.count_notes() == 0


def test_the_context_manager_no_longer_carries_any_note_upkeep():
    from pal.context_manager.chat_context import ChatContext
    assert not any(hasattr(ChatContext, name) for name in ("upkeep", "_tidy_memory"))
    import inspect
    params = list(inspect.signature(ChatContext.__init__).parameters)
    assert "notes" in params and "memory_lock" not in params and "scheduler" not in params
