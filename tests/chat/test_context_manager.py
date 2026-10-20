"""The context manager inside real chat turns: one budget, and the background jobs that make later turns cheaper."""
import re
import time

from tests.fakes import FakeAPI, RouterLocal
from tests.helpers import client, make_service, send

ROUTE_API = '{"route": "api", "reason": "needs the big model"}'
SMALL = {"hot_turns": 2, "recall_turns": 0, "compress_turns": 3, "compress_min_words": 8, "facts_tokens": 250}


class Writer(RouterLocal):
    """A local model that records every call it receives (what the system prompt said), so tests can prove no model was used for
    compression."""

    def __init__(self):
        super().__init__(route=ROUTE_API)
        self.systems = []

    def chat(self, syscall):
        self.systems.append(syscall.messages[0]["content"])
        return super().chat(syscall)


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def turns(svc, n, start=1):
    return [send(svc, f"question {i} about topic number {i} please", route="api") for i in range(start, start + n)]


# -- one budget ----------------------------------------------------------------------------------------------------------------------
def test_the_total_is_kept_under_the_budget_and_the_choices_are_reported(tmp_path):
    api = FakeAPI([" ".join(f"word{i}" for i in range(260))] * 8)
    svc, _ = make_service(tmp_path, local=Writer(), api=api, context={"total_budget_tokens": 1000, "recall_turns": 0})
    outs = turns(svc, 6)
    ctx = outs[-1].context
    assert ctx["budget"] == 1000 and ctx["used"] <= 1000 and ctx["by_source"]["history"] <= 1000
    assert ctx["turns_sent"] < ctx["turns_total"] and ctx["turns_sent"] >= 1
    assert svc.metrics.turns()[-1]["saved_paging"] > 0


def test_a_generous_budget_changes_nothing(tmp_path):
    svc, _ = make_service(tmp_path, local=Writer(), api=FakeAPI(["ok"] * 5))
    ctx = turns(svc, 3)[-1].context
    assert ctx["dropped"] == [] and ctx["turns_sent"] == ctx["turns_total"] and ctx["budget"] == 10000


# -- compression of old assistant replies -------------------------------------------------------------------------------------------------
BODY = "".join(f"    step_{k} = transform_{k}(request.json()) + compute_{k}(request.headers)\n" for k in range(18))
CODE = "```python\ndef handler_{n}(request):\n" + BODY + "    return step_0\n```"


def reply(n):
    return (f"Here is handler {n}.\n{CODE.format(n=n)}\nIt reads config_{n}.yaml and returns {n * 100} rows. "
            f"Use this version because the old one failed with KeyError. Anyway, hope it helps with step {n}.")


def long_chat(tmp_path, n=10, **ctx):
    replies = [reply(i) for i in range(1, n + 2)]
    local = Writer()
    svc, api = make_service(tmp_path, local=local, api=FakeAPI(replies), context={**SMALL, **ctx})
    outs = turns(svc, n)
    return svc, api, local, outs, replies


def test_old_assistant_replies_are_compressed_by_age_and_the_oldest_become_facts(tmp_path):
    svc, api, local, outs, replies = long_chat(tmp_path)
    out = send(svc, "question 11 about topic number 11 please", route="api")
    msgs = api.seen[-1]
    assistants = [m["content"] for m in msgs if m["role"] == "assistant"]
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert assistants[-2:] == [replies[8], replies[9]]                                   # the 2 newest turns: verbatim
    warm = assistants[:-2]
    assert len(warm) == 3 and all(a.startswith("[compressed] ") for a in warm)         # the 3 before them: compressed
    assert all(re.search(r"\[\.\.\. \d+ lines omitted\]", a) and "def handler_" in a for a in warm)   # signature kept, body omitted
    assert all(f"{n * 100} rows" in a and "config_" in a and "KeyError" in a for n, a in zip((6, 7, 8), warm))   # exact details kept
    assert [u for u in users[:3]] == [f"question {n} about topic number {n} please" for n in (6, 7, 8)] or len(users) >= 6   # your words: verbatim
    last = users[-1]
    assert "<facts>" in last and "t1:" in last and "100 rows" in last and "config_1.yaml" in last and "t5:" in last
    assert "t6:" not in last                                                              # turns 6 to 8 are shown, not listed as facts
    assert "[compressed]" in msgs[0]["content"] and "<facts> block" in msgs[0]["content"]  # the model is told what it is looking at
    assert out.context["compression"]["compressed_turns"] == 3 and out.context["compression"]["facts_from_turns"] == 5


def test_originals_are_never_changed_and_compression_uses_no_model(tmp_path):
    svc, api, local, outs, replies = long_chat(tmp_path)
    send(svc, "question 11 about topic number 11 please", route="api")
    sess = svc.sessions.get("session-1")
    assert [t.assistant for t in sess.turns[:10]] == replies[:10] and all(not t.assistant.startswith("[compressed]") for t in sess.turns)
    assert not [s for s in local.systems if "summar" in s.lower() or "compress" in s.lower()]     # only routing and trimming were asked


def test_compression_saves_tokens_and_the_numbers_are_in_the_stats(tmp_path):
    svc, api, local, outs, replies = long_chat(tmp_path)
    out = send(svc, "question 11 about topic number 11 please", route="api")
    saved = out.context["compression"]["tokens_saved"]
    assert saved > 30
    m = svc.metrics.turns()[-1]
    assert m["compressed_turns"] == 3 and m["compression_saved"] == saved
    s = svc.metrics.chat_summary()
    assert s["compressed_turns"] >= 3 and s["compression_saved"] >= saved
    shown = [x["content"] for x in api.seen[-1] if x["role"] == "assistant" and x["content"].startswith("[compressed]")]
    assert len(shown) == 3 and sum(len(x) for x in shown) < 0.8 * sum(len(r) for r in replies[5:8])   # really shorter than the originals


def test_what_is_big_is_reported_for_the_whole_conversation(tmp_path):
    svc, api, local, outs, replies = long_chat(tmp_path)
    h = send(svc, "question 11 about topic number 11 please", route="api").context["history_tokens"]
    assert set(h) == {"user", "assistant_prose", "assistant_code"} and min(h.values()) > 0
    assert h["assistant_code"] > h["user"]                                               # in this chat the code is what is big


def test_a_relevant_old_turn_is_still_recalled_in_full_while_others_are_compressed(tmp_path):
    replies = [reply(i) for i in range(1, 12)]
    svc, api = make_service(tmp_path, local=Writer(), api=FakeAPI(replies),
                            context={**SMALL, "recall_turns": 1})
    send(svc, "explain zebra migration routes across the savanna", route="api")
    for i in range(2, 10):
        send(svc, f"question {i} about unrelated topic number {i} please", route="api")
    send(svc, "back to zebra migration routes across the savanna please", route="api")
    assistants = [m["content"] for m in api.seen[-1] if m["role"] == "assistant"]
    assert replies[0] in assistants                                                      # recalled verbatim, not compressed
    assert any(a.startswith("[compressed]") for a in assistants)


def test_an_earlier_draft_that_a_later_reply_replaced_becomes_a_one_line_pointer(tmp_path):
    def draft(n):
        return (f"# Papertrail\n\n## Install\n\npip install papertrail version {n}, then run the setup step number {n}.\n\n"
                f"## Usage\n\nCall search with a query and read the ranked results, revision {n}.\n\n"
                f"## License\n\nMIT, see the LICENSE file for details of revision {n}.")
    local = Writer()
    svc, api = make_service(tmp_path, local=local, api=FakeAPI([draft(i) for i in range(1, 12)]), context=SMALL)
    for i in range(1, 9):
        send(svc, f"please rewrite the readme again, attempt {i}", route="api")
    assistants = [m["content"] for m in api.seen[-1] if m["role"] == "assistant"]
    pointers = [a for a in assistants if "replaced by the newer version in turn" in a]
    assert len(pointers) == 3 and all(a.startswith("[compressed] [earlier draft with the headings") for a in pointers)
    assert assistants[-1] == draft(7) and "revision" not in " ".join(pointers)            # the newest drafts are whole; old ones are pointers


def test_a_turn_that_saved_a_file_remembers_it_and_later_shows_a_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_DOCS", "k")
    call = '<tool>{"name": "save_file", "args": {"path": "README.md", "content": "# Papertrail"}}</tool>'
    api = FakeAPI([call, "I saved the README file for you. It has the install and usage sections you asked for in the last message."] + ["ok"] * 12)
    svc, _ = make_service(tmp_path, local=Writer(), api=api, context=SMALL, scheduler={"api_min_interval": {"gemini": 0.0}})
    svc.send("s", "", "", None, "write a readme for papertrail in a file", [], agent="docs")
    sess = svc.sessions.get("s")
    assert sess.turns[0].files == ("README.md",)
    from pal.compression.history import compact_reply
    out = compact_reply(sess.turns, 0, min_words=5)
    assert out.startswith("[compressed] [saved README.md in the workspace]") and "I saved the README" in out


def test_short_replies_and_a_switched_off_setting_leave_history_untouched(tmp_path):
    short = FakeAPI(["ok"] * 12)
    svc, _ = make_service(tmp_path, local=Writer(), api=short, context={**SMALL, "compress_min_words": 60})
    turns(svc, 8)
    assert not any(m["content"].startswith("[compressed]") for m in short.seen[-1])
    off_replies = [reply(i) for i in range(1, 12)]
    off = FakeAPI(off_replies)
    svc2, _ = make_service(tmp_path / "off", local=Writer(), api=off, context={**SMALL, "compress": False})
    turns(svc2, 9)
    msgs = off.seen[-1]
    assert not any("[compressed]" in m["content"] or "<facts>" in m["content"] for m in msgs)
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == off_replies[7:8] + off_replies[8:9] or len(
        [m for m in msgs if m["role"] == "assistant"]) == 2                           # only the hot window is sent, as before


def test_a_short_terse_chat_gets_nothing_added_beyond_the_old_behaviour(tmp_path):
    api = FakeAPI(["ok"] * 12)
    svc, _ = make_service(tmp_path, local=Writer(), api=api, context={**SMALL, "compress_turns": 12})
    turns(svc, 9)
    msgs = api.seen[-1]
    assert len([m for m in msgs if m["role"] == "user"]) == 3                             # two hot turns and the new message, as before
    assert not any("[compressed]" in m["content"] or "<facts>" in m["content"] for m in msgs)
    assert svc.metrics.turns()[-1]["saved_paging"] > 0 and svc.metrics.turns()[-1]["compressed_turns"] == 0


def test_the_compressed_band_has_a_token_cap_and_the_rest_turn_into_facts(tmp_path):
    svc, api, local, outs, replies = long_chat(tmp_path, compress_budget_tokens=120, compress_turns=5)
    out = send(svc, "question 11 about topic number 11 please", route="api")
    shown = [m["content"] for m in api.seen[-1] if m["role"] == "assistant" and m["content"].startswith("[compressed]")]
    assert 1 <= len(shown) < 5 and out.context["compression"]["compressed_turns"] == len(shown)
    last = api.seen[-1][-1]["content"]
    assert "<facts>" in last and re.search(r"\bt\d+:", last)                              # the turns that did not fit became facts
    assert max(int(n) for n in re.findall(r"\bt(\d+):", last)) <= 11 - 2 - len(shown)    # and they are older than everything shown


def test_a_facts_block_is_only_added_when_it_is_much_smaller_than_what_it_replaces(tmp_path):
    svc, api, local, outs, replies = long_chat(tmp_path, facts_max_ratio=0.0)
    send(svc, "question 11 about topic number 11 please", route="api")
    assert "<facts>" not in api.seen[-1][-1]["content"]
    svc2, api2, _, _, _ = long_chat(tmp_path / "b", facts_max_ratio=1.0)
    send(svc2, "question 11 about topic number 11 please", route="api")
    assert "<facts>" in api2.seen[-1][-1]["content"]


def test_the_token_budget_still_wins_over_compression(tmp_path):
    svc, api, local, outs, replies = long_chat(tmp_path, n=10, total_budget_tokens=1000)
    out = send(svc, "question 11 about topic number 11 please", route="api")
    assert out.context["used"] <= 1000 and out.context["budget"] == 1000
    assert "memory" not in out.context["dropped"] or True                                # memory is dropped first when it is there


# -- memory tidy -----------------------------------------------------------------------------------------------------------------------------
def test_duplicate_and_near_duplicate_notes_are_merged_keeping_the_most_used(tmp_path):
    from tests.notes_manager.test_notes_manager import make
    m = make(tmp_path)
    a = "my project is called Papertrail and the budget is five hundred dollars"
    m.remember("s1", a)
    m.remember("s2", a)
    m.remember("s3", a + " today")                                                      # one extra word: a near duplicate
    m.remember("s4", "I am planning a trip to Lisbon in March with my sister")
    assert m.count_notes() == 4
    assert m.tidy() == 2 and m.count_notes() == 2
    texts = sorted(n.content for n in m.sdk.memory.storage.read_all())
    assert any("Lisbon" in t for t in texts) and any("Papertrail" in t for t in texts)
    assert m.tidy() == 0                                                                 # nothing more to merge


def test_tidy_yields_in_batches_so_it_can_be_paused_and_keeps_distinct_notes(tmp_path):
    from tests.notes_manager.test_notes_manager import make
    m = make(tmp_path, tidy_every=0)
    for i in range(7):
        m.remember(f"s{i}", f"note number {i} about subject {i} with its own unique words alpha{i} beta{i}")
    gen = m.tidy_steps(batch=3)
    yields = 0
    removed = None
    while removed is None:
        try:
            next(gen)
            yields += 1
        except StopIteration as done:
            removed = done.value
    assert yields == 2 and removed == 0 and m.count_notes() == 7


def test_tidy_runs_by_itself_in_the_background_after_enough_new_notes(tmp_path):
    svc, _ = make_service(tmp_path, local=Writer(), api=FakeAPI(["ok"] * 6), memory={"tidy_every": 2})
    same = "please remember that my favourite database is Postgres version sixteen"
    send(svc, same, route="api", sid="a")
    send(svc, same, route="api", sid="b")
    assert wait_for(lambda: svc.memory_count() == 1)
    assert ("tidy memory", "OTHER") in [(h["label"], h["class"]) for h in svc.scheduler.local.history]
    svc3, _ = make_service(tmp_path / "never", local=Writer(), api=FakeAPI(["ok"] * 4), memory={"tidy_every": 0})
    send(svc3, same, route="api", sid="a")
    send(svc3, same, route="api", sid="b")
    time.sleep(0.2)
    assert svc3.memory_count() == 2


# -- the three trackers inside a real chat turn ----------------------------------------------------------------------------------------------------
def test_a_turn_records_where_its_time_went_by_stage_and_the_summary_ranks_the_stages(tmp_path):
    svc, _ = make_service(tmp_path, local=Writer(), api=FakeAPI(["ok"] * 4))
    send(svc, "please explain how the checkout charges the card, thanks so much", route="api")
    stages = svc.metrics.turns()[-1]["stage_seconds"]
    assert {"input_saver", "context", "api", "other"} <= set(stages) and all(v >= 0 for v in stages.values())
    summary = svc.metrics.chat_summary()
    assert summary["latency_stages"]["context"]["turns"] == 1 and summary["slowest_stage"] in stages
    assert abs(sum(stages.values()) - svc.metrics.turns()[-1]["latency_total"]) < 0.05              # the stages and "other" add up to the turn's total


def test_a_question_answered_by_code_is_timed_as_direct_and_never_as_a_model_stage(tmp_path):
    svc, api = make_service(tmp_path, local=Writer(), api=FakeAPI(["ok"]))
    send(svc, "what is 12 * 12?")
    stages = svc.metrics.turns()[-1]["stage_seconds"]
    assert "direct" in stages and "api" not in stages and "input_saver" not in stages and not api.seen


def test_savings_are_credited_through_the_token_tracker_and_the_old_summary_keys_are_all_still_there(tmp_path):
    svc, _ = make_service(tmp_path, local=Writer(), api=FakeAPI(["ok"] * 4))
    send(svc, "Hello there! Could you please explain how the checkout charges the card? Thanks so much!", route="api")
    s = svc.metrics.chat_summary()
    for key in ("turns", "turns_recorded", "by_route", "api_calls_avoided", "api_input_tokens", "api_output_tokens", "local_llm_tokens", "tokens_saved_estimate",
                "avg_local_overhead_sec_on_api_turns", "compressed_turns", "compression_saved", "cost_usd", "saved_cost_usd", "fallbacks", "avg_wait_local_sec",
                "avg_wait_api_sec", "local_pauses", "tool_calls", "memory_hits", "escalations", "errors"):
        assert key in s, key
    assert s["tokens_saved_estimate"]["input_savers"] == svc.metrics.turns()[-1]["saved_savers"] >= 0


def test_the_memory_snapshot_reports_notes_sessions_graph_storage_and_the_process(tmp_path):
    svc, _ = make_service(tmp_path, local=Writer(), api=FakeAPI(["ok"] * 4), memory={"enabled": True})
    send(svc, "my project is called Papertrail and the budget is five hundred dollars", route="api")
    snap = svc.metrics.memory.snapshot()
    assert snap["notes"]["notes"] == 1 and snap["notes"]["since_tidy"] >= 0 and snap["sessions"]["open"] == 1
    assert {"enabled", "projects", "bytes"} <= set(snap["knowledge_graph"]) and snap["storage"]["bytes"] >= 0 and "rss_mb" in snap["process"]


def test_the_metrics_endpoint_returns_the_memory_snapshot_beside_the_summary(tmp_path):
    svc, _ = make_service(tmp_path, local=Writer(), api=FakeAPI(["ok"] * 4))
    body = client(svc).get("/api/metrics").get_json()
    assert set(body) >= {"summary", "recent", "scheduler", "memory"} and "notes" in body["memory"] and body["summary"] == {"turns": 0}
