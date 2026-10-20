import threading

import pytest
from tests.fakes import FakeAPI, RouterLocal
from tests.helpers import doc, make_service, send

from pal.errors import ProviderError, UserInputError
from pal.task_router import API_NOTICE, LOCAL_TAG

API = {"route": "api"}


def user_msgs(api, call=-1):
    return [m["content"] for m in api.seen[call] if m["role"] == "user"]


# -- prompts -------------------------------------------------------------------------------------
def test_output_rules_ride_on_every_api_request(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "explain caching", route="api")
    send(svc, "and eviction", route="api")
    for msgs in api.seen:
        assert msgs[0]["role"] == "system" and "No fluff. No intro. No outro." in msgs[0]["content"]


def test_reduced_message_is_what_the_api_receives_and_what_history_keeps(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "please explain the cache", route="api")
    send(svc, "and the eviction", route="api")
    assert user_msgs(api, 1)[0] == "explain cache" and user_msgs(api, 1)[-1] == "and eviction"


# -- direct answers: no model at all ---------------------------------------------------------------
def test_arithmetic_is_answered_without_any_model(tmp_path):
    local = RouterLocal()
    svc, api = make_service(tmp_path, local)
    out = send(svc, "what is 15*(3+4)?")
    assert (out.handled_by, out.reply) == ("tool", "105")
    assert api.seen == [] and local.classified == [] and local.calls == []


def test_direct_answer_can_be_turned_off_or_bypassed(tmp_path):
    svc, api = make_service(tmp_path, tools={"direct_answers": False})
    assert send(svc, "2+2", route="api").handled_by == "api"
    svc, api = make_service(tmp_path / "b")
    assert send(svc, "2+2", route="api").handled_by == "api"


def test_direct_answer_is_visible_to_the_api_model_later(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "12*12")
    send(svc, "what was that number in words", route="api")
    assert {"role": "assistant", "content": f"{LOCAL_TAG} 144"} in api.seen[-1]


# -- routing -----------------------------------------------------------------------------------------
def test_easy_task_is_done_locally_and_api_is_not_called(tmp_path):
    svc, api = make_service(tmp_path, RouterLocal(answer="Paris"))
    out = send(svc, "capital of france")
    assert out.handled_by == "local" and out.reply == "Paris" and out.saver is None
    assert out.input_tokens == 0 and out.local_tokens == {"input": 7, "output": 2} and api.seen == []


def test_api_sees_local_turns_tagged_plus_a_notice(tmp_path):
    local = RouterLocal(answer="Paris")
    svc, api = make_service(tmp_path, local)
    send(svc, "capital of france")
    local.route_reply = '{"route": "api", "reason": "needs reasoning"}'
    send(svc, "now explain why")
    msgs = api.seen[-1]
    assert API_NOTICE in msgs[0]["content"]
    assert {"role": "user", "content": "capital of france"} in msgs
    assert {"role": "assistant", "content": f"{LOCAL_TAG} Paris"} in msgs


def test_no_notice_without_local_turns(tmp_path):
    svc, api = make_service(tmp_path, RouterLocal(route='{"route": "api", "reason": "hard"}'))
    send(svc, "explain quantum field theory")
    assert API_NOTICE not in api.seen[-1][0]["content"] and LOCAL_TAG not in str(api.seen[-1])


def test_local_escalation_goes_to_api_and_history_stays_clean(tmp_path):
    svc, api = make_service(tmp_path, RouterLocal(answer="ESCALATE"))
    out = send(svc, "capital of france")
    assert out.handled_by == "api" and out.route_reason == "local model asked to escalate"
    assert LOCAL_TAG not in str(api.seen[-1])


def test_forced_local_that_declines_is_an_error(tmp_path):
    svc, api = make_service(tmp_path, RouterLocal(answer="ESCALATE"))
    with pytest.raises(UserInputError, match="could not answer"):
        send(svc, "hard", route="local")
    assert api.seen == []


def test_routing_off_never_asks_the_local_model_to_classify(tmp_path):
    local = RouterLocal()
    svc, api = make_service(tmp_path, local, task_routing={"enabled": False})
    assert send(svc, "capital of france").handled_by == "api" and local.classified == []


# -- context paging -----------------------------------------------------------------------------------------
def test_only_the_last_hot_turns_are_sent_and_a_notice_says_so(tmp_path):
    svc, api = make_service(tmp_path, context={"hot_turns": 2, "recall_turns": 0})
    for i in range(5):
        out = send(svc, f"question number {i} about widgets", **API)
    sent_users = user_msgs(api)
    assert len(sent_users) == 3  # two hot turns + the new message
    assert out.context["turns_sent"] == 2 and out.context["turns_total"] == 4
    assert "Older turns may be omitted" in api.seen[-1][0]["content"]
    assert out.context["tokens_saved"] > 0


def test_a_relevant_old_turn_is_paged_back_in(tmp_path):
    svc, api = make_service(tmp_path, context={"hot_turns": 1, "recall_turns": 1})
    send(svc, "our kubernetes deployment yaml uses port 8443", **API)
    send(svc, "unrelated chat about lunch options", **API)
    send(svc, "more unrelated chat about weekend hiking plans", **API)
    out = send(svc, "what port did the kubernetes deployment yaml use", **API)
    assert out.context["recalled"] == 1
    assert any("8443" in c for c in user_msgs(api))


def test_paging_can_be_disabled(tmp_path):
    svc, api = make_service(tmp_path, context={"enabled": False})
    for i in range(9):
        send(svc, f"message {i} about gadgets", **API)
    assert len(user_msgs(api)) == 9


# -- documents (storage manager + vector store) ------------------------------------------------------------------
BIG = "\n\n".join(f"## Section {i}\n\n" + (f"Topic {i} details about alpha{i} and beta{i}. " * 30) for i in range(12))


def test_small_document_is_inline_and_history_keeps_only_a_stub(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "summarise", [doc("brief.md", "Quarterly revenue was 42 million.")], **API)
    assert "Quarterly revenue was 42 million." in user_msgs(api)[-1]
    send(svc, "thanks", **API)
    later = " ".join(user_msgs(api))
    assert later.count("Quarterly revenue was 42 million.") == 1 or "42 million" not in user_msgs(api)[0]
    assert "[file: brief.md | " in user_msgs(api)[0] and "indexed" in user_msgs(api)[0]


def test_big_document_gets_a_preview_and_outline_then_only_relevant_parts(tmp_path):
    svc, api = make_service(tmp_path, documents={"inline_max_tokens": 300, "preview_tokens": 150, "chunk_tokens": 200})
    out = send(svc, "what is in it", [doc("report.md", BIG)], **API)
    first = user_msgs(api)[-1]
    assert "Preview only" in first and "Outline:" in first and "Section 5" not in first.split("Preview only")[0]
    assert len(first) < len(BIG) / 3
    out = send(svc, "tell me about alpha7 and beta7", **API)
    sent = user_msgs(api)[-1]
    assert "<documents>" in sent and "alpha7" in sent and "alpha2 " not in sent
    assert out.documents["chunks"] >= 1 and out.documents["tokens"] < out.documents["indexed_tokens"]
    assert "Attached files are indexed" in api.seen[-1][0]["content"]


def test_documents_are_removed_when_the_session_ends(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "read", [doc("a.md", "alpha beta gamma")], **API)
    sess = svc.sessions.get("session-1")
    assert svc.storage.list_keys(sess.agent_id)
    svc.reset("session-1")
    assert svc.storage.list_keys(sess.agent_id) == []


# -- memory (memory manager) ----------------------------------------------------------------------------------------
def test_memory_from_an_earlier_session_is_recalled_in_a_new_one(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "my dog is called Biscuit and he hates thunderstorms", sid="one", **API)
    out = send(svc, "what should I do about my dog thunderstorms", sid="two", **API)
    assert out.memory == {"notes": 1, "tokens": out.memory["tokens"]}
    assert "<memory>" in user_msgs(api)[-1] and "Biscuit" in user_msgs(api)[-1]


def test_memory_never_recalls_the_current_session_and_is_not_stored_in_history(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "my dog is called Biscuit and he hates thunderstorms", sid="one", **API)
    out = send(svc, "tell me more about my dog and thunderstorms", sid="one", **API)
    assert out.memory is None and "<memory>" not in user_msgs(api)[-1]
    sess = svc.sessions.get("one")
    assert all("<memory>" not in t.user for t in sess.turns)


def test_secrets_and_short_messages_are_not_remembered(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "my api key is sk-abcdefghijklmnopqrstuvwxyz", **API)
    send(svc, "hi there", **API)
    assert svc.memory_count() == 0


def test_memory_can_be_turned_off_and_cleared(tmp_path):
    svc, api = make_service(tmp_path, memory={"enabled": False})
    send(svc, "my dog is called Biscuit and he hates thunderstorms", **API)
    assert svc.memory_count() == 0
    svc, api = make_service(tmp_path / "b")
    send(svc, "my dog is called Biscuit and he hates thunderstorms", **API)
    assert svc.memory_count() == 1 and svc.clear_memory() == 1 and svc.memory_count() == 0


def test_memory_survives_a_restart(tmp_path):
    svc, _ = make_service(tmp_path)
    send(svc, "my dog is called Biscuit and he hates thunderstorms", sid="one", **API)
    svc2, api2 = make_service(tmp_path)  # same storage_dir, fresh process state
    out = send(svc2, "what should I do about my dog thunderstorms", sid="two", **API)
    assert out.memory and out.memory["notes"] == 1


# -- tools (tool manager) -------------------------------------------------------------------------------------------------
def call(tool, /, **args):
    import json
    return f'<tool>{json.dumps({"name": tool, "args": args})}</tool>'


def test_model_can_call_the_calculator_and_then_answers(tmp_path):
    api = FakeAPI([call("calculator", expression="1234*5678"), "It is 7006652."])
    svc, _ = make_service(tmp_path, api=api)
    out = send(svc, "multiply 1234 by 5678 and explain", **API)
    assert out.reply == "It is 7006652." and out.tools == [{"name": "calculator", "ok": True, "seconds": out.tools[0]["seconds"]}]
    result_msg = api.seen[1][-1]
    assert result_msg["role"] == "user" and '<tool_result name="calculator">' in result_msg["content"]
    assert "7006652" in result_msg["content"] and len(api.seen) == 2
    assert "Tools: to call one" in api.seen[0][0]["content"]


def test_tool_prompt_costs_nothing_when_no_tool_can_help(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "explain caching", **API)
    assert "Tools:" not in api.seen[-1][0]["content"]


def test_bad_tool_calls_become_error_results_not_crashes(tmp_path):
    api = FakeAPI(['<tool>{oops</tool>', call("nope"), call("calculator", expression="__import__('os')"), "done"])
    svc, _ = make_service(tmp_path, api=api)
    out = send(svc, "compute 5 things", **API)
    assert out.reply == "done" and [t["ok"] for t in out.tools] == [False, False, False]


def test_tool_rounds_are_capped_and_the_last_call_is_stripped(tmp_path):
    api = FakeAPI([call("calculator", expression="1+1")] * 10)
    svc, _ = make_service(tmp_path, api=api, tools={"max_rounds": 2})
    out = send(svc, "keep calculating 1 plus 1", **API)
    assert len(api.seen) == 3 and len(out.tools) == 2 and "<tool>" not in out.reply


def test_document_tools_read_only_the_sessions_own_documents(tmp_path):
    api = FakeAPI([call("read_document", name="report.md", part=2), "ok"])
    svc, _ = make_service(tmp_path, api=api, documents={"inline_max_tokens": 300, "preview_tokens": 150, "chunk_tokens": 200})
    send(svc, "what is in it", [doc("report.md", BIG)], **API)
    api.seen.clear()
    api.replies = [call("read_document", name="report.md", part=2), "ok"]
    out = send(svc, "read part 2 of the report", **API)
    assert out.tools[0]["ok"] and "[report.md part 2/" in api.seen[1][-1]["content"]
    # another session cannot see it
    api.replies = [call("read_document", name="report.md", part=1), "ok"]
    out = send(svc, "read part 1 of the report", sid="other-session", **API)
    assert out.tools == [] or not out.tools[0]["ok"]


def test_tools_can_be_disabled(tmp_path):
    api = FakeAPI([call("calculator", expression="1+1")])
    svc, _ = make_service(tmp_path, api=api, tools={"enabled": False})
    out = send(svc, "compute 1 plus 1 please", **API)
    assert out.tools == [] and "Tools:" not in api.seen[0][0]["content"]


# -- errors ---------------------------------------------------------------------------------------------------------------------
def test_provider_failures_are_wrapped_and_never_leak_the_key(tmp_path):
    api = FakeAPI(boom=ConnectionError("401 for key sk-test-secret-key-000000"))
    svc, _ = make_service(tmp_path, api=api)
    with pytest.raises(ProviderError) as e:
        send(svc, "explain caching", **API)
    assert "sk-test-secret-key-000000" not in str(e.value) and "***" in str(e.value)


def test_bad_input_is_rejected_with_user_errors(tmp_path):
    svc, api = make_service(tmp_path)
    with pytest.raises(UserInputError, match="empty"):
        send(svc, "  ")
    with pytest.raises(UserInputError, match="invalid session id"):
        send(svc, "hi", sid="../../etc")
    with pytest.raises(UserInputError, match="unsupported provider"):
        from pal.chat import build_api_core
        build_api_core("nope", "m", "k")


def test_a_failed_turn_leaves_history_untouched_and_is_counted(tmp_path):
    api = FakeAPI(boom=ConnectionError("down"))
    svc, _ = make_service(tmp_path, api=api)
    with pytest.raises(ProviderError):
        send(svc, "explain caching", **API)
    assert svc.sessions.get("session-1").turns == []
    assert svc.metrics.chat_summary()["errors"] == 1


# -- metrics ----------------------------------------------------------------------------------------------------------------------
def test_metrics_record_route_latency_and_savings(tmp_path):
    svc, api = make_service(tmp_path, RouterLocal(answer="Paris"), context={"hot_turns": 1, "recall_turns": 0})
    send(svc, "12*12")                         # no model
    send(svc, "capital of france")             # local
    send(svc, "please explain the caching layer", **API)  # api
    send(svc, "please describe the eviction policy", **API)
    send(svc, "please describe the expiry policy", **API)
    m = svc.metrics.chat_summary()
    assert m["turns"] == 5 and m["api_calls_avoided"] == 2
    assert {r: m["by_route"][r]["turns"] for r in ("api", "local", "tool")} == {"api": 3, "local": 1, "tool": 1}
    assert m["api_input_tokens"] == 30 and m["tokens_saved_estimate"]["input_savers"] > 0
    assert m["tokens_saved_estimate"]["context_paging"] > 0
    assert m["by_route"]["api"]["p95_latency_sec"] >= m["by_route"]["api"]["p50_latency_sec"] >= 0


def test_metrics_can_be_disabled_and_reset(tmp_path):
    svc, api = make_service(tmp_path, metrics={"enabled": False})
    send(svc, "12*12")
    assert svc.metrics.chat_summary() == {"turns": 0}
    svc, api = make_service(tmp_path / "b")
    send(svc, "12*12")
    svc.metrics.reset()
    assert svc.metrics.chat_summary() == {"turns": 0}


# -- sessions -------------------------------------------------------------------------------------------------------------------------
def test_sessions_are_bounded_and_evicted_sessions_are_cleaned_up(tmp_path):
    svc, api = make_service(tmp_path, server={"max_sessions": 2})
    send(svc, "read", [doc("a.md", "alpha beta gamma")], sid="s1", **API)
    agent = svc.sessions.get("s1").agent_id
    send(svc, "hello there friend", sid="s2", **API)
    send(svc, "hello there again", sid="s3", **API)
    assert len(svc.sessions) == 2 and svc.sessions.get("s1") is None
    assert svc.storage.list_keys(agent) == []


def test_idle_sessions_expire(tmp_path):
    svc, api = make_service(tmp_path)
    send(svc, "hello there friend", sid="old", **API)
    svc.sessions.get("old").last_used -= 10 * 3600
    send(svc, "hello there again", sid="new", **API)
    assert svc.sessions.get("old") is None


def test_concurrent_sessions_do_not_interfere(tmp_path):
    svc, api = make_service(tmp_path)
    errors, results = [], {}

    def run(i):
        try:
            for j in range(4):
                results[(i, j)] = send(svc, f"thread {i} message {j} about topic{i}", sid=f"s{i}", **API)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(results) == 24
    assert all(len(svc.sessions.get(f"s{i}").turns) == 4 for i in range(6))
    assert svc.metrics.chat_summary()["turns"] == 24


def test_same_session_turns_are_serialised(tmp_path):
    svc, api = make_service(tmp_path)
    threads = [threading.Thread(target=lambda k=k: send(svc, f"message {k} about gadgets", **API)) for k in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(svc.sessions.get("session-1").turns) == 8


# -- experiment (compare raw vs optimised) ---------------------------------------------------------------------------
def test_experiment_compares_raw_and_optimised_api_calls(tmp_path):
    svc, api = make_service(tmp_path)
    out = svc.experiment("please explain the cache", None, "anthropic", "m", "k", call_api=True, mode="rules")
    assert out["api"]["raw"]["reply"] == "api reply" and len(api.seen) == 2
    assert "No fluff" not in api.seen[0][0]["content"] and "No fluff" in api.seen[1][0]["content"]
    assert out["counter"] == "heuristic (chars/4)"


def test_experiment_without_a_model_still_measures_the_savers(tmp_path):
    svc, api = make_service(tmp_path)
    out = svc.experiment("please explain the cache", None, mode="rules")
    assert out["api"] is None and out["saver"].tokens_saved >= 0 and api.seen == []
    with pytest.raises(UserInputError, match="choose a provider"):
        svc.experiment("hello", None, call_api=True, mode="rules")
