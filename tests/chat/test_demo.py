import pytest
from tests.fakes import FakeAPI, RouterLocal
from tests.helpers import client, make_service

from pal.chat.demo import STEPS, SURVEY, SURVEY_NAME, Demo
from pal.errors import NotFoundError, ProviderError, UserInputError
from pal.llm_wrapper.components.base import LLMResponse


class SizedAPI(FakeAPI):
    """Token counts follow the prompt size (chars/4), so bigger prompts really cost more."""

    def chat(self, syscall):
        self.seen.append(syscall.messages)
        if self.boom:
            raise self.boom
        n = sum(len(m["content"]) for m in syscall.messages) // 4
        long_reply = not any("Output short" in m["content"] for m in syscall.messages if m["role"] == "system")
        reply = "word " * (120 if long_reply else 30)
        return LLMResponse(text=reply, prompt_tokens=n, completion_tokens=len(reply) // 4)


def make(tmp_path, api=None, local=None, **kw):
    svc, api = make_service(tmp_path, local or RouterLocal(route='{"route": "local", "reason": "easy"}', answer="Return on investment"),
                            api=api or SizedAPI(), **kw)
    svc.local_status = lambda: {"ok": True, "detail": "llama3 ready"}
    return svc, api, Demo(svc)


def run_all(demo):
    info = demo.start("anthropic", "m", "k")
    out = None
    for _ in STEPS:
        out = demo.step(info["run_id"], "k")
    return info, out


def test_script_has_six_messages_one_attachment_one_direct_answer():
    assert len(STEPS) == 6 and sum(1 for s in STEPS if s.get("attach")) == 1
    assert any(s["message"] == "what is 500*0.35" for s in STEPS)


def test_start_reports_the_script_and_local_status(tmp_path):
    svc, api, demo = make(tmp_path)
    info = demo.start("anthropic", "m", "k")
    assert len(info["steps"]) == 6 and info["steps"][1]["attachment"] == SURVEY_NAME and info["local"]["ok"]


def test_a_full_run_measures_both_arms_and_saves_tokens(tmp_path):
    svc, api, demo = make(tmp_path)
    info, out = run_all(demo)
    assert out["done"] and out["row"]["step"] == 6
    s = out["summary"]
    assert s["naive"]["api_calls"] == 6 and s["pal"]["api_calls"] < 6
    assert s["pal"]["direct_answers"] == 1 and s["pal"]["local_answers"] == 1 and s["api_calls_avoided"] == 2
    assert s["tokens_saved"] == s["naive"]["total_tokens"] - s["pal"]["total_tokens"] > 0
    assert 0 < s["tokens_saved_pct"] < 100 and s["input_saved"] > 0 and s["output_saved"] > 0
    assert s["result"] and s["result_naive"] and s["local"]["ok"]


def test_the_naive_arm_resends_everything_and_pal_does_not(tmp_path):
    svc, api, demo = make(tmp_path)
    run_all(demo)
    naive_calls = [m for m in api.seen if not any("Output short" in x["content"] for x in m if x["role"] == "system")]
    pal_calls = [m for m in api.seen if any("Output short" in x["content"] for x in m if x["role"] == "system")]
    assert len(naive_calls) == 6
    # naive: the whole survey is in the request on every turn after the upload
    assert all(SURVEY[:200] in " ".join(x["content"] for x in m) for m in naive_calls[1:])
    # pal: the survey body appears only on the upload turn, later turns carry a stub or excerpts
    with_full = [m for m in pal_calls if SURVEY in " ".join(x["content"] for x in m)]
    assert len(with_full) == 1 and len(pal_calls) < 6


def test_demo_turns_leave_no_memory_no_stats_and_no_session(tmp_path):
    svc, api, demo = make(tmp_path)
    run_all(demo)
    assert svc.memory_count() == 0 and svc.metrics.chat_summary() == {"turns": 0} and len(svc.sessions) == 0


def test_local_model_down_is_reported_honestly(tmp_path):
    svc, api, demo = make(tmp_path)
    svc.local_status = lambda: {"ok": False, "detail": "Ollama unreachable: ConnectionError"}
    _, out = run_all(demo)
    assert any("not reachable" in n for n in out["summary"]["notes"])


def test_steps_must_be_run_in_order_and_only_once(tmp_path):
    svc, api, demo = make(tmp_path)
    info, _ = run_all(demo)
    with pytest.raises(NotFoundError):          # finished runs are cleaned up
        demo.step(info["run_id"], "k")
    with pytest.raises(NotFoundError):
        demo.step("nope", "k")


def test_a_failed_step_can_be_retried_without_double_counting(tmp_path):
    svc, api, demo = make(tmp_path)
    info = demo.start("anthropic", "m", "k")
    demo.step(info["run_id"], "k")
    api.boom = ConnectionError("network down key k")
    with pytest.raises(ProviderError):
        demo.step(info["run_id"], "k")
    api.boom = None
    row = demo.step(info["run_id"], "k")["row"]
    assert row["step"] == 2
    naive_history = demo._runs[info["run_id"]].naive_history
    assert [m["role"] for m in naive_history] == ["user", "assistant"] * 2


def test_start_validates_provider_and_key(tmp_path, monkeypatch):
    from pal.chat import ChatService
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    svc, api, demo = make(tmp_path)
    svc.api_core = ChatService.api_core.__get__(svc)
    with pytest.raises(UserInputError, match="no API key"):
        demo.start("anthropic", "m", None)
    with pytest.raises(UserInputError, match="unsupported provider"):
        demo.start("nope", "m", "k")


def test_runs_are_bounded(tmp_path):
    svc, api, demo = make(tmp_path)
    ids = [demo.start("anthropic", "m", "k")["run_id"] for _ in range(25)]
    assert len(demo._runs) == 20 and ids[0] not in demo._runs and ids[-1] in demo._runs


def test_http_flow(tmp_path):
    svc, api, demo = make(tmp_path)
    c = client(svc)
    body = {"provider": "anthropic", "model": "m", "api_key": "k"}
    info = c.post("/api/demo/start", json=body).get_json()
    last = None
    for _ in info["steps"]:
        r = c.post("/api/demo/step", json={"run_id": info["run_id"], "api_key": "k"})
        assert r.status_code == 200
        last = r.get_json()
    assert last["done"] and last["summary"]["tokens_saved"] != 0
    assert c.post("/api/demo/step", json={"run_id": info["run_id"]}).status_code == 404
    assert c.post("/api/demo/cancel", json={"run_id": "x"}).get_json() == {"ok": True}


def test_per_session_metrics_endpoint(tmp_path):
    svc, api, _ = make(tmp_path)
    c = client(svc)
    for sid in ("aaaaaaaa-1", "bbbbbbbb-2"):
        c.post("/api/chat", data={"session_id": sid, "provider": "anthropic", "model": "m", "api_key": "k",
                                  "message": "what is 2+2"})
    assert c.get("/api/metrics?session=aaaaaaaa-1").get_json()["summary"]["turns"] == 1
    assert c.get("/api/metrics").get_json()["summary"]["turns"] == 2
    assert c.get("/api/metrics?session=cccccccc").get_json()["summary"] == {"turns": 0}
