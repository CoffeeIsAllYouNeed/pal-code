import io
import shutil

import pytest
from tests.fakes import RouterLocal
from tests.helpers import client, make_service
from tests.helpers import send as _send

from pal.chat import ChatService
from pal.task_router import API_NOTICE, LOCAL_TAG, TaskRouter


@pytest.fixture
def make(tmp_path):
    """make(local, routing) -> (service, fake api, fake local). Direct answers are off so simple
    arithmetic still reaches the router, which is what these tests are about."""
    def _make(local=None, routing=None):
        local = local or RouterLocal()
        svc, api = make_service(tmp_path, local, task_routing=routing or {}, tools={"direct_answers": False})
        return svc, api, local
    return _make


def send(svc, msg, atts=None, route=None, sid="s"):
    return _send(svc, msg, atts, route=route, sid=sid)


# -- gates: obvious hard input never reaches the classifier ---------------------------
@pytest.mark.parametrize("msg,atts", [
    ("what is 2+2", [{"name": "a.txt", "text": "x", "kind": "document"}]),
    ("fix this:\n```py\nprint(1)\n```", []),
    ("word " * 200, []),
    ("\n".join(["line"] * 10), []),
    ("   ", []),
])
def test_gates_send_hard_input_to_api_without_calling_the_classifier(make, msg, atts):
    svc, api, local = make()
    if msg.strip():
        out = send(svc, msg, atts)
        assert out.handled_by == "api" and api.seen
        assert local.classified == []


# -- local route -----------------------------------------------------------------------
def test_easy_task_is_done_locally_and_api_is_not_called(make):
    svc, api, local = make()
    out = send(svc, "what is 2+2")
    assert out.handled_by == "local" and out.reply == "4"
    assert out.saver is None and out.input_tokens == 0 and out.output_tokens == 0
    assert out.route_reason == "simple math" and out.local_tokens == {"input": 7, "output": 2}
    assert api.seen == []


def test_local_turn_needs_no_api_key(make):
    svc, api, _ = make()
    svc.api_core = ChatService.api_core.__get__(svc)  # the real one: raises without a key
    out = svc.send("s", "", "", None, "what is 2+2", [], None, mode="rules")
    assert out.handled_by == "local"


# -- API agents are told what the local LLM did ------------------------------------------
def test_api_sees_local_turns_tagged_plus_a_notice_on_the_next_api_request(make):
    svc, api, local = make()
    send(svc, "what is 2+2")
    local.route_reply = '{"route": "api", "reason": "needs reasoning"}'
    send(svc, "now explain why")
    msgs = api.seen[-1]
    assert API_NOTICE in msgs[0]["content"]
    assert {"role": "user", "content": "what is 2+2"} in msgs
    assert {"role": "assistant", "content": f"{LOCAL_TAG} 4"} in msgs
    assert msgs[-1]["role"] == "user"


def test_no_notice_when_the_session_has_no_local_turns(make):
    svc, api, local = make(RouterLocal(route='{"route": "api", "reason": "hard"}'))
    send(svc, "explain quantum field theory")
    assert API_NOTICE not in api.seen[-1][0]["content"] and LOCAL_TAG not in str(api.seen[-1])


def test_converted_attachment_carries_a_note_for_the_api_model(make):
    svc, api, _ = make()
    out = send(svc, "summarise", [{"name": "b.docx", "text": "# T", "note": "converted from docx to markdown", "kind": "document"},
                                  {"name": "n.txt", "text": "plain", "kind": "document"}])
    assert "[file: b.docx | converted from docx to markdown]\n# T" in out.sent
    assert "[file: n.txt]\nplain" in out.sent


# -- fail-safe: anything unclear goes to the API model ---------------------------------------
@pytest.mark.parametrize("local", [
    RouterLocal(route="I think local"), RouterLocal(route='{"route": "maybe"}'),
    RouterLocal(route="{not json}"), RouterLocal(boom=True),
])
def test_unclear_or_failed_classification_goes_to_api(make, local):
    local.route_boom = local.route_boom or False
    svc, api, _ = make(local)
    out = send(svc, "what is 2+2")
    assert out.handled_by == "api" and api.seen


def test_local_model_can_escalate_and_history_stays_clean(make):
    svc, api, local = make(RouterLocal(answer="ESCALATE"))
    out = send(svc, "what is 2+2")
    assert out.handled_by == "api" and out.route_reason == "local model asked to escalate"
    assert LOCAL_TAG not in str(api.seen[-1]) and API_NOTICE not in api.seen[-1][0]["content"]


def test_empty_local_answer_escalates(make):
    svc, api, _ = make(RouterLocal(answer="   "))
    assert send(svc, "what is 2+2").handled_by == "api"


# -- overrides and config ------------------------------------------------------------------------
def test_api_override_skips_the_classifier(make):
    svc, api, local = make()
    assert send(svc, "what is 2+2", route="api").handled_by == "api"
    assert local.classified == [] and api.seen


def test_local_override_answers_locally_even_for_hard_input(make):
    svc, api, local = make()
    out = send(svc, "word " * 200, route="local")
    assert out.handled_by == "local" and api.seen == [] and local.classified == []


def test_local_override_with_attachment_shows_the_file_to_the_local_model(make):
    svc, api, local = make()
    send(svc, "what is in it", [{"name": "n.txt", "text": "secret body", "kind": "document"}], route="local")
    assert "[file: n.txt]\nsecret body" in local.answered[-1][-1]["content"]


def test_local_override_that_escalates_is_an_error_not_a_silent_api_call(make):
    svc, api, _ = make(RouterLocal(answer="ESCALATE"))
    with pytest.raises(ValueError, match="could not answer"):
        send(svc, "hard", route="local")
    assert api.seen == []


def test_routing_disabled_always_uses_api_and_never_asks_the_local_model(make):
    svc, api, local = make(routing={"enabled": False})
    out = send(svc, "what is 2+2")
    assert out.handled_by == "api" and local.classified == []


def test_config_default_local_applies_when_no_override(make):
    svc, api, _ = make(routing={"default": "local"})
    assert send(svc, "word " * 200).handled_by == "local"


def test_local_model_sees_recent_history_without_the_tag(make):
    svc, api, local = make()
    send(svc, "what is 2+2")
    send(svc, "and doubled?")
    prior = [m["content"] for m in local.answered[-1]]
    assert "4" in prior and not any(LOCAL_TAG in c for c in prior)


def test_router_decide_is_pure_for_gates():
    r = TaskRouter(RouterLocal(), {"max_local_chars": 10})
    assert r.decide("x" * 11, []).source == "gate"


# -- web endpoint ---------------------------------------------------------------------------------------
def test_flask_chat_reports_who_answered(make):
    svc, api, _ = make()
    c = client(svc)
    r = c.post("/api/chat", data={"session_id": "x", "message": "what is 2+2"})
    d = r.get_json()
    assert r.status_code == 200 and d["handled_by"] == "local" and d["saver"] is None and d["reply"] == "4"
    r = c.post("/api/chat", data={"session_id": "y", "provider": "anthropic", "model": "m", "api_key": "k",
                                  "message": "what is 2+2", "route": "api"})
    assert r.get_json()["handled_by"] == "api" and r.get_json()["saver"]["steps"]
    assert c.get("/api/config").get_json()["routing"] == {"enabled": True, "default": "auto"}


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_flask_upload_of_converted_file_tells_the_api_model(make):
    import subprocess
    svc, api, _ = make()
    docx = subprocess.run(["pandoc", "-f", "markdown", "-t", "docx", "-o", "-"], input=b"# Hi\n\nBody",
                          capture_output=True, check=True).stdout
    r = client(svc).post("/api/chat", data={
        "session_id": "z", "provider": "anthropic", "model": "m", "api_key": "k", "message": "read it",
        "files": (io.BytesIO(docx), "brief.docx")})
    assert "[file: brief.docx | converted from docx to markdown]" in r.get_json()["sent"]
