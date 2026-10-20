"""Local runtimes through LiteLLM. The first half uses a fake `litellm` module (always runs). The second half runs the REAL pinned
LiteLLM against fake Ollama and OpenAI-style servers on this machine, and skips itself when LiteLLM is not installed."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from pal.llm_wrapper import DeploymentType, LLMAdapter, LLMSyscall, WrapperConfig, build_wrapper
from pal.llm_wrapper.components import litellm_wrapper
from pal.llm_wrapper.components.litellm_wrapper import LOCAL_PREFIX, close_stream, full_model

MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "weather in Pune?"}]


# -- fake-module tests -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend,model,want", [
    ("ollama", "llama3", "ollama_chat/llama3"), ("vllm", "my-model", "hosted_vllm/my-model"),
    ("lmstudio", "qwen", "lm_studio/qwen"), ("llamacpp", "local", "openai/local")])
def test_local_runtimes_get_litellms_provider_prefix(backend, model, want):
    assert full_model(backend, model) == want and set(LOCAL_PREFIX) == {"ollama", "vllm", "lmstudio", "llamacpp"}


def test_a_local_request_has_no_key_uses_the_host_and_waits_longer_for_the_first_call(fake_litellm):
    core = build_wrapper(WrapperConfig(backend="ollama", model="llama3", host="http://gpu-box:11434"))
    core.chat(LLMSyscall(messages=MSGS, temperature=0.1, max_tokens=40))
    kw = fake_litellm.calls[0]
    assert kw["model"] == "ollama_chat/llama3" and kw["api_base"] == "http://gpu-box:11434" and "api_key" not in kw
    assert kw["timeout"] == 300 and kw["num_retries"] == 0 and (kw["temperature"], kw["max_tokens"]) == (0.1, 40)
    cloud = build_wrapper(WrapperConfig(backend="anthropic", model="claude-x", api_key="k"))
    assert cloud.timeout == 120 and cloud.local is False


def test_a_local_model_is_free_local_and_can_embed(fake_litellm):
    core = build_wrapper(WrapperConfig(backend="vllm", model="m", host="http://localhost:8000/v1"))
    caps = core.capabilities()
    assert (caps.deployment_type, caps.billed_per_token, caps.supports_embeddings) == (DeploymentType.LOCAL_SERVER, False, True)
    assert core.price_per_token() is None
    calls = []
    fake_litellm.embedding = lambda **kw: calls.append(kw) or SimpleNamespace(data=[{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}])
    assert core.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert calls[0]["api_base"] == "http://localhost:8000/v1" and calls[0]["input"] == ["a", "b"]
    assert calls[0]["api_key"] == "not-needed"                                  # a placeholder, never a real key from the environment


def test_extra_litellm_parameters_such_as_keep_alive_reach_the_call(fake_litellm):
    core = build_wrapper(WrapperConfig(backend="ollama", model="llama3", extra={"litellm_params": {"keep_alive": "30m"}, "timeout": 600}))
    core.chat(LLMSyscall(messages=MSGS))
    assert fake_litellm.calls[0]["keep_alive"] == "30m" and fake_litellm.calls[0]["timeout"] == 600


def test_the_adapter_never_hands_a_global_key_to_a_local_runtime(fake_litellm):
    a = LLMAdapter([{"name": "llama3", "backend": "ollama"}, {"name": "m", "backend": "vllm"}, {"name": "c", "backend": "anthropic"}],
                   api_key="shared")
    assert [c.api_key for c in a.llm_configs] == [None, None, "shared"]


class Closable:
    def __init__(self, fail=False):
        self.closed, self.fail = 0, fail

    def close(self):
        self.closed += 1
        if self.fail:
            raise RuntimeError("already closed")


def test_closing_a_stream_closes_everything_underneath_litellms_wrapper_and_tolerates_failures():
    handles = {n: Closable() for n in ("streaming_response", "response_iterator", "http_response")}
    inner = SimpleNamespace(**handles)
    wrapper = SimpleNamespace(completion_stream=inner)               # like LiteLLM's wrapper: no close() of its own
    close_stream(wrapper)
    assert all(h.closed == 1 for h in handles.values())
    nested = SimpleNamespace(completion_stream=SimpleNamespace(streaming_response=Closable(fail=True), response_iterator=Closable()))
    close_stream(nested)                                              # one handle raises: the others are still closed
    assert nested.completion_stream.response_iterator.closed == 1
    own = Closable()
    close_stream(own)                                                 # an object with its own close() (an openai Stream)
    assert own.closed == 1
    close_stream(object())                                            # nothing to close: no error
    close_stream(None)


def test_stopping_a_stream_early_closes_it_so_the_server_stops(fake_litellm):
    streams = []

    class Wrapper:
        def __init__(self):
            self.completion_stream = SimpleNamespace(streaming_response=Closable(), response_iterator=Closable())

        def __iter__(self):
            for piece in ("a ", "b ", "c "):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
    fake_litellm.completion = lambda **kw: streams.append(Wrapper()) or streams[-1]
    core = build_wrapper(WrapperConfig(backend="ollama", model="llama3"))
    gen = core.stream_chat(LLMSyscall(messages=MSGS))
    assert next(gen) == "a "
    gen.close()                                                       # the scheduler pausing the job
    assert streams[0].completion_stream.streaming_response.closed == 1 and streams[0].completion_stream.response_iterator.closed == 1


# -- the REAL pinned LiteLLM against fake local servers (skips without it) ---------------------------------------------------
class Server:
    """A fake Ollama (/api/chat, newline-delimited JSON) or OpenAI-style (/v1/chat/completions, SSE) server that streams slowly and
    records whether the client went away."""

    def __init__(self, kind):
        self.kind, self.log = kind, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.log.append(("request", self.path, body.get("stream"), self.headers.get("Authorization")))
                if body.get("stream"):
                    return self._stream()
                text = "plain reply"
                data = (json.dumps({"model": "m", "message": {"role": "assistant", "content": text}, "done": True,
                                    "prompt_eval_count": 7, "eval_count": 2}) if outer.kind == "ollama" else json.dumps(
                    {"id": "x", "object": "chat.completion", "created": 1, "model": "m", "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}})).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return None

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson" if outer.kind == "ollama" else "text/event-stream")
                self.end_headers()
                try:
                    for i in range(60):
                        piece = f"w{i} "
                        line = (json.dumps({"model": "m", "message": {"role": "assistant", "content": piece}, "done": False}) + "\n"
                                if outer.kind == "ollama" else "data: " + json.dumps(
                                    {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "m",
                                     "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}) + "\n\n")
                        self.wfile.write(line.encode())
                        self.wfile.flush()
                        outer.log.append(("sent", i))
                        time.sleep(0.03)
                    outer.log.append(("finished",))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    outer.log.append(("client disconnected",))
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}" + ("" if kind == "ollama" else "/v1")

    def chunks_sent(self):
        return len([e for e in self.log if e[0] == "sent"])

    def ended_by(self):
        return [e[0] for e in self.log if e[0] in ("finished", "client disconnected")]

    def stop(self):
        self.http.shutdown()


@pytest.fixture
def real_litellm():
    pytest.importorskip("litellm")
    litellm_wrapper._module = None
    yield litellm_wrapper._import_litellm()
    litellm_wrapper._module = None


RUNTIMES = [("ollama", "ollama", "llama3"), ("vllm", "openai", "my-model"), ("lmstudio", "openai", "qwen"), ("llamacpp", "openai", "local")]


def test_a_real_openai_key_in_the_environment_is_never_sent_to_a_local_server(real_litellm, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-that-must-not-leak")
    for backend in ("llamacpp", "vllm", "lmstudio"):
        srv = Server("openai")
        try:
            build_wrapper(WrapperConfig(backend=backend, model="m", host=srv.url)).chat(LLMSyscall(messages=MSGS))
            auth = srv.log[0][3]
            assert auth == "Bearer not-needed" and "sk-real" not in str(auth)
        finally:
            srv.stop()
    configured = Server("openai")
    try:                                                                         # a key you DID configure for your server is used
        build_wrapper(WrapperConfig(backend="vllm", model="m", host=configured.url, api_key="my-vllm-token")).chat(LLMSyscall(messages=MSGS))
        assert configured.log[0][3] == "Bearer my-vllm-token"
    finally:
        configured.stop()


@pytest.mark.parametrize("backend,kind,model", RUNTIMES)
def test_real_litellm_chats_with_every_local_runtime_and_reports_tokens(real_litellm, backend, kind, model):
    srv = Server(kind)
    try:
        core = build_wrapper(WrapperConfig(backend=backend, model=model, host=srv.url))
        r = core.chat(LLMSyscall(messages=MSGS, max_tokens=20))
        assert (r.text, r.prompt_tokens, r.completion_tokens) == ("plain reply", 7, 2) and r.meta["cost_usd"] in (None, 0, 0.0)
        assert srv.log[0][1].endswith("/api/chat" if kind == "ollama" else "/chat/completions")
    finally:
        srv.stop()


@pytest.mark.parametrize("backend,kind,model", RUNTIMES[:2])
def test_real_litellm_streams_to_the_end(real_litellm, backend, kind, model):
    srv = Server(kind)
    try:
        core = build_wrapper(WrapperConfig(backend=backend, model=model, host=srv.url))
        text = "".join(core.stream_chat(LLMSyscall(messages=MSGS)))
        assert text.startswith("w0 w1 ") and text.endswith("w59 ")
        assert srv.ended_by() == ["finished"]
    finally:
        srv.stop()


@pytest.mark.parametrize("backend,kind,model", RUNTIMES[:2])
def test_real_litellm_stopping_a_stream_early_really_stops_the_model_server(real_litellm, backend, kind, model):
    """The scheduler's pause depends on this: a paused answer must not keep the model busy."""
    srv = Server(kind)
    try:
        core = build_wrapper(WrapperConfig(backend=backend, model=model, host=srv.url))
        gen = core.stream_chat(LLMSyscall(messages=MSGS))
        assert next(gen) == "w0 " and next(gen) == "w1 "
        gen.close()
        end = time.time() + 3
        while time.time() < end and not srv.ended_by():
            time.sleep(0.05)
        assert srv.ended_by() == ["client disconnected"] and srv.chunks_sent() < 20       # it stopped long before 60 chunks
        sent = srv.chunks_sent()
        time.sleep(0.3)
        assert srv.chunks_sent() == sent                                                    # and stays stopped
    finally:
        srv.stop()


def test_real_litellm_scheduler_pause_and_resume_work_through_a_local_runtime(real_litellm):
    """An AGENT job arrives while a CHAT answer streams from a (fake) Ollama: the answer is stopped, the agent job runs, and the
    answer continues from its saved text."""
    from pal.llm_scheduler import LocalScheduler, TaskClass
    srv = Server("ollama")
    sched = LocalScheduler()
    try:
        core = build_wrapper(WrapperConfig(backend="ollama", model="llama3", host=srv.url))
        log = []
        ticket = sched.submit_stream(core, LLMSyscall(messages=MSGS), TaskClass.CHAT, "answer")
        end = time.time() + 5
        while srv.chunks_sent() < 3 and time.time() < end:
            time.sleep(0.02)
        agent = sched.submit(lambda: log.append("agent ran"), TaskClass.AGENT, "agent")
        agent.result(5)
        text = ticket.result(15)
        assert log == ["agent ran"] and ticket.pauses == 1
        assert text.startswith("w0 w1 w2 ") and "disconnected" in " ".join(srv.ended_by())     # the first stream was cut
        requests = [e for e in srv.log if e[0] == "request"]
        assert len(requests) == 2                                                               # the answer was resumed with a 2nd request
    finally:
        sched.close()
        srv.stop()


def test_real_litellm_model_strings_are_accepted_for_the_local_runtimes(real_litellm):
    for backend, model in (("ollama", "llama3"), ("vllm", "m"), ("lmstudio", "m"), ("llamacpp", "m")):
        core = build_wrapper(WrapperConfig(backend=backend, model=model, host="http://127.0.0.1:9"))
        assert core.model == full_model(backend, model) and core.capabilities().deployment_type is DeploymentType.LOCAL_SERVER
