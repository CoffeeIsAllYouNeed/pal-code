import io
import json
from pathlib import Path

import pytest

from localrag.core.config import ServerConfig
from localrag.core.errors import LocalRAGError, ProtocolError
from localrag.mcp import jsonrpc
from localrag.mcp.capabilities import SUPPORTED_VERSIONS, negotiate_version
from localrag.mcp.factory import build_server
from localrag.mcp.middleware import RateLimiter, Timing, check_token
from localrag.mcp.registry import Prompt, PromptRegistry, ResourceRegistry, Tool, ToolRegistry
from localrag.mcp.schema import apply_defaults, validate
from localrag.mcp.server import McpServer
from localrag.mcp.transport_memory import MemoryClient
from localrag.mcp.transport_stdio import serve_stdio


# -- JSON-RPC ----------------------------------------------------------------------------------------------------------------------
def test_requests_notifications_and_responses_parse_and_round_trip():
    req = jsonrpc.parse_message('{"jsonrpc":"2.0","id":1,"method":"ping","params":{"a":1}}')
    note = jsonrpc.parse_message('{"jsonrpc":"2.0","method":"notifications/initialized"}')
    ok = jsonrpc.parse_message('{"jsonrpc":"2.0","id":"x","result":{"v":1}}')
    err = jsonrpc.parse_message('{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"nope","data":[1]}}')
    assert (req.id, req.method, req.params) == (1, "ping", {"a": 1}) and note.method == "notifications/initialized" and ok.result == {"v": 1}
    assert err.error.code == -32601 and err.error.data == [1]
    for m in (req, note, ok, err):
        assert jsonrpc.parse_message(jsonrpc.encode(m)) == m
    assert jsonrpc.to_object(jsonrpc.Request(3, "m")) == {"jsonrpc": "2.0", "id": 3, "method": "m"}
    assert jsonrpc.parse_message('[{"jsonrpc":"2.0","id":1,"method":"a"},{"jsonrpc":"2.0","method":"b"}]')[1].method == "b"


@pytest.mark.parametrize("raw,code", [("{bad", jsonrpc.PARSE_ERROR), ("[]", jsonrpc.INVALID_REQUEST), ('{"id":1,"method":"x"}', jsonrpc.INVALID_REQUEST),
                                      ('{"jsonrpc":"2.0","id":1,"method":5}', jsonrpc.INVALID_REQUEST), ('{"jsonrpc":"2.0","id":true,"method":"x"}', jsonrpc.INVALID_REQUEST),
                                      ('{"jsonrpc":"2.0","id":1,"method":"x","params":3}', jsonrpc.INVALID_PARAMS), ('{"jsonrpc":"2.0","id":1}', jsonrpc.INVALID_REQUEST),
                                      ("3", jsonrpc.INVALID_REQUEST)])
def test_malformed_messages_carry_the_right_error_code(raw, code):
    with pytest.raises(ProtocolError) as e:
        jsonrpc.parse_message(raw)
    assert e.value.code == code


# -- schema ------------------------------------------------------------------------------------------------------------------------
SCHEMA = {"type": "object", "required": ["q"], "additionalProperties": False, "properties": {
    "q": {"type": "string", "minLength": 2, "maxLength": 5, "pattern": "^[a-z]+$"}, "n": {"type": "integer", "minimum": 1, "maximum": 9, "default": 3},
    "mode": {"enum": ["a", "b"]}, "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2}, "x": {"type": ["number", "null"]}}}


def test_schema_accepts_valid_and_explains_every_kind_of_violation():
    assert validate({"q": "abc", "n": 5, "mode": "a", "tags": ["t"], "x": None}, SCHEMA) == []
    bad = validate({"n": 0, "q": "A", "mode": "z", "tags": [], "extra": 1}, SCHEMA)
    joined = " | ".join(bad + validate({}, SCHEMA))
    for part in ("missing required property 'q'", "at least 2 characters", "must match", "must be at least 1", "one of ['a', 'b']", "at least 1 items", "unexpected property 'extra'"):
        assert part in joined, part
    assert "at most 5 characters" in validate({"q": "abcdef"}, SCHEMA)[0] and "at most 9" in validate({"q": "ab", "n": 10}, SCHEMA)[0]
    assert "at most 2 items" in validate({"q": "ab", "tags": ["a", "b", "c"]}, SCHEMA)[0] and "$.tags[0]: expected string" in validate({"q": "ab", "tags": [1]}, SCHEMA)[0]


def test_booleans_are_not_numbers_and_types_are_checked_strictly():
    assert validate(True, {"type": "integer"}) and validate(1.5, {"type": "integer"}) and not validate(2, {"type": "number"}) and not validate(True, {"type": "boolean"})
    assert validate("1", {"type": "number"}) and validate({}, {"type": "array"}) and validate(None, {"type": "string"}) == ["$: expected string, got NoneType"]
    assert apply_defaults({"q": "ab"}, SCHEMA) == {"q": "ab", "n": 3} and apply_defaults({"n": 7}, SCHEMA)["n"] == 7


# -- registries --------------------------------------------------------------------------------------------------------------------
def test_tools_register_validate_run_and_report_failures_as_results():
    reg = ToolRegistry()

    @reg.tool("double", "Double a number", {"type": "object", "properties": {"n": {"type": "integer", "default": 2}}, "additionalProperties": False})
    def double(a):
        return {"result": a["n"] * 2}
    reg.register(Tool("boom", "fails", {"type": "object"}, lambda a: 1 / 0))
    reg.register(Tool("known", "raises a known error", {"type": "object"}, lambda a: (_ for _ in ()).throw(LocalRAGError("no such thing")), False))
    reg.register(Tool("hello", "text", {"type": "object"}, lambda a: "hi"))
    assert reg.names() == ["boom", "double", "hello", "known"] and double({"n": 1}) == {"result": 2}
    r = reg.call("double", {"n": 21})
    assert r.to_dict() == {"content": [{"type": "text", "text": json.dumps({"result": 42}, indent=2)}], "isError": False, "structuredContent": {"result": 42}}
    assert reg.call("double", None).structured == {"result": 4} and reg.call("hello", {}).to_dict() == {"content": [{"type": "text", "text": "hi"}], "isError": False}
    assert reg.call("boom", {}).is_error and "ZeroDivisionError" in reg.call("boom", {}).content[0]["text"] and "LocalRAGError: no such thing" in reg.call("known", {}).content[0]["text"]
    with pytest.raises(ProtocolError, match="unexpected property 'z'"):
        reg.call("double", {"z": 1})
    with pytest.raises(ProtocolError, match="unknown tool"):
        reg.call("nope", {})
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Tool("hello", "dup", {}, lambda a: ""))
    d = {x["name"]: x for x in reg.descriptors()}
    assert d["known"]["annotations"] == {"readOnlyHint": False, "destructiveHint": True} and d["hello"]["annotations"]["readOnlyHint"]


def test_resources_and_prompts_report_unknown_names_and_missing_arguments():
    from localrag.mcp.registry import Resource
    res = ResourceRegistry(lambda: [Resource("r://a", "a", "text/plain", lambda: "AAA", "d")])
    assert res.list()[0]["uri"] == "r://a" and res.read("r://a") == {"uri": "r://a", "mimeType": "text/plain", "text": "AAA"}
    with pytest.raises(ProtocolError) as e:
        res.read("r://zzz")
    assert e.value.code == jsonrpc.RESOURCE_NOT_FOUND
    prompts = PromptRegistry()
    prompts.register(Prompt("p", "d", [{"name": "who", "required": True}, {"name": "opt"}], lambda a: [{"role": "user", "content": {"type": "text", "text": a["who"]}}]))
    assert prompts.get("p", {"who": "Ann"})["messages"][0]["content"]["text"] == "Ann" and prompts.descriptors()[0]["name"] == "p"
    with pytest.raises(ProtocolError, match="missing prompt arguments: who"):
        prompts.get("p", None)
    with pytest.raises(ProtocolError, match="unknown prompt"):
        prompts.get("q", {})


# -- middleware and capabilities ---------------------------------------------------------------------------------------------------
def test_rate_limiter_refills_over_time():
    now = [0.0]
    lim = RateLimiter(60, lambda: now[0])
    assert all(lim.allow() for _ in range(60)) and not lim.allow()
    now[0] += 1.0
    assert lim.allow() and not lim.allow()
    now[0] += 1000
    assert lim.tokens <= 60 and lim.allow() and lim.tokens < 60


def test_token_check_timing_and_version_negotiation():
    assert check_token(None, "") and check_token("x", "") and check_token("s3", "s3") and not check_token("s4", "s3") and not check_token(None, "s3")
    t = Timing()
    t.record("a", 0.002)
    t.record("a", 0.004)
    assert t.summary() == {"a": {"calls": 2, "avg_ms": 3.0}}
    assert negotiate_version("2025-03-26") == "2025-03-26" and negotiate_version("1999-01-01") == SUPPORTED_VERSIONS[0] and negotiate_version(None) == SUPPORTED_VERSIONS[0]


# -- the server --------------------------------------------------------------------------------------------------------------------
def test_the_handshake_is_required_and_negotiates_the_version(engine):
    server = build_server(engine)
    c = MemoryClient(server)
    with pytest.raises(ProtocolError) as e:
        c.request("tools/list")
    assert e.value.code == jsonrpc.NOT_INITIALIZED and c.request("ping") == {}                  # ping is allowed before the handshake
    info = c.initialize("2025-03-26")
    assert info["protocolVersion"] == "2025-03-26" and info["serverInfo"] == {"name": "localrag", "version": "0.1.0"} and "tools" in info["capabilities"]
    assert MemoryClient(build_server(engine)).initialize("9999-01-01")["protocolVersion"] == SUPPORTED_VERSIONS[0]
    assert len(c.request("tools/list")["tools"]) == 9


def test_authentication_is_enforced_when_a_token_is_configured(engine):
    engine.config.server.auth_token = "s3cret"
    c = MemoryClient(build_server(engine))
    with pytest.raises(ProtocolError) as e:
        c.initialize()
    assert e.value.code == jsonrpc.UNAUTHORIZED
    with pytest.raises(ProtocolError):
        c.initialize(authToken="wrong")
    assert c.initialize(authToken="s3cret")["protocolVersion"]


def test_the_rate_limit_applies_after_the_handshake(engine):
    now = [0.0]
    server = McpServer(*(lambda s: (s.tools, s.resources, s.prompts))(build_server(engine)), ServerConfig(rate_limit_per_minute=3), lambda: now[0])
    c = MemoryClient(server)
    c.initialize()
    assert [c.request("tools/list") and "ok" for _ in range(3)] == ["ok"] * 3
    with pytest.raises(ProtocolError) as e:
        c.request("tools/list")
    assert e.value.code == jsonrpc.RATE_LIMITED


def test_malformed_oversized_and_unknown_requests_get_proper_errors(engine):
    server = build_server(engine)
    c = MemoryClient(server)
    c.initialize()
    assert json.loads(server.handle_raw("{bad"))["error"]["code"] == jsonrpc.PARSE_ERROR
    assert json.loads(server.handle_raw("x" * (server.config.max_message_bytes + 1)))["error"]["message"] == "message is too large"
    for method, params, code in [("nope/method", {}, jsonrpc.METHOD_NOT_FOUND), ("tools/call", {}, jsonrpc.INVALID_PARAMS), ("tools/call", {"name": "search", "arguments": [1]}, jsonrpc.INVALID_PARAMS),
                                 ("tools/call", {"name": "search", "arguments": {}}, jsonrpc.INVALID_PARAMS), ("resources/read", {}, jsonrpc.INVALID_PARAMS),
                                 ("resources/read", {"uri": "localrag://nope"}, jsonrpc.RESOURCE_NOT_FOUND), ("prompts/get", {}, jsonrpc.INVALID_PARAMS)]:
        with pytest.raises(ProtocolError) as e:
            c.request(method, params)
        assert e.value.code == code, method


def test_notifications_and_batches(engine):
    server = build_server(engine)
    assert server.handle_raw('{"jsonrpc":"2.0","method":"notifications/initialized"}') is None
    assert server.handle_raw('{"jsonrpc":"2.0","id":1,"result":{}}') is None
    batch = json.loads(server.handle_raw('[{"jsonrpc":"2.0","id":1,"method":"ping"},{"jsonrpc":"2.0","method":"x"},{"jsonrpc":"2.0","id":2,"method":"ping"}]'))
    assert [r["id"] for r in batch] == [1, 2] and server.handle_raw('[{"jsonrpc":"2.0","method":"x"}]') is None


def test_an_internal_failure_becomes_an_error_response_not_a_crash(engine):
    server = build_server(engine)
    c = MemoryClient(server)
    c.initialize()
    server._methods["ping2"] = lambda p: 1 / 0
    with pytest.raises(ProtocolError) as e:
        c.request("ping2")
    assert e.value.code == jsonrpc.INTERNAL_ERROR and "ZeroDivisionError" in str(e.value) and c.request("ping") == {}
    assert server.timing.summary()["ping2"]["calls"] == 1


def test_the_tools_search_ingest_browse_and_delete(engine, tmp_path):
    c = MemoryClient(build_server(engine, [tmp_path]))
    c.initialize()
    found = json.loads(c.tool_text("search", query="how long to boil pasta", top_k=2))
    assert found["hits"][0]["source"] == "cooking.md" and found["context"].startswith("[1] cooking.md") and found["citations"][0]["source"] == "cooking.md"
    assert c.call_tool("search", query="boil pasta", filters={"source": "dogs.md"})["structuredContent"]["hits"] == []      # the filter excludes the answer
    assert c.call_tool("search", query="dogs fetch sticks", filters={"source": "dogs.md"})["structuredContent"]["hits"][0]["source"] == "dogs.md"
    added = json.loads(c.tool_text("ingest_text", text="# Moon\n\nThe moon orbits the earth.", source="moon.md"))
    assert added["chunks"] == 1 and json.loads(c.tool_text("search", query="what orbits the earth"))["hits"][0]["source"] == "moon.md"
    assert json.loads(c.tool_text("tree_search", query="moon orbits"))[0]["source"] == "moon.md"
    doc = json.loads(c.tool_text("get_document", name="moon.md", max_chars=100))
    assert doc["source"] == "moon.md" and "- Moon" in doc["outline"] and not doc["truncated"]
    chunk_id = found["hits"][0]["chunk_id"]
    assert json.loads(c.tool_text("get_chunk", chunk_id=chunk_id))["id"] == chunk_id
    assert json.loads(c.tool_text("delete_document", name="moon.md")) == {"chunks_removed": 1} and len(json.loads(c.tool_text("list_documents"))) == 4
    missing = c.call_tool("get_document", name="gone.md")
    assert missing["isError"] and "no document 'gone.md'" in missing["content"][0]["text"] and json.loads(c.tool_text("stats"))["documents"] == 4


def test_ingest_path_cannot_read_outside_the_allowed_folders(engine, tmp_path):
    allowed, outside = tmp_path / "ok", tmp_path / "secret"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "a.md").write_text("# A\n\nallowed text")
    (outside / "s.md").write_text("# S\n\nsecret text")
    c = MemoryClient(build_server(engine, [allowed]))
    c.initialize()
    assert json.loads(c.tool_text("ingest_path", path=str(allowed)))[0]["source"] == "a.md"
    for target in (outside, allowed / ".." / "secret", outside / "s.md"):
        r = c.call_tool("ingest_path", path=str(target))
        assert r["isError"] and "outside the folders" in r["content"][0]["text"], target


def test_documents_are_resources_and_prompts_are_filled_from_the_index(engine):
    c = MemoryClient(build_server(engine))
    c.initialize()
    listing = c.request("resources/list")["resources"]
    assert listing[0]["uri"] == "localrag://stats" and len(listing) == 5 and c.request("resources/templates/list")["resourceTemplates"][0]["uriTemplate"].endswith("{id}")
    cats = next(r for r in listing if r["name"] == "cats.md")
    assert cats["mimeType"] == "text/markdown" and "purr" in c.request("resources/read", {"uri": cats["uri"]})["contents"][0]["text"]
    assert json.loads(c.request("resources/read", {"uri": "localrag://stats"})["contents"][0]["text"])["documents"] == 4
    names = [p["name"] for p in c.request("prompts/list")["prompts"]]
    assert names == ["answer_with_citations", "summarize_document"]
    ans = c.request("prompts/get", {"name": "answer_with_citations", "arguments": {"question": "do cats purr?", "top_k": "2"}})["messages"][0]["content"]["text"]
    assert "[1] cats.md" in ans and ans.endswith("Question: do cats purr?")
    assert "Outline:" in c.request("prompts/get", {"name": "summarize_document", "arguments": {"name": "cats.md"}})["messages"][0]["content"]["text"]


def test_the_stdio_transport_answers_line_by_line_and_skips_blank_lines_and_notifications(engine):
    lines = ['{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}', "", '{"jsonrpc":"2.0","method":"notifications/initialized"}',
             '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"stats","arguments":{}}}', "garbage"]
    out = io.StringIO()
    handled = serve_stdio(build_server(engine), io.StringIO("\n".join(lines) + "\n"), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert handled == 4 and [r["id"] for r in replies] == [1, 2, None] and replies[2]["error"]["code"] == jsonrpc.PARSE_ERROR
    assert json.loads(replies[1]["result"]["content"][0]["text"])["documents"] == 4 and Path(__file__).exists()
