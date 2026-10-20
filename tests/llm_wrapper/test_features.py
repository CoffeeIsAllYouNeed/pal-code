import pytest

from pal.llm_wrapper.components.adapter_types import LLMAdapterResponse, LLMConfig
from pal.llm_wrapper.components.features import (
    ToolCallError,
    check_tool_names,
    extract_json,
    json_instruction,
    normalize_tools,
    parse_arguments,
    parse_emulated_tool_calls,
    tools_instruction,
)

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "weather",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


def test_tools_are_normalised_from_openai_style_or_flat_form():
    flat = {"name": "get_weather", "description": "weather", "parameters": WEATHER["function"]["parameters"]}
    assert normalize_tools([WEATHER]) == normalize_tools([flat]) == [flat]
    assert normalize_tools([{"name": "x", "input_schema": {"type": "object"}}])[0]["parameters"] == {"type": "object"}
    assert normalize_tools([{"name": "bare"}])[0]["parameters"] == {"type": "object", "properties": {}}
    assert normalize_tools(None) == []
    with pytest.raises(ValueError, match="needs a name"):
        normalize_tools([{"description": "no name"}])


def test_arguments_are_parsed_from_json_strings_or_dicts():
    assert parse_arguments('{"a": 1}') == {"a": 1} and parse_arguments({"a": 1}) == {"a": 1}
    assert parse_arguments("") == {} and parse_arguments(None) == {}
    assert parse_arguments("not json") == {"_raw": "not json"} and parse_arguments("[1]") == {"value": [1]}


@pytest.mark.parametrize("text,want", [
    ('{"a": 1}', {"a": 1}), ('```json\n{"a": 1}\n```', {"a": 1}), ('Sure! {"a": 1} Hope that helps.', {"a": 1}),
    ("[1, 2]", [1, 2]), ('prefix [1, 2] suffix', [1, 2])])
def test_json_is_extracted_from_messy_replies(text, want):
    assert extract_json(text) == want


@pytest.mark.parametrize("text", ["", "no json here", "{broken", "```json\n{oops\n```"])
def test_replies_without_json_raise(text):
    with pytest.raises(ValueError, match="no valid JSON"):
        extract_json(text)


def test_emulated_tool_calls_are_parsed_or_none_for_plain_answers():
    tools = normalize_tools([WEATHER])
    calls = parse_emulated_tool_calls('{"tool_calls": [{"name": "get_weather", "parameters": {"city": "Pune"}}]}', tools)
    assert calls == [{"name": "get_weather", "parameters": {"city": "Pune"}}]
    assert parse_emulated_tool_calls('```json\n{"tool_calls": [{"name": "get_weather", "arguments": "{\\"city\\": \\"X\\"}"}]}\n```',
                                     tools) == [{"name": "get_weather", "parameters": {"city": "X"}}]
    for plain in ("It is sunny.", '{"answer": 4}', '{"tool_calls": []}', '{"tool_calls": [{"x": 1}]}'):
        assert parse_emulated_tool_calls(plain, tools) is None
    with pytest.raises(ToolCallError):
        check_tool_names([{"name": "launch_missiles", "parameters": {}}], tools)


def test_instructions_carry_the_contract():
    assert "get_weather" in tools_instruction(normalize_tools([WEATHER])) and "tool_calls" in tools_instruction(normalize_tools([WEATHER]))
    assert '"city"' in json_instruction({"type": "object", "properties": {"city": {}}})
    assert "schema" not in json_instruction(None) and "schema" not in json_instruction({"type": "json_object"})
    from pal.llm_wrapper.components.features import is_response_format
    assert is_response_format({"type": "json_object"}) and not is_response_format({"type": "object", "properties": {}})
    assert not is_response_format(None) and not is_response_format({"properties": {}})


def test_llm_config_reads_the_paper_style_dict_and_keeps_extras():
    c = LLMConfig.from_dict({"name": "llama3", "backend": "ollama", "hostname": "http://h:1", "max_new_tokens": 50,
                             "temperature": 0.2, "device": "cuda"})
    assert (c.name, c.backend, c.hostname, c.max_new_tokens, c.temperature, c.extra) == ("llama3", "ollama", "http://h:1", 50, 0.2, {"device": "cuda"})
    d = LLMConfig.from_dict({"model": "gpt-x", "backend": "openai", "host": "http://x", "max_tokens": 7, "api_key": "k"})
    assert (d.name, d.hostname, d.max_new_tokens, d.api_key, d.temperature) == ("gpt-x", "http://x", 7, "k", None)
    core = c.to_core_config()
    assert (core.backend, core.model, core.host, core.device) == ("ollama", "llama3", "http://h:1", "cuda")


def test_the_response_converts_to_the_core_shape():
    r = LLMAdapterResponse(response_message="hi", finished=True, prompt_tokens=3, completion_tokens=2, latency=0.5)
    core = r.to_core_response()
    assert (core.text, core.prompt_tokens, core.completion_tokens, core.meta["finished"]) == ("hi", 3, 2, True)
    assert r.ok and r.text == "hi" and LLMAdapterResponse(error="x").ok is False and LLMAdapterResponse().text == ""
