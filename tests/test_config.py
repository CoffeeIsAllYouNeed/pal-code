import pytest

from pal.config import load_settings, settings_from_dict
from pal.errors import ConfigError


def test_defaults_load_and_resolve_packaged_files():
    s = load_settings(env={})
    assert s.server.host == "127.0.0.1" and s.server.port == 5000 and s.context.hot_turns == 6
    assert s.token_counter == "heuristic"
    assert "No fluff" in s.read_text_file(s.output_config)
    assert s.resolve_text_file(s.input_rules).name == "INPUT_RULES.md"
    assert s.read_text_file("does/not/exist.md") == ""


def test_env_overrides_are_typed_and_use_double_underscores():
    s = load_settings(env={"PAL__SERVER__PORT": "8080", "PAL__MEMORY__ENABLED": "false",
                           "PAL__CONTEXT__RECALL_THRESHOLD": "0.5", "PAL__TOKEN_COUNTER": "provider"})
    assert s.server.port == 8080 and s.memory.enabled is False and s.context.recall_threshold == 0.5
    assert s.token_counter == "provider"


def test_user_file_overrides_defaults_and_beats_nothing_but_env(tmp_path):
    f = tmp_path / "my.yaml"
    f.write_text("server:\n  port: 9000\nmemory:\n  profile: work\n", encoding="utf-8")
    s = load_settings(f, env={"PAL__SERVER__PORT": "9100"})
    assert s.server.port == 9100 and s.memory.profile == "work" and s.server.host == "127.0.0.1"


def test_relative_rule_files_resolve_next_to_the_user_config(tmp_path):
    (tmp_path / "out.md").write_text("Be brief.", encoding="utf-8")
    f = tmp_path / "my.yaml"
    f.write_text("output_config: out.md\n", encoding="utf-8")
    assert load_settings(f, env={}).read_text_file("out.md") == "Be brief."


def test_env_var_config_path(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("server:\n  threads: 3\n", encoding="utf-8")
    assert load_settings(env={"PAL_CONFIG": str(f)}).server.threads == 3


@pytest.mark.parametrize("bad,match", [
    ({"servr": {}}, "unknown key"),
    ({"server": {"prot": 1}}, "unknown key"),
    ({"server": {"port": "x"}}, "expected a number"),
    ({"server": {"port": 70000}}, "1-65535"),
    ({"memory": {"enabled": "yes"}}, "true/false"),
    ({"context": {"hot_turns": 0}}, ">= 1"),
    ({"context": {"recall_threshold": 2}}, "between 0 and 1"),
    ({"task_routing": {"default": "sometimes"}}, "auto, local or api"),
    ({"tools": {"max_rounds": 99}}, "between 0 and 8"),
    ({"input_mode": "magic"}, "rules"),
    ({"token_counter": "exact"}, "heuristic"),
    ({"local_llm": {"backend": "ollama", "modle": "x"}}, "unknown key"),
    ({"documents": "nope"}, "mapping"),
])
def test_bad_settings_fail_loudly_at_startup(bad, match):
    with pytest.raises(ConfigError, match=match):
        load_settings(env={}, overrides=bad)


def test_missing_and_invalid_config_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml", env={})
    f = tmp_path / "bad.yaml"
    f.write_text("server: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_settings(f, env={})


def test_to_dict_is_plain_data_and_contains_no_secrets():
    d = load_settings(env={}).to_dict()
    assert d["server"]["port"] == 5000 and isinstance(d["memory"], dict)
    assert "api_key" not in str(d)


def test_settings_from_dict_accepts_partial_input():
    assert settings_from_dict({"context": {"hot_turns": 3}}).context.hot_turns == 3


def test_local_llm_needs_a_backend_when_not_merged_with_defaults():
    with pytest.raises(ConfigError, match="backend"):
        settings_from_dict({"local_llm": {"model": "x"}})


# -- a section renamed with its component keeps working under its old name -------------------------------------------------------------------------
def test_an_old_settings_file_that_says_graph_context_still_loads_into_knowledge_graph(tmp_path):
    f = tmp_path / "old.yaml"
    f.write_text("graph_context:\n  budget_tokens: 1234\n  max_slice_lines: 77\n", encoding="utf-8")
    s = load_settings(f, env={})
    assert s.knowledge_graph.budget_tokens == 1234 and s.knowledge_graph.max_slice_lines == 77


def test_the_old_environment_variable_and_the_old_override_name_work_too_and_the_new_name_wins():
    assert load_settings(env={"PAL__GRAPH_CONTEXT__BUDGET_TOKENS": "999"}).knowledge_graph.budget_tokens == 999
    assert load_settings(env={}, overrides={"graph_context": {"budget_tokens": 555}}).knowledge_graph.budget_tokens == 555
    both = load_settings(env={}, overrides={"graph_context": {"budget_tokens": 1, "max_slice_lines": 9}, "knowledge_graph": {"budget_tokens": 2}})
    assert both.knowledge_graph.budget_tokens == 2 and both.knowledge_graph.max_slice_lines == 9


def test_a_genuinely_unknown_section_is_still_an_error():
    import pytest as _pytest
    with _pytest.raises(ConfigError, match="unknown key"):
        load_settings(env={}, overrides={"graph_contex": {}})


# -- the Studio section ------------------------------------------------------------------------------------------------------------------------------
def test_the_studio_section_has_defaults_that_stop_ollama_cutting_a_long_prompt_and_rejects_silly_values():
    s = load_settings(env={})
    assert (s.studio.num_ctx, s.studio.planner_num_ctx, s.studio.max_runs) == (16384, 8192, 6) and s.to_dict()["studio"]["num_ctx"] == 16384
    assert load_settings(env={"PAL__STUDIO__NUM_CTX": "32768"}).studio.num_ctx == 32768
    for bad in ({"num_ctx": 100}, {"planner_num_ctx": 0}, {"max_runs": 0}):
        with pytest.raises(ConfigError):
            load_settings(env={}, overrides={"studio": bad})
