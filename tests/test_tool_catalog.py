"""The named list of tools cannot drift from the tools that exist."""
from pathlib import Path

import pytest

from pal import cli
from pal.config import ToolsConfig, load_settings
from pal.sdk import AgentSDK, register_os_tools
from pal.tool_catalog import BY_NAME, CATALOG, GROUPS, render, state
from tests.fakes import FakeAPI
from tests.helpers import make_service, send


def registered_everywhere(tmp_path):
    sdk = AgentSDK.standalone(FakeAPI(["x"]), root=str(tmp_path / "sdk"))
    sdk_tools = set(sdk.list_tools()) | set(register_os_tools(sdk))
    svc, _ = make_service(tmp_path / "chat")
    send(svc, "hello", route="api", sid="s")
    return set(svc.sessions.get("s").toolbox.manager.list_tools()), sdk_tools


def test_the_catalogue_is_exactly_the_set_of_registered_tools(tmp_path):
    chat, sdk_tools = registered_everywhere(tmp_path)
    assert {e.name for e in CATALOG} == chat | sdk_tools and len(CATALOG) == 34
    assert len(chat) == 28 and len(chat | sdk_tools) == 34


def test_every_entry_is_well_formed_and_every_group_is_used():
    names, titles = [e.name for e in CATALOG], [e.title for e in CATALOG]
    assert len(set(names)) == len(names) and len(set(titles)) == len(titles)
    for e in CATALOG:
        assert e.title.endswith("Tool") or e.title.endswith("Tool (graphify)")
        assert e.group in GROUPS and e.where and 10 < len(e.summary) <= 120
    assert {e.group for e in CATALOG} == set(GROUPS)
    assert BY_NAME["read_file"].title == "File Read Tool" and BY_NAME["file_to_markdown"].title == "File to Markdown Converter Tool"
    assert BY_NAME["build_code_graph"].group == BY_NAME["find_usages"].group


def test_the_state_of_a_switchable_tool_follows_the_settings_and_the_environment():
    off = ToolsConfig()
    assert state(BY_NAME["google_search"], off, {}) == "off" and state(BY_NAME["wikipedia"], off, {}) == "off"
    on = ToolsConfig(services=["google_search", "moon_phase"], network=True)
    assert state(BY_NAME["google_search"], on, {}) == "needs SERPAPI_API_KEY"
    assert state(BY_NAME["google_search"], on, {"SERPAPI_API_KEY": "k"}) == "ready" and state(BY_NAME["moon_phase"], on, {}) == "ready"
    assert state(BY_NAME["wikipedia"], on, {}) == "ready" and state(BY_NAME["wolfram_alpha"], on, {}) == "off"
    assert state(BY_NAME["wikipedia"], on, {}) == "ready" and state(BY_NAME["calculator"], off, {}) == ""
    assert state(BY_NAME["words_api"], ToolsConfig(services=["all"]), {}) == "needs RAPIDAPI_KEY"


def test_the_rendered_list_names_every_tool_by_title_and_group():
    out = render(ToolsConfig(), {})
    assert out.startswith("Pal tools (34)\n")
    for g in GROUPS:
        assert f"\n{g}\n" in out
    for e in CATALOG:
        assert e.title in out and e.name in out
    assert "File Read Tool" in out and "Directory Read Tool" in out and "Document Search Tool" in out


def test_the_tools_command_prints_the_list_with_live_state(capsys, monkeypatch):
    for var in ("SERPAPI_API_KEY", "HF_TOKEN", "RAPIDAPI_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert cli.main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "Pal tools (34)" in out and "Google Search Tool" in out and "off" in out
    monkeypatch.setenv("SERPAPI_API_KEY", "k")
    from pal import config as cfg
    monkeypatch.setattr(cfg, "load_settings", lambda *a, **k: load_settings(env={}, overrides={"tools": {"services": ["google_search", "text_to_image"]}}))
    monkeypatch.setattr(cli, "load_settings", cfg.load_settings, raising=False)
    cli.main(["tools"])
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "google_search" in ln)
    assert "ready" in line and "needs HF_TOKEN" in next(ln for ln in out.splitlines() if "text_to_image" in ln)


def test_the_docs_list_every_tool_by_title_and_name():
    text = (Path(__file__).resolve().parents[1] / "docs" / "TOOLS.md").read_text(encoding="utf-8")
    missing = [e.name for e in CATALOG if e.title not in text or f"`{e.name}`" not in text]
    assert not missing, f"docs/TOOLS.md does not list: {missing}"
    assert "34 distinct tools" in text and "28 tools in the chat" in text


@pytest.mark.parametrize("name", ["calculator", "read_file", "build_code_graph", "file_to_markdown", "speech_to_text"])
def test_a_few_titles_are_what_the_user_expects(name):
    assert BY_NAME[name].title.endswith("Tool") or "graphify" in BY_NAME[name].title
