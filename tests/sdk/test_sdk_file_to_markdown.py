"""file_to_markdown for SDK agents: the converter, reached through the OS kernel interface like the other OS tools."""
import io

import pytest
from tests.fakes import FakeAPI

from pal.sdk import AgentSDK, ReActAgent, register_os_tools


def xlsx_bytes():
    openpyxl = pytest.importorskip("openpyxl")                                           # an optional library: skip without it
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["item", "cost"])
    ws.append(["hosting", 500])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def sdk(tmp_path):
    s = AgentSDK.standalone(FakeAPI(["x"]), root=str(tmp_path / "sdk"))
    register_os_tools(s)
    s._os_kernel.workspace(s.agent_id).write_bytes("inbox/budget.xlsx", xlsx_bytes())
    return s


def test_an_sdk_agent_converts_a_workspace_file_and_gets_the_text_and_a_saved_copy(sdk):
    out = sdk.call_tool("file_to_markdown", path="inbox/budget.xlsx")
    assert out.startswith("converted inbox/budget.xlsx (xlsx to markdown):") and "hosting" in out and "500" in out
    assert "Saved the full text as converted/budget.xlsx.md." in out
    assert "hosting" in sdk.os.read_file("converted/budget.xlsx.md")
    assert sdk.os.read_bytes("inbox/budget.xlsx")[:2] == b"PK"                           # the original is untouched


def test_converting_again_does_not_overwrite_and_a_bad_file_is_a_clear_error(sdk):
    first = sdk.call_tool("file_to_markdown", path="inbox/budget.xlsx")
    saved = sdk.os.read_file("converted/budget.xlsx.md")
    again = sdk.call_tool("file_to_markdown", path="inbox/budget.xlsx")
    assert "Saved the full text" in first and "converted/budget.xlsx.md already exists and was not overwritten" in again
    assert sdk.os.read_file("converted/budget.xlsx.md") == saved                           # the first copy is untouched
    with pytest.raises(RuntimeError, match="only .*docx.*pdf.*can be converted"):
        sdk.call_tool("file_to_markdown", path="notes.txt")
    with pytest.raises(RuntimeError, match="no such file"):
        sdk.call_tool("file_to_markdown", path="inbox/missing.docx")
    with pytest.raises(RuntimeError, match="invalid path"):
        sdk.call_tool("file_to_markdown", path="../escape.pdf")


def test_a_react_agent_can_use_it_and_the_model_only_reasons(tmp_path):
    core = FakeAPI(['Thought: convert it.\nAction: file_to_markdown {"path": "inbox/budget.xlsx"}', "Final Answer: it lists hosting at 500"])
    s = AgentSDK.standalone(core, root=str(tmp_path / "sdk2"))
    register_os_tools(s)
    s._os_kernel.workspace(s.agent_id).write_bytes("inbox/budget.xlsx", xlsx_bytes())
    res = ReActAgent(s).run("what does the budget say?")
    assert res.answer == "it lists hosting at 500" and "hosting" in res.steps[0].observation and len(core.seen) == 2
