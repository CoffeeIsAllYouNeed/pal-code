import pytest

from pal.chat.tools import ToolBox, direct_answer, strip_tool_tags
from pal.config import ToolsConfig
from pal.tool_manager.builtin import CalculatorError, safe_eval


@pytest.mark.parametrize("expr,want", [
    ("2+3*4", 14), ("(1+2)*3/4", 2.25), ("2^10", 1024), ("2**3**2", 512), ("7//2", 3), ("7%4", 3), ("-3**2", -9),
    ("0.1+0.2", 0.3), ("10/4", 2.5), ("1e3+1", 1001), ("3 \u00d7 4", 12), ("8 \u00f7 2", 4), ("  5  ", 5),
    ("12345678901234567890*98765432109876543210", 12345678901234567890 * 98765432109876543210)])
def test_safe_eval_computes(expr, want):
    assert safe_eval(expr) == want


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "abs(1)", "a+1", "[1]", "(1,2)", "1 if 1 else 2", "lambda: 1", "'a'*3",
    "True+1", "None", "1/0", "1//0", "1%0", "9**9**9", "10**10**10", "2**100000", "(-8)**0.5", "1e308*10",
    "", "   ", "1+", "((", "1" + "+1" * 200, "(" * 300 + "1" + ")" * 300, "\x00", "1j"])
def test_safe_eval_rejects_everything_else(expr):
    with pytest.raises(CalculatorError):
        safe_eval(expr)


def test_safe_eval_is_fast_on_hostile_input():
    import time
    t = time.perf_counter()
    for e in ("9**9**9**9", "2**99999999", "9" * 190 + "**" + "9" * 5):
        with pytest.raises(CalculatorError):
            safe_eval(e)
    assert time.perf_counter() - t < 1


@pytest.mark.parametrize("msg,want", [
    ("2+2", "4"), ("what is 15*(3+4)?", "105"), ("What's 2^10 =", "1024"), ("calculate 100/8", "12.5"), ("10 - 3", "7"),
    ("12", None), ("-5", None), ("hello", None), ("2 + two", None), ("7/0", None), ("what is 2+2 and why", None),
    ("explain 2+2", None), ("(1+", None), ("", None)])
def test_direct_arithmetic(msg, want):
    assert direct_answer(msg) == want


@pytest.mark.parametrize("msg", ["what time is it", "What's the time?", "current time", "what is the date", "time now"])
def test_direct_time_and_date(msg):
    ans = direct_answer(msg)
    assert ans and (":" in ans or "-" in ans)


def test_parse_and_strip():
    call = ToolBox.parse('sure <tool>{"name": "calculator", "args": {"expression": "1+1"}}</tool> ok')
    assert (call.name, call.args, call.error) == ("calculator", {"expression": "1+1"}, "")
    assert ToolBox.parse("no call here") is None
    for bad in ("<tool>{oops</tool>", "<tool>[1]</tool>", '<tool>{"args": {}}</tool>', '<tool>{"name": "x", "args": 5}</tool>'):
        assert ToolBox.parse(bad).error
    assert strip_tool_tags('a <tool>{"name":"x"}</tool> b <tool_result name="x">r</tool_result>') == "a  b r"


def test_toolbox_runs_tools_and_truncates_results():
    box = ToolBox("chat-x", ToolsConfig(max_result_tokens=50), lambda t: len(t) // 4, session_id="s")
    ok = box.run(ToolBox.parse('<tool>{"name":"calculator","args":{"expression":"6*7"}}</tool>'))
    assert (ok.ok, ok.text) == (True, "42")
    unknown = box.run(ToolBox.parse('<tool>{"name":"rm","args":{}}</tool>'))
    assert not unknown.ok and "unknown tool" in unknown.text
    wrong = box.run(ToolBox.parse('<tool>{"name":"calculator","args":{"expr":"1"}}</tool>'))
    assert not wrong.ok and "missing required parameter" in wrong.text
    box.manager.register_tool("big", lambda: (lambda: "x" * 5000))
    assert box.run(ToolBox.parse('<tool>{"name":"big","args":{}}</tool>')).text.endswith("[truncated]")


def test_available_tools_depend_on_the_message_and_the_session():
    box = ToolBox("chat-x", ToolsConfig(), len, session_id="s")
    assert box.available("explain caching") == [] and box.prompt("explain caching") == ""
    assert "calculator(expression)" in box.available("what is 12 times 3")
    assert box.available("what time is it") and ToolBox("c", ToolsConfig(enabled=False), len).available("2+2") == []
    assert ToolBox("c", ToolsConfig(max_rounds=0), len).available("2+2") == []
