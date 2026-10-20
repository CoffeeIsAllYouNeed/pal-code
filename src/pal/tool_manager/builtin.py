"""Built-in read-only tools. `safe_eval` is an AST-based calculator: it never calls eval(), only
allows numbers and + - * / // % ** with parentheses, and refuses expressions that would run for
a long time or produce enormous numbers."""
from __future__ import annotations

import ast
import operator
import time
from collections.abc import Callable
from typing import Union

from pal.tool_manager.components.base import ToolParam
from pal.tool_manager.pipeline import ToolManagerPipeline

Number = Union[int, float]

_BIN: dict[type, Callable[[Number, Number], Number]] = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY: dict[type, Callable[[Number], Number]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_LEN = 200
_MAX_BITS = 20_000


class CalculatorError(ValueError):
    pass


def _pow(a: Number, b: Number) -> Number:
    if isinstance(a, int) and isinstance(b, int) and b > 0 and a not in (0, 1, -1) and a.bit_length() * b > _MAX_BITS:
        raise CalculatorError("number too large")
    if isinstance(b, float) and abs(b) > 1e6:
        raise CalculatorError("exponent too large")
    try:
        r = a ** b
    except (OverflowError, ZeroDivisionError) as e:
        raise CalculatorError(str(e)) from e
    if isinstance(r, complex):
        raise CalculatorError("result is not a real number")
    return r


def _ev(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow):
            return _pow(a, b)
        try:
            return _BIN[type(node.op)](a, b)
        except ZeroDivisionError as e:
            raise CalculatorError("division by zero") from e
    raise CalculatorError("only numbers and + - * / // % ** ( ) are allowed")


def safe_eval(expression: str) -> Number:
    text = expression.strip().replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/")
    if not text or len(text) > _MAX_LEN:
        raise CalculatorError("empty or too long")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as e:
        raise CalculatorError("not a valid arithmetic expression") from e
    try:
        result = _ev(tree)
    except RecursionError as e:
        raise CalculatorError("expression too deeply nested") from e
    if isinstance(result, float):
        if result != result or result in (float("inf"), float("-inf")):
            raise CalculatorError("result is not finite")
        return int(result) if result.is_integer() and abs(result) < 1e15 else float(f"{result:.12g}")
    return result


def register_builtin_tools(tm: ToolManagerPipeline) -> None:
    """calculator and current_time: no state, no files, no network."""
    tm.register_tool("calculator", lambda: safe_eval, params=[ToolParam("expression", str)])
    tm.register_tool("current_time", lambda: (lambda: time.strftime("%Y-%m-%d %H:%M:%S %z")), params=[])
