from pal.tool_manager.components.base import ToolParam, ToolSpec
from pal.tool_manager.components.param_validator import ToolValidationError, validate_params
from pal.tool_manager.components.tool_registry import ToolDependencyError, ToolRegistry


def test_validate_params_fills_defaults_and_checks_types():
    params = [
        ToolParam("query", str),
        ToolParam("limit", int, required=False, default=10),
    ]
    validated = validate_params("search", params, {"query": "hello"})
    assert validated == {"query": "hello", "limit": 10}


def test_validate_params_rejects_missing_required():
    params = [ToolParam("query", str)]
    try:
        validate_params("search", params, {})
        assert False, "expected ToolValidationError"
    except ToolValidationError as e:
        assert "missing required parameter 'query'" in str(e)


def test_validate_params_rejects_wrong_type():
    params = [ToolParam("limit", int)]
    try:
        validate_params("search", params, {"limit": "not an int"})
        assert False, "expected ToolValidationError"
    except ToolValidationError as e:
        assert "expected int" in str(e)


def test_validate_params_rejects_unknown_param():
    params = [ToolParam("query", str)]
    try:
        validate_params("search", params, {"query": "hi", "extra": 1})
        assert False, "expected ToolValidationError"
    except ToolValidationError as e:
        assert "unexpected parameter" in str(e)


def test_registry_loads_lazily_and_caches():
    load_count = {"n": 0}

    def loader():
        load_count["n"] += 1
        return lambda x: x * 2

    registry = ToolRegistry()
    registry.register(ToolSpec(name="double", loader=loader))

    assert not registry.is_loaded("double")
    fn = registry.get_callable("double")
    assert fn(5) == 10
    assert load_count["n"] == 1

    # second call reuses the cached instance -- loader not called again
    fn2 = registry.get_callable("double")
    assert fn2(3) == 6
    assert load_count["n"] == 1


def test_registry_raises_clear_error_for_missing_dependency():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="fancy_tool",
        loader=lambda: (_ for _ in ()).throw(AssertionError("loader should not run")),
        dependencies=["definitely_not_a_real_module_xyz"],
    ))

    try:
        registry.get_callable("fancy_tool")
        assert False, "expected ToolDependencyError"
    except ToolDependencyError as e:
        assert "fancy_tool" in str(e)
        assert "definitely_not_a_real_module_xyz" in str(e)
