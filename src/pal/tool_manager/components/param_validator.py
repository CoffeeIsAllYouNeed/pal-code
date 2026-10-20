"""
Parameter validation before execution -- "incorporating parameter
validation before execution to prevent tool crashes." Runs before a
tool's underlying callable is ever invoked, so a malformed call fails
with a clear ToolValidationError instead of an arbitrary exception (or
worse, silently wrong behavior) inside the tool itself.
"""
from pal.tool_manager.components.base import ToolParam


class ToolValidationError(ValueError):
    pass


def validate_params(tool_name: str, params: list[ToolParam], args: dict) -> dict:
    """Returns a normalized args dict (missing optional params filled in
    with their declared defaults) or raises ToolValidationError."""
    validated = {}
    declared_names = {p.name for p in params}

    for p in params:
        if p.name in args:
            value = args[p.name]
            if p.type is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)                                     # JSON writes 100 for 100.0
            if not isinstance(value, p.type) or (p.type in (int, float) and isinstance(value, bool)):
                raise ToolValidationError(
                    f"{tool_name}: parameter '{p.name}' expected {p.type.__name__}, "
                    f"got {type(value).__name__}"
                )
            validated[p.name] = value
        elif p.required:
            raise ToolValidationError(f"{tool_name}: missing required parameter '{p.name}'")
        else:
            validated[p.name] = p.default

    unknown = set(args) - declared_names
    if unknown:
        raise ToolValidationError(f"{tool_name}: unexpected parameter(s): {sorted(unknown)}")

    return validated
