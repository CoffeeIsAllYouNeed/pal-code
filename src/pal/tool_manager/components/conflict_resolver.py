"""
Resolution of tool call conflicts (AIOS section 3.7): tools with a
parallel-access limit (ToolSpec.parallel_limit) are tracked in a
hashmap of tool_name -> current in-flight call count. A request is only
admitted if doing so would not push that count past the tool's limit.
"""
from typing import Optional


class ToolConflictResolver:
    def __init__(self):
        self._in_flight: dict[str, int] = {}

    def can_acquire(self, tool_name: str, parallel_limit: Optional[int]) -> bool:
        if parallel_limit is None:
            return True
        return self._in_flight.get(tool_name, 0) < parallel_limit

    def acquire(self, tool_name: str) -> None:
        self._in_flight[tool_name] = self._in_flight.get(tool_name, 0) + 1

    def release(self, tool_name: str) -> None:
        if tool_name in self._in_flight:
            self._in_flight[tool_name] = max(0, self._in_flight[tool_name] - 1)

    def in_flight_count(self, tool_name: str) -> int:
        return self._in_flight.get(tool_name, 0)
