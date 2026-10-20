import json
from dataclasses import asdict, dataclass


@dataclass
class ToolRequest:
    agent_name: str
    tool_name: str
    args: dict

    def encode(self) -> bytes:
        return (json.dumps(asdict(self)) + "\n").encode("utf-8")

    @staticmethod
    def decode(raw: bytes) -> "ToolRequest":
        data = json.loads(raw.decode("utf-8"))
        return ToolRequest(**data)


@dataclass
class ToolResponse:
    ok: bool
    result: object = None
    error: str | None = None

    def encode(self) -> bytes:
        return (json.dumps(asdict(self)) + "\n").encode("utf-8")

    @staticmethod
    def decode(raw: bytes) -> "ToolResponse":
        data = json.loads(raw.decode("utf-8"))
        return ToolResponse(**data)
