"""Pal-backed model client for the current AutoGen (autogen-agentchat 0.4 and later).

The redesigned AutoGen takes a `model_client=` object instead of calling a patchable class, so no monkeypatching
is needed: pass this client to any AutoGen agent and every model call goes through the Pal SDK.

    sdk = AgentSDK.standalone(WrapperConfig(backend="gemini", model="...", api_key="..."))
    agent = AssistantAgent("helper", model_client=autogen_model_client(sdk))

Text only: AutoGen's tool and function calling are not redirected (a clear error is raised if requested).
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pal.sdk.components.framework_adapter_base import FrameworkAdapterError
from pal.tokens import estimate_tokens


def autogen_model_client(sdk, model: str = "pal-code") -> Any:
    try:
        from autogen_core.models import (
            AssistantMessage,
            ChatCompletionClient,
            CreateResult,
            ModelInfo,
            RequestUsage,
            SystemMessage,
        )
    except ImportError as e:
        raise FrameworkAdapterError(
            "autogen-agentchat is not installed -- run `pip install autogen-agentchat` (AutoGen 0.4+) to use this client"
        ) from e

    def to_text(content: Any) -> str:
        return content if isinstance(content, str) else " ".join(str(getattr(p, "content", p)) for p in content)

    def convert(messages: Sequence[Any]) -> list[dict]:
        out = []
        for m in messages:
            role = "system" if isinstance(m, SystemMessage) else "assistant" if isinstance(m, AssistantMessage) else "user"
            out.append({"role": role, "content": to_text(getattr(m, "content", ""))})
        return out

    class PalChatCompletionClient(ChatCompletionClient):
        def __init__(self) -> None:
            self._sdk = sdk
            self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)
            self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)

        async def create(self, messages, *, tools=(), tool_choice="auto", json_output=None, extra_create_args=None,
                         cancellation_token=None):
            if tools:
                raise FrameworkAdapterError("tool calling is not supported through the Pal model client")
            args = dict(extra_create_args or {})
            call = {k: args[k] for k in ("max_tokens", "temperature") if isinstance(args.get(k), (int, float))}
            if "max_tokens" in call:
                call["max_tokens"] = int(call["max_tokens"])
            resp = await asyncio.to_thread(self._sdk.chat_response, convert(messages), **call)  # never block the loop
            self._last = RequestUsage(prompt_tokens=resp.prompt_tokens or 0, completion_tokens=resp.completion_tokens or 0)
            self._total = RequestUsage(prompt_tokens=self._total.prompt_tokens + self._last.prompt_tokens,
                                       completion_tokens=self._total.completion_tokens + self._last.completion_tokens)
            return CreateResult(finish_reason="stop", content=resp.text, usage=self._last,
                                cached=bool(resp.meta.get("cache")))

        async def create_stream(self, messages, *, tools=(), tool_choice="auto", json_output=None,
                                extra_create_args=None, cancellation_token=None):
            result = await self.create(messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
                                       extra_create_args=extra_create_args, cancellation_token=cancellation_token)
            yield str(result.content)
            yield result

        async def close(self) -> None:
            return None

        def actual_usage(self):
            return self._total

        def total_usage(self):
            return self._total

        def count_tokens(self, messages, *, tools=()) -> int:
            return sum(estimate_tokens(m["content"]) for m in convert(messages))

        def remaining_tokens(self, messages, *, tools=()) -> int:
            return max(0, 100_000 - self.count_tokens(messages))

        @property
        def capabilities(self):
            return self.model_info

        @property
        def model_info(self):
            return ModelInfo(vision=False, function_calling=False, json_output=False, family=model,
                             structured_output=False)

    return PalChatCompletionClient()
