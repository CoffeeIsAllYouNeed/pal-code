"""
Adapter for AutoGen (Wu et al., 2023). AutoGen's ConversableAgent
ultimately calls out to an OpenAIWrapper instance's .create() method,
which builds the actual chat-completion request. This adapter
monkeypatches OpenAIWrapper.create to call the AgentSDK's chat()
instead, and wraps the reply back into the minimal ChatCompletion-like
shape (.choices[0].message.content) AutoGen's own code reads from --
so callers inside AutoGen don't notice the difference.

Targets the AutoGen 0.2 API (`autogen.oai.client.OpenAIWrapper`, the `pyautogen<0.3` / `ag2<1` line). The
redesigned AutoGen (autogen-agentchat 0.4+) has no such class; use `PalChatCompletionClient` for that one.
Text replies only: tool and function calling are not redirected.
"""
from types import SimpleNamespace

from pal.sdk.components.framework_adapter_base import FrameworkAdapter, FrameworkAdapterError


class AutoGenAdapter(FrameworkAdapter):
    framework_name = "autogen"

    def install(self) -> None:
        try:
            from autogen.oai.client import OpenAIWrapper
        except ImportError as e:
            raise FrameworkAdapterError(
                "AutoGen 0.2 is not installed -- run `pip install \"pyautogen==0.2.35\"` to use this adapter. For AutoGen 0.4+ "
                "use pal.sdk.autogen_model_client(sdk) instead"
            ) from e

        if not hasattr(OpenAIWrapper, "create"):
            raise FrameworkAdapterError(
                "autogen.oai.client.OpenAIWrapper has no 'create' method -- this adapter "
                "targets the AutoGen API this project was built against; a newer or older "
                "AutoGen version may have changed it"
            )

        self._check_free(OpenAIWrapper.create)
        self._target = OpenAIWrapper
        self._original = OpenAIWrapper.create
        sdk = self.sdk

        def _redirected_create(self_wrapper, **kwargs):
            messages = kwargs.get("messages", [])
            call = {k: kwargs[k] for k in ("max_tokens", "temperature") if isinstance(kwargs.get(k), (int, float))}
            if "max_tokens" in call:
                call["max_tokens"] = int(call["max_tokens"])
            resp = sdk.chat_response(messages, **call)
            message = SimpleNamespace(content=resp.text, role="assistant", function_call=None, tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
            usage = SimpleNamespace(prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                                    total_tokens=resp.prompt_tokens + resp.completion_tokens)
            # AutoGen 0.2 reads these back from every response (extract_text_or_completion_object, usage, cost)
            return SimpleNamespace(
                choices=[choice], model=kwargs.get("model", "pal-code"), usage=usage, cost=0.0, config_id=0,
                pass_filter=None,
                message_retrieval_function=lambda r: [c.message.content for c in r.choices])

        OpenAIWrapper.create = self._mark(_redirected_create)
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            self._target.create = self._original
            self._installed = False
