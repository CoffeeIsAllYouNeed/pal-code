"""
Adapter for MetaGPT (Hong et al., 2023). Every MetaGPT LLM provider
(OpenAILLM, etc.) subclasses BaseLLM and implements `acompletion_text`
as the actual async call to the model. This adapter monkeypatches
BaseLLM.acompletion_text so MetaGPT's roles/actions keep working
unmodified while the underlying call routes through pal.
"""
import asyncio

from pal.sdk.components.framework_adapter_base import FrameworkAdapter, FrameworkAdapterError


class MetaGPTAdapter(FrameworkAdapter):
    framework_name = "metagpt"

    def install(self) -> None:
        try:
            from metagpt.provider.base_llm import BaseLLM
        except ImportError as e:
            raise FrameworkAdapterError(
                "metagpt is not installed -- run `pip install metagpt` to use this adapter"
            ) from e

        if not hasattr(BaseLLM, "acompletion_text"):
            raise FrameworkAdapterError(
                "metagpt.provider.base_llm.BaseLLM has no 'acompletion_text' method -- this "
                "adapter targets the MetaGPT API this project was built against; a newer or "
                "older version may have changed it"
            )

        self._check_free(BaseLLM.acompletion_text)
        self._target = BaseLLM
        self._original = BaseLLM.acompletion_text
        sdk = self.sdk

        async def _redirected_acompletion_text(self_llm, messages, stream=False, timeout=None):
            # the SDK call is synchronous: run it in a thread so MetaGPT's event loop is never blocked
            return await asyncio.to_thread(sdk.chat, messages)

        BaseLLM.acompletion_text = self._mark(_redirected_acompletion_text)
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            self._target.acompletion_text = self._original
            self._installed = False
