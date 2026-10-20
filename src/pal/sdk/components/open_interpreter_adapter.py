"""
Adapter for Open Interpreter (Lucas, 2024). Open Interpreter's core
completion call runs through its `interpreter.llm` object's
`completions` generator (normally built on top of litellm). This
adapter monkeypatches that object's `completions` attribute so the
AgentSDK is what actually produces tokens, while Open Interpreter's own
tool-execution loop keeps running unmodified.
"""
from pal.sdk.components.framework_adapter_base import FrameworkAdapter, FrameworkAdapterError


class OpenInterpreterAdapter(FrameworkAdapter):
    framework_name = "open_interpreter"

    def install(self) -> None:
        try:
            import interpreter as oi
        except ImportError as e:
            raise FrameworkAdapterError(
                "open-interpreter is not installed -- run `pip install open-interpreter` to use this adapter"
            ) from e

        llm = getattr(oi.interpreter, "llm", None)
        if llm is None:
            raise FrameworkAdapterError(
                "interpreter.interpreter.llm was not found -- this adapter targets the "
                "Open Interpreter API this project was built against; a newer or older "
                "version may have changed it"
            )

        self._check_free(getattr(llm, "completions", None))
        self._target = llm
        self._original = getattr(llm, "completions", None)
        sdk = self.sdk

        def _redirected_completions(**kwargs):
            messages = kwargs.get("messages", [])
            text = sdk.chat(messages)
            yield {"choices": [{"delta": {"content": text}}]}

        llm.completions = self._mark(_redirected_completions)
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            self._target.completions = self._original
            self._installed = False
