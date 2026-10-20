import sys

import pytest

from pal.sdk.components.autogen_adapter import AutoGenAdapter
from pal.sdk.components.framework_adapter_base import FrameworkAdapterError
from pal.sdk.components.metagpt_adapter import MetaGPTAdapter
from pal.sdk.components.open_interpreter_adapter import OpenInterpreterAdapter


class _FakeSDK:
    """None of these tests get as far as calling sdk.chat(), so a bare stand-in is enough."""
    agent_id = "agent-1"


def absent(monkeypatch, *names):
    """Simulate a library that is not installed, even on a machine where it is (None in sys.modules = ImportError)."""
    for n in names:
        monkeypatch.setitem(sys.modules, n, None)


def test_autogen_adapter_raises_clean_error_when_not_installed(monkeypatch):
    absent(monkeypatch, "autogen", "autogen.oai", "autogen.oai.client")
    adapter = AutoGenAdapter(_FakeSDK())
    with pytest.raises(FrameworkAdapterError, match="pip install") as e:
        adapter.install()
    assert "autogen" in str(e.value).lower() and "autogen_model_client" in str(e.value)
    assert not adapter.installed


def test_open_interpreter_adapter_raises_clean_error_when_not_installed(monkeypatch):
    absent(monkeypatch, "interpreter")
    adapter = OpenInterpreterAdapter(_FakeSDK())
    with pytest.raises(FrameworkAdapterError, match="pip install") as e:
        adapter.install()
    assert "open-interpreter" in str(e.value).lower()
    assert not adapter.installed


def test_metagpt_adapter_raises_clean_error_when_not_installed(monkeypatch):
    absent(monkeypatch, "metagpt", "metagpt.provider", "metagpt.provider.base_llm")
    adapter = MetaGPTAdapter(_FakeSDK())
    with pytest.raises(FrameworkAdapterError, match="pip install") as e:
        adapter.install()
    assert "metagpt" in str(e.value).lower()
    assert not adapter.installed


def test_uninstall_before_install_is_a_safe_no_op():
    adapter = AutoGenAdapter(_FakeSDK())
    adapter.uninstall()  # never installed -- should not raise
    assert not adapter.installed
