import pytest

from pal.llm_wrapper.components import litellm_wrapper
from tests.fakes import FakeLiteLLM


@pytest.fixture(autouse=True)
def _default_fake_litellm(monkeypatch):
    """Every model call goes through LiteLLM now, so a core can only be built when it is importable. By default tests get a fake
    module (no install, no network). Tests that need the real pinned library reset `litellm_wrapper._module` and import it."""
    monkeypatch.setattr(litellm_wrapper, "_module", FakeLiteLLM())


@pytest.fixture
def fake_litellm(monkeypatch):
    """The fake itself, for tests that script its replies or inspect the calls it received."""
    fake = FakeLiteLLM()
    monkeypatch.setattr(litellm_wrapper, "_module", fake)
    return fake


@pytest.fixture(autouse=True)
def _no_folder_setup(monkeypatch):
    """`pal cli` and `pal web` set up the folder they start in (map, code graph, documents). Tests of other things skip that, so they do not
    scan the repository; the folder setup has its own tests, which turn it back on."""
    monkeypatch.setenv("PAL_SKIP_PROJECT", "1")
