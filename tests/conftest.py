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


def pytest_runtest_logreport(report):
    """On GitHub Actions, each failing test also becomes an annotation, so the failure (test name and error) shows on the public run page
    without opening the logs."""
    import os
    import sys
    if not os.environ.get("GITHUB_ACTIONS") or not report.failed:
        return
    text = str(report.longrepr) if report.longrepr is not None else "failed"
    lines = [ln for ln in text.splitlines() if ln.startswith("E ")] or text.splitlines()[-3:]
    msg = " | ".join(ln.strip() for ln in lines[:6])[:900].replace("%", "%25").replace("\r", "").replace("\n", " ")
    title = f"{report.nodeid} ({report.when})".replace("%", "%25").replace(":", "%3A").replace(",", "%2C")
    sys.__stdout__.write(f"\n::error title={title}::{msg}\n")
    sys.__stdout__.flush()
