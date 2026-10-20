"""The tokens-left counters (terminal and web) and choosing the agent."""
from pal.context_window import API_DEFAULT, LOCAL_CONTEXT, context_window, short, tokens_left
from pal.terminal import backend
from pal.terminal.setup import choose_agent


def test_windows():
    assert context_window("local", "mistral:7b") == LOCAL_CONTEXT
    assert context_window("openrouter", "no-such/model-xyz") == API_DEFAULT
    assert tokens_left(8192, 3000) == 5192 and tokens_left(8192, 9000) == 0
    assert short(127_100) == "127.1k" and short(8192) == "8,192"


def test_session_counts_down(monkeypatch):
    from pal.terminal.session import PalSession

    class S:  # the two methods only need these attributes
        provider, model, role_models, mode, last_prompt, budget, pal_tokens = "local", "m", {}, "ask", 2000, 10_000, 2500
        model_for = PalSession.model_for
        tokens_left = PalSession.tokens_left
        budget_left = PalSession.budget_left
    s = S()
    assert s.tokens_left() == (LOCAL_CONTEXT - 2000, LOCAL_CONTEXT)
    assert s.budget_left() == 7500


def test_choose_agent_has_no_default(monkeypatch):
    saved, seen = {}, []
    monkeypatch.setattr(backend, "save_prefs", lambda **k: saved.update(k))

    def pick(q, options, recommended=None, other=False, header=""):
        seen.append((options, recommended))
        return "Plan"
    assert choose_agent(None, pick=pick) == "plan" and saved == {"mode": "plan"}
    assert seen[0][1] is None and [o[0] for o in seen[0][0]] == ["Ask", "Plan", "Build"]


