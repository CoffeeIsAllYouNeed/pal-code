"""The Token Tracker: counting, crediting savings to a turn, and adding turns up."""
import pytest

from pal.metrics.components.collector import TurnMetric
from pal.token_tracker import SOURCES, TokenTracker
from pal.tokens import estimate_tokens


def test_counting_is_the_estimate_by_default_and_the_providers_own_counter_when_asked():
    t = TokenTracker()
    assert t.count("abcdefgh") == estimate_tokens("abcdefgh") == 2

    class Core:
        def token_count(self, text):
            return 99
    assert TokenTracker.counter_for("provider", Core())("anything") == 99
    assert TokenTracker.counter_for("estimate", Core())("abcdefgh") == 2 and TokenTracker.counter_for("provider", None)("abcdefgh") == 2
    assert TokenTracker(lambda s: 7).count("x") == 7


def test_each_part_credits_its_saving_to_its_own_field_of_the_turn():
    m = TurnMetric(route="api")
    for source, field, n in (("input_savers", "saved_savers", 10), ("context_paging", "saved_paging", 20), ("code_graph", "saved_graph", 30), ("documents", "saved_documents", 40)):
        TokenTracker.credit(m, source, n)
        assert getattr(m, field) == n
    assert m.saved_total == 100 and set(SOURCES) == {"input_savers", "context_paging", "code_graph", "documents"}


def test_a_negative_saving_is_recorded_as_zero_and_an_unknown_source_is_an_error_not_a_lost_saving():
    m = TurnMetric(route="api")
    TokenTracker.credit(m, "documents", -50)
    assert m.saved_documents == 0
    with pytest.raises(KeyError, match="unknown saving source 'docs'"):
        TokenTracker.credit(m, "docs", 5)


def test_summarize_adds_up_tokens_savings_compression_and_cost():
    a = TurnMetric(route="api", input_tokens=100, output_tokens=40, saved_savers=5, saved_paging=10, saved_graph=20, saved_documents=30,
                   compressed_turns=2, compression_saved=15, cost_usd=0.001, saved_cost_usd=0.0004)
    b = TurnMetric(route="local", local_tokens=60, saved_savers=1)
    s = TokenTracker.summarize([a, b])
    assert s["api_input_tokens"] == 100 and s["api_output_tokens"] == 40 and s["local_llm_tokens"] == 60
    assert s["tokens_saved_estimate"] == {"input_savers": 6, "context_paging": 10, "code_graph": 20, "documents": 30, "total": 66}
    assert s["compressed_turns"] == 2 and s["compression_saved"] == 15 and s["cost_usd"] == 0.001 and s["saved_cost_usd"] == 0.0004
    assert TokenTracker.summarize([])["tokens_saved_estimate"]["total"] == 0
