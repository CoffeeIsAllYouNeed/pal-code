from pal.context_manager import Turn, TurnPager
from pal.context_manager.utils.relevance import keyword_overlap, text_similarity


def turns(n, text="filler chat about lunch", reply="ok"):
    return [Turn(f"{text} {i}", reply, ts=1000 + i) for i in range(n)]


def test_short_history_is_sent_whole():
    t = turns(3)
    p = TurnPager(hot_turns=6).select(t, "anything")
    assert p.turns == t and p.saved_tokens == 0 and p.dropped_turns == 0


def test_only_the_hot_turns_are_kept_when_nothing_old_is_relevant():
    t = turns(10)
    p = TurnPager(hot_turns=3, recall_turns=2).select(t, "quantum chromodynamics")
    assert p.turns == t[-3:] and p.dropped_turns == 7 and p.recalled == 0 and p.saved_tokens > 0


def test_a_relevant_old_turn_is_recalled_and_order_stays_chronological():
    t = turns(10)
    t[1] = Turn("our kubernetes deployment uses port 8443", "noted", ts=1001)
    p = TurnPager(hot_turns=3, recall_turns=1).select(t, "which port does the kubernetes deployment use")
    assert p.recalled == 1 and p.turns == [t[1], *t[-3:]]


def test_recall_turns_zero_never_recalls():
    t = turns(10)
    t[1] = Turn("kubernetes port 8443", "noted")
    assert TurnPager(hot_turns=2, recall_turns=0).select(t, "kubernetes port").recalled == 0


def test_token_cap_trims_the_hot_window_but_keeps_the_newest_turn():
    big = [Turn("x" * 4000, "y" * 4000, ts=i) for i in range(5)]
    p = TurnPager(hot_turns=5, recall_turns=0, max_history_tokens=2500).select(big, "q")
    assert 1 <= len(p.turns) < 5 and p.turns[-1] is big[-1] and p.sent_tokens <= 2500 + 16


def test_identical_turns_do_not_break_ordering():
    t = [Turn("same", "same", ts=1) for _ in range(6)]
    assert len(TurnPager(hot_turns=2, recall_turns=0).select(t, "q").turns) == 2


def test_empty_history_and_empty_query():
    assert TurnPager().select([], "q").turns == []
    assert TurnPager(hot_turns=1).select(turns(4), "  ").recalled == 0


def test_keyword_overlap_ignores_stopwords_punctuation_and_case():
    assert keyword_overlap("What port does the Kubernetes deployment use?", "kubernetes deployment port 8443") == 0.75
    assert keyword_overlap("Which port, Kubernetes deployment?", "Kubernetes DEPLOYMENT: port 8443") == 1.0
    assert keyword_overlap("the a of", "anything") == 0.0
    assert keyword_overlap("", "x") == 0.0
    assert text_similarity("port", ["port 1", "lunch"]) == [1.0, 0.0]
