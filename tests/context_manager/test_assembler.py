import pytest

from pal.config import load_settings
from pal.context_manager.components.assembler import Block, ContextAssembler
from pal.context_manager.components.turns import Turn
from pal.errors import ConfigError


def turn(n, tokens=100):
    return Turn(f"user {n} " + "w " * tokens, f"assistant {n}")


def cost(t):
    return len(t.user.split()) + len(t.assistant.split())


def fit(budget, turns, blocks):
    return ContextAssembler(budget, lambda s: len(s.split())).fit(turns, blocks, cost)


def blk(source, n):
    return Block(source, " ".join(["w"] * n))


def test_everything_that_fits_is_kept_untouched():
    turns, blocks, rep = fit(1000, [turn(i) for i in range(3)], [blk("facts", 50), blk("memory", 50)])
    assert len(turns) == 3 and [b.source for b in blocks] == ["facts", "memory"]
    assert rep.dropped == [] and rep.turns_dropped == 0 and rep.used <= 1000 and rep.by_source["memory"] == 50


def test_least_useful_goes_first_memory_then_facts_then_old_turns_then_documents_then_code():
    base = [turn(i, 50) for i in range(6)]                       # about 53 each
    blocks = [blk("facts", 40), blk("memory", 40), blk("documents", 80), blk("graph", 80)]
    total = sum(cost(t) for t in base) + 240
    expectations = [
        (total - 30, ["memory"], 0), (total - 70, ["memory", "facts"], 0),
        (total - 70 - 53, ["memory", "facts"], 1)]
    for budget, dropped, turns_gone in expectations:
        t, b, rep = fit(budget, base, blocks)
        assert rep.dropped == dropped and rep.turns_dropped == turns_gone and rep.used <= budget
    # a tiny budget: all old turns go (down to two), then documents, then code, then down to the last turn
    t, b, rep = fit(60, base, blocks)
    assert rep.dropped == ["memory", "facts", "documents", "graph"] and len(t) == 1 and t[0].user.startswith("user 5")


def test_the_two_newest_turns_outlive_old_ones_and_documents_outlive_older_history():
    base = [turn(i, 50) for i in range(6)]
    t, b, rep = fit(sum(cost(x) for x in base[-2:]) + 100, base, [blk("documents", 90)])
    assert [x.user.split()[1] for x in t] == ["4", "5"] and [x.source for x in b] == ["documents"]   # older turns went, the excerpt stayed


def test_the_report_counts_tokens_dropped_from_turns_and_never_empties_the_history():
    t, b, rep = fit(1, [turn(i) for i in range(4)], [])
    assert len(t) == 1 and rep.turns_dropped == 3 and rep.tokens_dropped_from_turns == sum(cost(turn(i)) for i in range(3))
    assert fit(10, [], [])[2].used == 0


def test_blocks_keep_their_given_order():
    _, blocks, _ = fit(10_000, [turn(0)], [blk("graph", 5), blk("facts", 5), blk("documents", 5)])
    assert [b.source for b in blocks] == ["graph", "facts", "documents"]


def test_context_settings_are_validated():
    c = load_settings(env={}).context
    assert (c.total_budget_tokens, c.compress, c.compress_turns, c.compress_min_words, c.facts_tokens) == (10000, True, 12, 60, 250)
    for gone in ("summarize", "summary_after_turns", "summary_words"):                  # the LLM summarizer was replaced by compression
        with pytest.raises(ConfigError, match="unknown"):
            load_settings(env={}, overrides={"context": {gone: 1}})
    for bad, msg in [({"total_budget_tokens": 500}, ">= 1000"), ({"compress_turns": -1}, ">= 0"),
                     ({"compress_min_words": 0}, ">= 1"), ({"facts_tokens": -5}, ">= 0"),
                     ({"compress_budget_tokens": -1}, ">= 0"), ({"facts_max_ratio": 1.5}, "between 0 and 1")]:
        with pytest.raises(ConfigError, match=msg):
            load_settings(env={}, overrides={"context": bad})
