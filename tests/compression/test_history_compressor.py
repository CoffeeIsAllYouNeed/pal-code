"""HistoryCompressor: the layering of a conversation by age, as its own component (it used to live inside the context manager)."""
from pal.compression import HistoryCompressor, LayeredHistory
from pal.config import ContextConfig
from pal.context_manager.components.turns import Turn, TurnPager
from pal.tokens import estimate_tokens


def convo(n, body_lines=30):
    return [Turn(f"question number {i} about the part called module{i}",
                 f"Here is module{i}:\n```python\ndef module{i}(x):\n" + "    y = x + 1\n" * body_lines + f"    return y\n```\nIt returns y. The limit is {1000 + i} items.")
            for i in range(n)]


def setup(turns, **cfg):
    config = ContextConfig(**cfg)
    pager = TurnPager(config.hot_turns, config.recall_turns, config.recall_threshold, config.max_history_tokens)
    paged = pager.select(turns, "what is the limit of module1")
    return config, pager, paged, HistoryCompressor(config, pager.turn_cost, estimate_tokens)


def test_a_short_conversation_is_left_exactly_as_it_is():
    turns = convo(3)
    config, pager, paged, comp = setup(turns)
    out = comp.layer(turns, paged.turns, paged)
    assert isinstance(out, LayeredHistory) and out.history == turns and not out.compact_ids and not out.cold_idx and comp.facts(turns, out.cold_idx) is None


def test_older_turns_keep_your_message_and_show_the_reply_compressed_and_the_originals_are_untouched():
    turns = convo(14)
    originals = [t.assistant for t in turns]
    config, pager, paged, comp = setup(turns, hot_turns=3, recall_turns=0, compress_turns=12, compress_budget_tokens=2000)
    out = comp.layer(turns, paged.turns, paged)
    shown = [t for t in out.history if id(t) in out.compact_ids]
    assert shown and all(t.user.startswith("question number") for t in shown)
    assert all("[... " in t.assistant and len(t.assistant) < 400 for t in shown)                  # the code body was cut to its signature
    assert [t.assistant for t in turns] == originals                                              # nothing was changed or deleted
    assert out.warm_verbatim > out.warm_compact > 0 and out.history == sorted(out.history, key=lambda t: turns.index(t) if t in turns else turns.index(next(x for x in turns if x.user == t.user)))


def test_the_compressed_layer_has_a_token_budget_and_what_does_not_fit_becomes_facts():
    turns = convo(16)
    config, pager, paged, comp = setup(turns, hot_turns=2, recall_turns=0, compress_turns=12, compress_budget_tokens=150)
    out = comp.layer(turns, paged.turns, paged)
    cost = sum(pager.turn_cost(t) for t in out.history if id(t) in out.compact_ids)
    assert cost <= 150 and out.cold_idx and out.cold_idx == sorted(out.cold_idx)
    facts = comp.facts(turns, out.cold_idx)
    assert facts is None or ("1000" in facts or "module" in facts)


def test_the_facts_block_is_only_used_when_it_is_much_smaller_than_the_turns_it_stands_for():
    turns = convo(12, body_lines=40)
    config, pager, paged, comp = setup(turns, hot_turns=2, recall_turns=0, compress_turns=0)
    out = comp.layer(turns, paged.turns, paged)
    assert out.cold_idx and not out.compact_ids                                                    # compress_turns=0: everything dropped goes to facts
    facts = comp.facts(turns, out.cold_idx)
    assert facts and estimate_tokens(facts) <= config.facts_max_ratio * sum(pager.turn_cost(turns[i]) for i in out.cold_idx)
    strict = HistoryCompressor(ContextConfig(facts_max_ratio=0.0001, hot_turns=2, recall_turns=0, compress_turns=0), pager.turn_cost)
    assert strict.facts(turns, out.cold_idx) is None


def test_compression_can_be_switched_off_in_the_settings():
    turns = convo(14)
    for off in ({"compress": False}, {"enabled": False}):
        config, pager, paged, comp = setup(turns, hot_turns=3, recall_turns=0, **off)
        out = comp.layer(turns, paged.turns, paged)
        assert not out.compact_ids and not out.cold_idx and out.history == list(paged.turns)
