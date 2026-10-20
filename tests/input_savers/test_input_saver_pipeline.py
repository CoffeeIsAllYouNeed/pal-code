import re

from tests.fakes import FakeLocal

from pal.input_savers.pipeline import InputSaverPipeline


def bpe_like(t):  # words, each punctuation mark, and runs of extra spaces count as tokens
    return len(re.findall(r"\w+|[^\w\s]| {2,}", t))


def run(text, core=None, methods=None, edits=None, counter=bpe_like, **kw):
    core = core or FakeLocal(edits=edits)
    return InputSaverPipeline(core, count_tokens=counter, **kw).run(text, methods), core


def step(res, mid):
    return next(s for s in res.steps if s.method_id == mid)


def test_articles_kept():
    res, _ = run("Summarize the report for a manager", methods=["remove_articles"])
    assert res.reduced == "Summarize report for manager"
    assert step(res, "remove_articles").status == "kept"
    assert res.tokens_saved == 2


def test_all_five_run_in_order_and_kept():
    text = "Please just explain the information, for a user!!   Really  now."
    res, _ = run(text, edits={"filler": lambda t: re.sub(r"\b(Please|just|Really)\s+", "", t)})
    assert [s.method_id for s in res.steps] == [
        "remove_articles", "remove_punctuation", "remove_fillers", "abbreviate", "collapse_whitespace"]
    assert "the" not in res.reduced.split() and "," not in res.reduced and "info" in res.reduced
    assert res.tokens_reduced < res.tokens_original


def test_guard_reverts_changed_number():
    res, _ = run("Send 42 items", edits={"articles": lambda t: t.replace("42", "24").replace("Send ", "")},
                 methods=["remove_articles"])
    s = step(res, "remove_articles")
    assert res.reduced == "Send 42 items" and s.status in ("reverted", "skipped")
    assert any("number" in n for n in s.chunk_notes)


def test_guard_reverts_dropped_negation():
    res, _ = run("Do not delete the file", edits={"filler": lambda t: t.replace("not ", "")}, methods=["remove_fillers"])
    assert res.reduced == "Do not delete the file"
    assert any("negation" in n for n in step(res, "remove_fillers").chunk_notes)


def test_verifier_rejection_reverts_and_later_methods_still_run():
    core = FakeLocal(verdict='{"ok": false, "lost": ["meaning"]}')
    res, _ = run("the cat", core=core, methods=["remove_articles"])
    assert res.reduced == "the cat"
    s = step(res, "remove_articles")
    assert s.status != "kept" and any("verifier" in n for n in s.chunk_notes)


def test_verifier_garbage_is_a_failure():
    res, _ = run("the cat sat", core=FakeLocal(verdict="looks fine to me"), methods=["remove_articles"])
    assert res.reduced == "the cat sat"


def test_applier_error_reverts():
    res, _ = run("the cat sat", core=FakeLocal(boom=True), methods=["remove_articles"])
    assert res.reduced == "the cat sat"
    assert "applier error" in step(res, "remove_articles").reason


def test_code_and_urls_are_protected():
    text = "Fix the bug in `the_func(a, b)` see https://x.io/a-b, thanks"
    res, _ = run(text, methods=["remove_articles", "remove_punctuation"])
    assert "`the_func(a, b)`" in res.reduced and "https://x.io/a-b" in res.reduced


def test_llm_that_alters_protected_span_is_reverted():
    bad = {"articles": lambda t: t.replace("\u27e6P0\u27e7", "").replace("the ", "")}
    res, _ = run("Use the `x.y()` call", edits=bad, methods=["remove_articles"])
    assert res.reduced == "Use the `x.y()` call"


def test_no_verification_call_when_guard_fails():
    core = FakeLocal(edits={"articles": lambda t: "totally different words"})
    run("the cat sat down", core=core, methods=["remove_articles"])
    assert "verify" not in core.calls


def test_no_token_gain_is_reverted():
    # whitespace-only change: fewer chars, same token count
    res, _ = run("a  b", methods=["collapse_whitespace"], counter=lambda t: len(t.split()))
    s = step(res, "collapse_whitespace")
    assert s.status == "reverted" and s.reason == "no token gain" and res.reduced == "a  b"


def test_partial_chunk_revert():
    calls = {"n": 0}

    def flaky(t):
        calls["n"] += 1
        return t.replace("the ", "") if "first" in t else t.replace("the ", "").replace("7", "8")

    text = "first the para\n\nsecond the para 7"
    res, _ = run(text, edits={"articles": flaky}, methods=["remove_articles"], max_chunk_chars=20)
    s = step(res, "remove_articles")
    assert res.reduced == "first para\n\nsecond the para 7"
    assert s.status == "kept" and s.chunks_total == 2 and s.chunks_kept == 1


# ---- rules-file mode ----

def rules(text, core=None, edits=None, protected=(), **kw):
    core = core or FakeLocal(edits=edits)
    return InputSaverPipeline(core, count_tokens=bpe_like, rules_text="Edit input.", **kw).run_rules(text, protected), core


def test_rules_mode_single_verified_pass():
    res, core = rules("Please explain the cache bug")
    assert res.reduced == "explain cache bug" and [s.method_id for s in res.steps] == ["rules"]
    assert res.steps[0].status == "kept" and core.calls == ["apply", "verify"]


def test_rules_mode_reverts_dropped_restriction_word():
    res, _ = rules("Change only the header", edits={"rules": lambda t: t.replace("only ", "")})
    assert res.reduced == "Change only the header"
    assert any("restriction" in n for n in res.steps[0].chunk_notes)


def test_rules_mode_reverts_changed_unit():
    res, _ = rules("Set timeout 5 km", edits={"rules": lambda t: t.replace("km", "m")})
    assert res.reduced == "Set timeout 5 km"


def test_pasted_text_is_never_edited_even_across_blank_lines():
    pasted = "the first para with the words\n\nthe second para, please"
    text = "Please review this:\n" + pasted + "\nthe end"
    res, core = rules(text, protected=[pasted])
    assert pasted in res.reduced and res.reduced.startswith("review this:")
    assert all(pasted not in m for m in core.seen)


def test_result_never_costlier_than_original():
    res, _ = rules("hi", edits={"rules": lambda t: t + " x"})
    assert res.reduced == "hi"
