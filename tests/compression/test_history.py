import random
import re

import pytest

from pal.compression import history as c
from pal.compression.history import (
    MARK,
    compact_reply,
    composition,
    compress_prose,
    defined_names,
    extract_facts,
    facts_block,
    shorten_code,
    turn_fact_split,
    turn_facts,
)
from pal.context_manager.components.turns import Turn


def words(t):
    return len(t.split())


# -- code blocks ----------------------------------------------------------------------------------------------------------------------
PY = "def load(path):\n    f = open(path)\n    data = f.read()\n    rows = data.split()\n    return rows\n\n\nclass Store:\n    def save(self, rows):\n        self.rows = rows\n"


def test_a_long_code_block_keeps_its_signatures_and_says_how_much_was_dropped():
    out = shorten_code("python", PY)
    lines = out.splitlines()
    assert lines[0] == "```python" and lines[-1] == "```"
    assert "def load(path):" in out and "class Store:" in out and "    def save(self, rows):" in out   # indentation kept
    assert "data = f.read()" not in out and re.search(r"\[\.\.\. \d+ lines omitted\]", out)
    kept = [ln for ln in lines[1:-2]]
    assert int(re.search(r"(\d+) lines omitted", out).group(1)) == len(PY.splitlines()) - len(kept)


def test_short_blocks_and_blocks_without_signatures_are_handled_sensibly():
    tiny = "x = 1\ny = 2\nz = 3\n"
    assert shorten_code("", tiny) == f"```\n{tiny}```"
    config = "\n".join(f"key{i}: value{i}" for i in range(20)) + "\n"
    out = shorten_code("yaml", config)
    assert out.splitlines()[1:4] == ["key0: value0", "key1: value1", "key2: value2"] and "[... 17 lines omitted]" in out


def test_javascript_and_other_languages_signatures_are_recognised():
    js = "export async function fetchUser(id) {\n  const r = await get(id);\n  return r;\n}\nconst parse = (x) => {\n  return x;\n};\nlet n = 1;\nlet m = 2;\n"
    assert defined_names(js) == ["fetchUser", "parse"]
    out = shorten_code("js", js)
    assert "export async function fetchUser(id) {" in out and "const parse = (x) => {" in out and "const r = await get(id);" not in out
    assert defined_names("fn main() {}\nstruct Point {\n}\ninterface Shape {\n}\ntrait Draw {\n}\ntype Id = u32\nenum Kind {\n}") == [
        "main", "Point", "Shape", "Draw", "Id", "Kind"]


def test_at_most_twelve_signatures_are_kept():
    many = "\n".join(f"def f{i}():\n    return {i}" for i in range(30)) + "\n"
    out = shorten_code("python", many)
    assert out.count("def f") == 12


# -- prose -------------------------------------------------------------------------------------------------------------------------------
def test_prose_keeps_the_first_and_last_sentence_and_every_sentence_with_an_exact_detail():
    text = ("Here is the plan for the work. First we think about the overall approach in general terms. "
            "The budget is 500 dollars. Then we discuss various other things that are not important at all. "
            "We edit src/app.py to fix the crash. Some more general talk follows here without any content. "
            "We should never call `eval` on input. A final remark closes the message.")
    out = compress_prose(text)
    assert out.startswith("Here is the plan") and out.endswith("A final remark closes the message.")
    for must in ("The budget is 500 dollars.", "We edit src/app.py to fix the crash.", "We should never call `eval` on input."):
        assert must in out
    assert "general terms" not in out and "not important at all" not in out and "without any content" not in out
    assert words(out) < 0.85 * words(text)


def test_prose_that_would_barely_shrink_or_has_two_sentences_is_left_alone():
    two = "First sentence here. Second sentence there."
    assert compress_prose(two) == two
    dense = " ".join(f"Item {i} is 5{i} units." for i in range(6))            # every sentence has a number: nothing can go
    assert compress_prose(dense) == dense.strip()


def test_list_items_stay_whole_and_numbered_lists_are_not_split_after_the_number():
    text = "Steps to follow.\n1. Install the tool first.\n2. Run the setup now.\n3. Check the output carefully.\nThat is everything required."
    units = c._units(text)
    assert units == ["Steps to follow.", "1. Install the tool first.", "2. Run the setup now.", "3. Check the output carefully.",
                     "That is everything required."]


# -- the whole reply -----------------------------------------------------------------------------------------------------------------------
def reply_with(code, prose_before="Here is the code.", prose_after="It is done now."):
    return f"{prose_before}\n```python\n{code}```\n{prose_after}"


def test_compact_reply_marks_shortens_and_never_makes_a_reply_longer():
    long_reply = reply_with(PY, prose_after="It reads 3 files and fails with KeyError sometimes. Hope it helps with the rest of the work today.")
    turns = [Turn("q", long_reply)]
    out = compact_reply(turns, 0, min_words=5)
    assert out.startswith(MARK) and len(out) < len(long_reply) and "KeyError" in out and "3 files" in out
    short = Turn("q", "ok done")
    assert compact_reply([short], 0, min_words=60) is short.assistant                    # untouched: the very same string
    already = Turn("q", "```python\ny = 1\nz = 2\n```")
    assert compact_reply([already], 0, min_words=1) is already.assistant                 # compressing would not help: untouched


def test_a_draft_that_a_later_reply_replaced_becomes_a_pointer_by_headings():
    def doc(rev):
        return f"# Guide\n\n## Install\n\nStep text number {rev} goes here for this draft.\n\n## Usage\n\nUsage text number {rev}.\n\n## Notes\n\nNotes number {rev}."
    turns = [Turn("q1", doc(1)), Turn("q2", "an unrelated reply about something else entirely"), Turn("q3", doc(3))]
    out = compact_reply(turns, 0, min_words=5)
    assert out == f'{MARK}[earlier draft with the headings "Guide", "Install", "Usage", replaced by the newer version in turn 3]'
    assert compact_reply(turns, 2, min_words=5) is turns[2].assistant or compact_reply(turns, 2, min_words=5).startswith(MARK)   # the newest is not a pointer
    assert "replaced" not in compact_reply(turns, 2, min_words=5)


def test_a_partial_heading_overlap_is_not_a_replacement():
    a = "# Guide\n\n## Install\n\ntext one\n\n## Usage\n\ntext two\n\n## Notes\n\ntext three\n\n## Extra\n\ntext four " + "pad " * 20
    b = "# Guide\n\n## Different\n\ntext\n\n## Other\n\ntext"
    assert "replaced" not in compact_reply([Turn("q", a), Turn("q", b)], 0, min_words=5)


def test_a_code_block_that_a_later_reply_redefined_becomes_a_pointer_but_the_prose_stays():
    body = "".join(f"    step_{k} = load_something_{k}(path) + compute_value_{k}()\n" for k in range(12))
    first = reply_with(f"def parse(path):\n{body}    return step_0\n", prose_after="It uses 42 rows. Done now.")
    later = reply_with("def parse(path, strict=False):\n    return []\n", prose_before="Updated.")
    out = compact_reply([Turn("q", first), Turn("q", later)], 0, min_words=5)
    assert "[earlier version of parse, replaced by the newer one in turn 2]" in out and "42 rows" in out and "step_3" not in out


def test_a_turn_that_saved_a_file_becomes_a_pointer_with_its_first_sentence():
    t = Turn("write it", "I wrote the README with install and usage sections. " + "More detail follows here. " * 20, files=("README.md",))
    out = compact_reply([t], 0)
    assert out.startswith(f"{MARK}[saved README.md in the workspace] I wrote the README") and len(out) < len(t.assistant)
    assert compact_reply([Turn("q", "ok", files=("a.md",))], 0) == "ok"                    # a pointer longer than the text is not used


# -- exact details are never lost (a property test) ---------------------------------------------------------------------------------------------
DETAIL_TEMPLATES = [
    "The budget is {n} dollars.", "Edit {f} to fix it.", "See https://example.com/doc{n} for details.", "Never call `risky_{n}` here.",
    'The label reads "Total {n}" on the page.', "It crashed with ValueError{n}.", "Port {n} is used by the server.", "Deploy on 2026-03-{d:02d}.",
]
FILLER = ["Let me think about the general approach.", "There are many things worth considering here.", "This is a broad topic overall.",
          "In general it depends on the situation.", "Hope that gives a good picture.", "Overall the idea is simple enough to follow."]


def build(seed):
    rnd = random.Random(seed)
    details, parts = [], [rnd.choice(FILLER)]
    for _ in range(rnd.randint(2, 5)):
        t = rnd.choice(DETAIL_TEMPLATES)
        s = t.format(n=rnd.randint(2, 9999), f=rnd.choice(["src/app.py", "docs/README.md", "config.yaml"]), d=rnd.randint(1, 28))
        details.append(s)
        parts += [s] + [rnd.choice(FILLER) for _ in range(rnd.randint(1, 3))]
    parts.append(rnd.choice(FILLER))
    return " ".join(parts), details


EXACT = [c._FILE, c._URL, c._TICK, c._QUOTE, c._ERROR, re.compile(r"\d[\d,.:-]*")]


@pytest.mark.parametrize("seed", range(60))
def test_every_exact_detail_in_the_prose_survives_compression(seed):
    text, details = build(seed)
    out = compress_prose(text)
    for pat in EXACT:
        for m in pat.finditer(text):
            assert m.group(0) in out, f"lost {m.group(0)!r}"
    for s in details:
        assert s in out
    assert len(out) <= len(text)


@pytest.mark.parametrize("seed", range(20))
def test_a_whole_reply_keeps_the_exact_details_of_its_prose_and_its_signatures(seed):
    text, details = build(seed)
    code = f"def fn_{seed}(x):\n    y = {seed * 7}\n    z = y + 1\n    w = z + 2\n    return w\n"
    reply = f"{text}\n```python\n{code}```\n{details[0]}"
    out = compact_reply([Turn("q", reply)], 0, min_words=5)
    for s in details:
        assert s in out
    assert f"def fn_{seed}(x):" in out and f"y = {seed * 7}" not in out


# -- facts --------------------------------------------------------------------------------------------------------------------------------------------
def test_facts_hold_numbers_with_their_words_files_urls_ticks_quotes_and_errors_without_duplicates():
    text = ("My budget is 500 dollars for the trip. We edit src/app.py and docs/README.md. See https://example.com/x for details. "
            'Call `parse_config` and read the "total cost" label. It raised KeyError and HTTP 404. The budget is 500 dollars again.')
    facts = extract_facts(text)
    joined = " | ".join(facts)
    assert "500 dollars" in joined and "src/app.py" in facts and "docs/README.md" in facts and "https://example.com/x" in facts
    assert "parse_config" in facts and "total cost" in facts and "KeyError" in facts and "HTTP 404" in facts
    assert len({f.lower() for f in facts}) == len(facts) and not any(len(f) > 80 for f in facts)


def test_facts_ignore_code_blocks_and_cap_the_number_of_numbers():
    text = "```python\nx = 12345\n```\n" + " ".join(f"Item {i} costs {i}0 euros." for i in range(1, 15)) + " See config.yaml."
    facts = extract_facts(text)
    assert not any("12345" in f for f in facts) and len(facts) <= 12 and "config.yaml" in facts
    assert extract_facts("") == [] and extract_facts("nothing exact here at all") == []


def test_turn_facts_merge_what_the_user_and_the_assistant_said_and_are_cached():
    t = Turn("my budget is 500 dollars and the trip is in March", "Noted. I will use trip.py for 3 days.")
    f = turn_facts(t)
    assert any("500" in x for x in f) and "trip.py" in f and any("3 days" in x for x in f)
    assert turn_facts(t) == f and turn_fact_split(t) is turn_fact_split(t)              # worked out once, then cached on the turn


def test_labelled_counters_are_not_facts_but_real_quantities_are():
    user = "improve the readme again, attempt 7, version 3, step 2 of 5. My budget is 500 dollars and the team has 4 people."
    facts = " | ".join(extract_facts(user))
    assert "500 dollars" in facts and "4 people" in facts and "attempt 7" not in facts and "version 3" not in facts and "step 2" not in facts
    reply = "Revision 7 adds a note. It reads 12,000 rows in 30 seconds with 50% less memory on port 8080. Item 3 is done."
    strict = " | ".join(extract_facts(reply, strict=True))
    assert "12,000 rows" in strict and "30 seconds" in strict and "50%" in strict and "8080" in strict
    assert "Revision 7" not in strict and "Item 3" not in strict


def test_facts_from_the_users_own_words_come_first_and_are_deduplicated_across_turns():
    turns = [Turn("my budget for this project is 500 dollars", "Noted. I will use config.yaml and run.sh and notes.md and a.py and b.py and c.py.")]
    turns += [Turn(f"improve it again, attempt {n}", f"Updated config.yaml with 7{n}0 rows and wrote out{n}.md and tmp{n}.txt and extra{n}.json.") for n in range(1, 15)]
    text, tokens = facts_block(turns, list(range(15)), 120)
    assert "500 dollars" in text and tokens <= 120                                      # the user's figure survives a tight budget
    items = [x.strip() for ln in text.splitlines() if re.match(r"t\d+:", ln) for x in ln.split(":", 1)[1].split(";")]
    assert items.count("config.yaml") <= 1                                               # a file named in every turn is listed once
    first = [ln for ln in text.splitlines() if "500 dollars" in ln][0]
    assert first.startswith("t1:")
    big, _ = facts_block(turns, list(range(15)), 600)
    big_items = [x.strip() for ln in big.splitlines() if re.match(r"t\d+:", ln) for x in ln.split(":", 1)[1].split(";")]
    assert big_items.count("config.yaml") == 1                                           # once, however many turns named it
    assert [ln for ln in big.splitlines() if "; config.yaml" in ln or ln.endswith(": config.yaml")][0].startswith("t15:")   # at its latest turn


def test_the_facts_block_lists_turns_oldest_first_and_drops_the_oldest_when_over_budget():
    turns = [Turn(f"the code is {n}{n}{n}", f"saved file_{n}.py") for n in range(1, 7)]
    text, tokens = facts_block(turns, list(range(6)), 400)
    assert text.startswith("<facts>\nExact details from older turns (not shown):") and text.endswith("</facts>")
    lines = [ln for ln in text.splitlines() if re.match(r"t\d+:", ln)]
    assert [ln.split(":")[0] for ln in lines] == ["t1", "t2", "t3", "t4", "t5", "t6"] and tokens <= 400
    small, small_tokens = facts_block(turns, list(range(6)), 40)
    kept = [ln.split(":")[0] for ln in small.splitlines() if re.match(r"t\d+:", ln)]
    assert kept and kept[-1] == "t6" and "t1" not in kept and small_tokens <= 40       # the oldest went first
    assert facts_block(turns, [], 400) == ("", 0) and facts_block([Turn("nothing", "nothing")], [0], 400) == ("", 0)
    assert facts_block(turns, [0], 5) == ("", 0)                                         # not even one line fits


def test_composition_splits_the_history_into_your_words_prose_and_code():
    t = [Turn("a" * 40, "prose " * 10 + "\n```python\n" + "code = 1\n" * 10 + "```")]
    h = composition(t)
    assert h["user"] == 10 and h["assistant_code"] > h["assistant_prose"] > 0
    assert composition([]) == {"user": 0, "assistant_prose": 0, "assistant_code": 0}
