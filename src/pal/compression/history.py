"""The Compression component for conversation history: it shortens OLD assistant replies, with no model and no new dependency.

What is compressed: only the assistant's earlier replies (the bulk of a long agent session: code and drafts). What is never
compressed: the system prompt, your own messages, the newest turns, and anything that is an exact detail (numbers, names, file
names and paths, URLs, `code` spans, "quoted" strings, error messages).

How, by kind of content:
  * a code block   -> its signature lines (def, class, function ...) stay, the body becomes "[... N lines omitted]";
  * an earlier draft that a later reply replaced (same functions, same headings) -> a one-line pointer to the newer turn;
  * a reply that saved a file -> a pointer to the file plus its first sentence;
  * prose          -> the first and last sentence, and every sentence holding an exact detail or a decision word.
Nothing is deleted: the original turn stays as it was, and a relevant old turn is still recalled in full. A turn that cannot be
made shorter is returned untouched. `extract_facts` keeps the exact details of turns that are too old to show at all.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pal.tokens import estimate_tokens

if TYPE_CHECKING:
    from pal.config import ContextConfig
    from pal.context_manager.components.turns import PagedHistory, Turn

MARK = "[compressed] "

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
_MODS = r"(?:export[ \t]+|async[ \t]+|public[ \t]+|private[ \t]+|protected[ \t]+|static[ \t]+|pub[ \t]+)*"
_DEF = re.compile(rf"[ \t]*{_MODS}(?:def|class|function|func|fn|interface|struct|enum|trait|type)[ \t]+([A-Za-z_$][\w$]*)")
_ARROW = re.compile(r"[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+([A-Za-z_$][\w$]*)[ \t]*=.*=>")
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.M)
_BULLET = re.compile(r"\s*(?:[-*\u2022]|\d+[.)])\s")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")

_FILE = re.compile(r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|js|jsx|ts|tsx|md|json|ya?ml|toml|java|kt|go|rs|c|cc|cpp|h|hpp|css|html|sh|txt|csv|sql|ini|cfg|lock|xml)\b")
_URL = re.compile(r"https?://[^\s)>\]]+")
_TICK = re.compile(r"`([^`\n]{1,40})`")
_QUOTE = re.compile(r"[\"\u201c]([^\"\u201d\n]{2,50})[\"\u201d]")
_ERROR = re.compile(r"\b[A-Z][A-Za-z]*(?:Error|Exception)\b|\bHTTP\s?\d{3}\b")
_DIGIT = re.compile(r"\d")
_ERRORWORD = re.compile(r"\b(?:error|exception|traceback|failed|failure|warning|deprecated)\b", re.I)
_DECISION = re.compile(r"\b(?:decid\w*|chose|because|instead of|must|should|recommend\w*|avoid|do not|don't|never|always|fixed?)\b", re.I)


def _is_exact(unit: str) -> bool:
    """Does this sentence hold something that must survive verbatim?"""
    return bool(_DIGIT.search(unit) or _FILE.search(unit) or _URL.search(unit) or _TICK.search(unit) or _QUOTE.search(unit)
                or _ERRORWORD.search(unit) or _ERROR.search(unit))


def _segments(text: str) -> list[tuple[str, str, str]]:
    """[(kind, lang, text)] with kind "prose" or "code"."""
    out, pos = [], 0
    for m in _FENCE.finditer(text):
        if text[pos:m.start()].strip():
            out.append(("prose", "", text[pos:m.start()]))
        out.append(("code", m.group(1).strip(), m.group(2)))
        pos = m.end()
    if text[pos:].strip():
        out.append(("prose", "", text[pos:]))
    return out


def defined_names(code: str) -> list[str]:
    names = []
    for line in code.splitlines():
        m = _DEF.match(line) or _ARROW.match(line)
        if m:
            names.append(m.group(1))
    return names


def shorten_code(lang: str, body: str, keep_other: int = 3) -> str:
    """Keep signatures and drop the body. A block of four lines or fewer is left alone."""
    lines = body.splitlines()
    if len(lines) <= 4:
        return f"```{lang}\n{body}```"
    sigs = [ln for ln in lines if _DEF.match(ln) or _ARROW.match(ln)][:12]
    kept = sigs or lines[:keep_other]
    omitted = len(lines) - len(kept)
    return f"```{lang}\n" + "\n".join(kept) + f"\n[... {omitted} lines omitted]\n```"


def _units(text: str) -> list[str]:
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _BULLET.match(line):
            units.append(line)
        else:
            units.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return units


def compress_prose(text: str) -> str:
    """First and last sentence, plus every sentence with an exact detail or a decision word. Unchanged if that saves under 15%."""
    units = _units(text)
    if len(units) <= 2:
        return text.strip()
    keep = [i in (0, len(units) - 1) or _is_exact(u) or bool(_DECISION.search(u)) for i, u in enumerate(units)]
    out = " ".join(u for u, k in zip(units, keep) if k)
    return out if len(out.split()) < 0.85 * len(text.split()) else text.strip()


# -- what a reply is "about", to notice that a later reply replaced it ----------------------------------------------------------------
def _keys(turn: Turn) -> tuple[frozenset, frozenset, list]:
    """(function and class names defined in its code, its headings lower-cased, its headings in document order)."""
    if turn._keys is None:
        text = turn.assistant
        names = frozenset(n for _, _, body in (s for s in _segments(text) if s[0] == "code") for n in defined_names(body))
        ordered = [h.strip() for h in _HEADING.findall(text)]
        turn._keys = (names, frozenset(h.lower() for h in ordered), ordered)
    return turn._keys


def _replaced_by(turns: list[Turn], i: int) -> tuple[int, list[str]] | None:
    """(later turn index, the earlier draft's headings) if a later reply is a newer version of the whole draft in turn i."""
    _, heads, ordered = _keys(turns[i])
    if len(heads) < 2:
        return None
    for j in range(i + 1, len(turns)):
        later = _keys(turns[j])[1]
        if later and len(heads & later) / len(heads) >= 0.6:
            return j, ordered[:3]
    return None


def _block_replaced_by(turns: list[Turn], i: int, names: list[str]) -> int | None:
    if not names:
        return None
    for j in range(i + 1, len(turns)):
        later = _keys(turns[j])[0]
        if later and len(set(names) & later) / len(set(names)) >= 0.5:
            return j
    return None


def compact_reply(turns: list[Turn], i: int, min_words: int = 60, count: Callable[[str], int] = estimate_tokens) -> str:
    """The compressed form of turn i's assistant reply, or the original text if compressing would not make it shorter."""
    turn = turns[i]
    text = turn.assistant
    if turn.files:
        first = " ".join(_units(" ".join(s[2] for s in _segments(text) if s[0] == "prose") or text)[:1]).split()
        out = f"{MARK}[saved {', '.join(turn.files)} in the workspace] " + " ".join(first[:25])
        return out.strip() if count(out) < count(text) else text
    if len(text.split()) < min_words:
        return text
    replaced = _replaced_by(turns, i)
    if replaced:
        j, heads = replaced
        title = ", ".join(f'"{h}"' for h in heads)
        out = f"{MARK}[earlier draft with the headings {title}, replaced by the newer version in turn {j + 1}]"
        return out if count(out) < count(text) else text
    parts = []
    for kind, lang, seg in _segments(text):
        if kind == "code":
            names = defined_names(seg)
            newer = _block_replaced_by(turns, i, names)
            parts.append(f"[earlier version of {', '.join(names[:4])}, replaced by the newer one in turn {newer + 1}]" if newer is not None
                         else shorten_code(lang, seg))
        else:
            parts.append(compress_prose(seg))
    out = MARK + "\n".join(p for p in parts if p)
    return out if count(out) < count(text) else text


# -- exact details of turns too old to show at all ----------------------------------------------------------------------------------------
# A number labelled like these is a counter or an ordinal, not a fact worth keeping ("attempt 7", "version 3", "step 2").
_ORDINAL = frozenset({"attempt", "version", "revision", "step", "turn", "number", "question", "item", "part", "page", "chapter",
                      "section", "no", "rev", "v", "draft", "round", "try", "line", "lines", "figure", "table", "example"})
_UNIT = re.compile(r"^(?:dollars?|euros?|usd|eur|gbp|inr|rupees?|rs|percent|ms|secs?|seconds?|minutes?|hours?|days?|weeks?|months?|"
                   r"years?|kb|mb|gb|tb|tokens?|rows?|users?|requests?|files?|items?|people|calls?|ports?)\b", re.I)


def _strong_number(token: str, nxt: str) -> bool:
    """Is this number worth keeping from an ASSISTANT reply? A currency, a percentage, three or more digits, or a unit after it."""
    return bool(re.search(r"[$\u20ac\u00a3\u20b9%]", token) or len(re.sub(r"\D", "", token)) >= 3 or _UNIT.match(nxt))


def extract_facts(text: str, limit: int = 12, strict: bool = False) -> list[str]:
    """Numbers (with the few words around them), file names, URLs, `code` spans, "quoted" strings and error names. No model.
    `strict` (used for assistant replies) keeps only numbers that look like real quantities."""
    prose = " ".join(s[2] for s in _segments(text) if s[0] == "prose") if "```" in text else text
    seen: set[str] = set()
    numbers: list[str] = []
    others: list[str] = []

    def add(bucket: list[str], s: str) -> None:
        s = " ".join(s.split())[:80].strip(" ,;:.()")
        if s and s.lower() not in seen:
            seen.add(s.lower())
            bucket.append(s)

    for unit in _units(prose):
        words = unit.split()
        for k, w in enumerate(words):
            if not _DIGIT.search(w) or (k > 0 and words[k - 1].lower().strip(":,.") in _ORDINAL):
                continue
            if k >= 2 and words[k - 1].lower() == "of" and _DIGIT.search(words[k - 2]):      # "2 of 5": a position, not a fact
                continue
            if strict and not _strong_number(w, words[k + 1] if k + 1 < len(words) else ""):
                continue
            add(numbers, " ".join(words[max(0, k - 3):k + 2]))
    for pat in (_FILE, _URL, _ERROR):
        for m in pat.finditer(prose):
            add(others, m.group(0))
    for pat in (_TICK, _QUOTE):
        for m in pat.finditer(prose):
            add(others, m.group(1))
    return (numbers[:6] + others[:6])[:limit] if len(numbers) > 6 else (numbers + others)[:limit]


def turn_fact_split(turn: Turn, limit: int = 12) -> tuple[list[str], list[str]]:
    """(details from what the USER said, details from the assistant's reply). The user's own words come first when space is short:
    they are the requirements and the figures that were given."""
    if turn._facts is None:
        user = extract_facts(turn.user, limit)
        have = {f.lower() for f in user}
        asst = [f for f in extract_facts(turn.assistant, limit, strict=True) if f.lower() not in have]
        turn._facts = (user, asst)
    return turn._facts


def turn_facts(turn: Turn, limit: int = 12) -> list[str]:
    user, asst = turn_fact_split(turn, limit)
    return (user + asst)[:limit]


def facts_block(turns: list[Turn], indexes: list[int], budget_tokens: int, count: Callable[[str], int] = estimate_tokens) -> tuple[str, int]:
    """(block, tokens): the exact details of the turns at `indexes`, oldest first.

    A fact that appears in several turns is listed once, at its most recent turn. When space is short, details from the user's
    own messages are kept first (newest first), then details from assistant replies (newest first)."""
    seen: set[str] = set()
    user_by: dict[int, list[str]] = {}
    asst_by: dict[int, list[str]] = {}
    for i in sorted(indexes, reverse=True):                     # newest first, so a repeated fact stays at its latest turn
        user, asst = turn_fact_split(turns[i])
        u = [f for f in user if f.lower() not in seen]
        seen.update(f.lower() for f in u)
        a = [f for f in asst if f.lower() not in seen]
        seen.update(f.lower() for f in a)
        if u:
            user_by[i] = u
        if a:
            asst_by[i] = a
    head = "<facts>\nExact details from older turns (not shown):"
    used = count(head) + count("\n</facts>")
    chosen_u: dict[int, list[str]] = {}
    chosen_a: dict[int, list[str]] = {}
    for i in sorted(user_by, reverse=True):
        cost = count(f"t{i + 1}: " + "; ".join(user_by[i])) + 1
        if used + cost > budget_tokens:
            break
        chosen_u[i] = user_by[i]
        used += cost
    for i in sorted(asst_by, reverse=True):
        for f in asst_by[i]:
            cost = count(f + "; ") + (0 if i in chosen_u or i in chosen_a else count(f"t{i + 1}: ") + 1)
            if used + cost > budget_tokens:
                continue
            chosen_a.setdefault(i, []).append(f)
            used += cost
    lines = [f"t{i + 1}: " + "; ".join(chosen_u.get(i, []) + chosen_a.get(i, [])) for i in sorted(set(chosen_u) | set(chosen_a))]
    if not lines:
        return "", 0
    return head + "\n" + "\n".join(lines) + "\n</facts>", used


def composition(turns: list[Turn], count: Callable[[str], int] = estimate_tokens) -> dict:
    """What is big? Tokens of the whole history split into your messages, assistant prose and assistant code."""
    user = prose = code = 0
    for t in turns:
        user += count(t.user)
        for kind, _, seg in _segments(t.assistant):
            if kind == "code":
                code += count(seg)
            else:
                prose += count(seg)
    return {"user": user, "assistant_prose": prose, "assistant_code": code}


# -- the layering of a conversation by age, which used to live inside the context manager ---------------------------------------------------
@dataclass
class LayeredHistory:
    """The history after compression: what to show (chronological, some turns are compressed copies) and what became only facts."""
    history: list                                   # turns to show
    compact_ids: set[int] = field(default_factory=set)    # id() of the compressed copies among them
    cold_idx: list[int] = field(default_factory=list)     # positions of the turns too old to show, left to the facts block
    warm_verbatim: int = 0                          # tokens the compressed turns would have cost in full
    warm_compact: int = 0                           # tokens they cost compressed


class HistoryCompressor:
    """Shows a conversation in layers by age. The newest turns (chosen by the pager) stay verbatim; the next `compress_turns` older ones keep
    your message and show the assistant's reply compressed, within `compress_budget_tokens`; everything older becomes a block of exact facts.
    Originals are never changed or deleted. No model is involved."""

    def __init__(self, cfg: ContextConfig, turn_cost: Callable, count: Callable[[str], int] = estimate_tokens):
        self.cfg, self.turn_cost, self.count = cfg, turn_cost, count

    def layer(self, all_turns: list, selected: list, paged: PagedHistory | None) -> LayeredHistory:
        from pal.context_manager.components.turns import Turn
        out = LayeredHistory(list(selected))
        if not (self.cfg.enabled and self.cfg.compress and paged and paged.dropped_turns):
            return out
        chosen = {id(t) for t in selected}
        dropped = [i for i, t in enumerate(all_turns) if id(t) not in chosen]
        n = self.cfg.compress_turns
        candidates = dropped[len(dropped) - n:] if n else []
        cold = dropped[:len(dropped) - len(candidates)]
        by_pos = {id(t): i for i, t in enumerate(all_turns)}
        entries = [(by_pos[id(t)], t) for t in selected]
        spent = 0
        for i in reversed(candidates):                       # newest first, so the budget goes to the most recent
            t = all_turns[i]
            text = compact_reply(all_turns, i, self.cfg.compress_min_words, self.count)
            if text is t.assistant:                          # nothing to compress: showing it would only add tokens
                cold.append(i)
                continue
            variant = Turn(t.user, text, local=t.local, ts=t.ts, files=t.files)
            cost = self.turn_cost(variant)
            if spent + cost > self.cfg.compress_budget_tokens:
                cold.append(i)
                continue
            spent += cost
            out.compact_ids.add(id(variant))
            out.warm_verbatim += self.turn_cost(t)
            out.warm_compact += cost
            entries.append((i, variant))
        out.cold_idx = sorted(cold)
        out.history = [t for _, t in sorted(entries, key=lambda e: e[0])]
        return out

    def facts(self, all_turns: list, cold_idx: list[int]) -> str | None:
        """The facts block for the turns too old to show, or None when there is none or it would not be much smaller than those turns."""
        if not cold_idx:
            return None
        text, tokens = facts_block(all_turns, cold_idx, self.cfg.facts_tokens, self.count)
        stands_for = sum(self.turn_cost(all_turns[i]) for i in cold_idx)
        return text if text and tokens <= self.cfg.facts_max_ratio * stands_for else None
