"""Deterministic (non-LLM) checks. They run before the LLM verifier, so a
clear violation reverts a change without spending a local-model call."""
from __future__ import annotations

import re
from collections import Counter

_PROTECT = re.compile(
    r"```.*?```"                              # fenced code
    r"|`[^`\n]+`"                             # inline code
    r"|https?://[^\s<>\"']+"                  # urls
    r"|[\w.+-]+@[\w-]+\.[\w.-]+"              # emails
    r"|[A-Za-z]:\\[^\s]+"                     # windows paths
    r"|(?<![\w/])(?:~|\.{1,2})?/[\w.~\-]+(?:/[\w.~\-]+)*",  # unix paths
    re.S,
)
_PH = re.compile(r"\u27e6P\d+\u27e7")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+")
_NEG = {
    "not", "no", "never", "none", "nothing", "without", "cannot", "neither", "nor",
    "dont", "doesnt", "didnt", "cant", "wont", "isnt", "arent", "wasnt", "werent",
    "shouldnt", "couldnt", "wouldnt", "havent", "hasnt", "hadnt", "mustnt",
    "only", "except",  # restriction words: dropping them flips the meaning
}
_UNIT = re.compile(
    r"\d(?:[.,]\d+)*\s*(%|km|cm|mm|kg|mg|lbs?|oz|ml|ms|sec|min|hrs?|gb|mb|kb|tb|ghz|mhz|hz|px|usd|eur|"
    r"[mglsh])\b", re.I)
_ARTICLES = {"a", "an", "the"}


def mask(text: str, protected=()) -> tuple[str, list[str]]:
    """Swap code/URLs/paths/emails, plus any `protected` literal strings
    (e.g. text the user pasted), for placeholders the LLM must copy."""
    found: list[str] = []
    extra = sorted({p for p in protected if p and p.strip()}, key=len, reverse=True)
    pattern = re.compile("|".join([re.escape(p) for p in extra] + [_PROTECT.pattern]), re.S)

    def sub(m):
        found.append(m.group(0))
        return f"\u27e6P{len(found) - 1}\u27e7"

    return pattern.sub(sub, text), found


def unmask(text: str, found: list[str]) -> str:
    return _PH.sub(lambda m: found[int(m.group(0)[2:-1])], text)


def _words(text: str) -> list[str]:
    text = text.replace("'", "").replace("\u2019", "")
    return [w.replace(",", "") if w[0].isdigit() else w for w in _WORD.findall(text)]


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(any(w == b for b in it) for w in small)


def check(method_id: str, original: str, reduced: str) -> str | None:
    """Return a reason string if `reduced` breaks an invariant, else None.
    Both inputs are the masked texts (placeholders still in place)."""
    if not reduced.strip():
        return "empty output"
    if Counter(_PH.findall(original)) != Counter(_PH.findall(reduced)):
        return "protected span (code/URL/path) altered or dropped"
    if len(reduced) > len(original):
        return "output longer than input"

    ow, rw = _words(original), _words(reduced)
    if Counter(n for n in ow if n[0].isdigit()) - Counter(n for n in rw if n[0].isdigit()):
        return "a number was changed or dropped"
    neg = lambda ws: Counter(w.lower() for w in ws if w.lower() in _NEG)
    if neg(ow) - neg(rw):
        return "a negation/restriction word was dropped"
    units = lambda t: Counter(u.lower() for u in _UNIT.findall(t))
    if units(original) - units(reduced):
        return "a unit was changed or dropped"

    if method_id == "remove_articles":
        gone = Counter(ow) - Counter(rw)
        if not _is_subsequence(rw, ow) or any(w.lower() not in _ARTICLES for w in gone):
            return "changed something other than a/an/the"
    elif method_id == "remove_punctuation":
        strip = lambda t: re.sub(r"[^A-Za-z0-9\u27e6\u27e7]", "", t)
        if strip(original).lower() != strip(reduced).lower():
            return "letters or digits changed"
    elif method_id == "collapse_whitespace":
        if re.sub(r"\s+", "", original) != re.sub(r"\s+", "", reduced):
            return "non-whitespace characters changed"
    elif method_id == "remove_fillers":
        if not _is_subsequence(rw, ow):
            return "added or reworded words (only deletion allowed)"
    return None
