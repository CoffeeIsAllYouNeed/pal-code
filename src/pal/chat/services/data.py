"""Reference services: currency, words and the moon.

  * currency   Frankfurter (central bank reference rates), no key. The paper's RapidAPI converter needs a key and a listing that may change.
  * moon phase computed here, no network and no key: the phase of the moon is arithmetic.
  * WordsAPI   its real RapidAPI address and headers (checked against its own documentation); tools.rapidapi_hosts can name another host.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from pal.chat.services.base import (
    ServiceCtx,
    ServiceSpec,
    excerpt,
    param,
)
from pal.errors import PalError

RAPID = "RAPIDAPI_KEY"
_C = re.I


# -- currency (Frankfurter, no key) --------------------------------------------------------------------------------------------------
def currency_convert(ctx: ServiceCtx, amount: float, from_currency: str, to_currency: str) -> str:
    try:
        amount = float(amount)
    except (TypeError, ValueError) as e:
        raise PalError("amount must be a number") from e
    if not math.isfinite(amount) or not 0 < amount <= 1e12:
        raise PalError("amount must be a positive number")
    codes = []
    for c in (from_currency, to_currency):
        c = str(c).strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", c):
            raise PalError(f"{c!r} is not a three-letter currency code such as USD or INR")
        codes.append(c)
    base, quote_ = codes
    if base == quote_:
        return f"{amount:,.2f} {base} = {amount:,.2f} {quote_}"
    data = ctx.fetch_json("Frankfurter", f"https://api.frankfurter.dev/v2/rate/{base}/{quote_}")
    rate = float(data["rate"]) if isinstance(data, dict) and isinstance(data.get("rate"), (int, float)) else None
    if rate is None:
        raise PalError("Frankfurter returned no rate for that pair")
    return (f"{amount:,.2f} {base} = {amount * rate:,.2f} {quote_} (1 {base} = {rate:g} {quote_}, reference rate of "
            f"{data.get('date', 'today')}; not a bank or card rate)")


# -- words (WordsAPI on RapidAPI) -----------------------------------------------------------------------------------------------------
_WORD_INFO = ("all", "definitions", "synonyms", "antonyms", "rhymes", "examples", "pronunciation", "syllables", "frequency")


def words_api(ctx: ServiceCtx, word: str, info: str = "definitions") -> str:
    w = " ".join(str(word).split())
    if not re.fullmatch(r"[A-Za-z][A-Za-z' -]{0,39}", w):               # too long is refused, never silently cut short
        raise PalError("give a single English word or short phrase")
    info = str(info).strip().lower()
    if info not in _WORD_INFO:
        raise PalError(f"info must be one of: {', '.join(_WORD_INFO)}")
    host = ctx.rapid_host("words_api", "wordsapiv1.p.rapidapi.com")
    url = f"https://{host}/words/{quote(w.lower())}" + ("" if info == "all" else f"/{info}")
    d = ctx.fetch_json("WordsAPI", url, key_env=RAPID, headers={"X-RapidAPI-Key": ctx.key(RAPID), "X-RapidAPI-Host": host})
    if not isinstance(d, dict):
        return excerpt(d)
    out: list[str] = []
    if info in ("definitions", "all"):
        rows = d.get("definitions") or d.get("results") or []
        out += [f"({r.get('partOfSpeech', '?')}) {r['definition']}" for r in rows[:6] if isinstance(r, dict) and r.get("definition")]
    if info in ("synonyms", "antonyms", "examples"):
        if d.get(info):
            out.append(f"{info}: " + ", ".join(map(str, d[info][:15])))
    if info == "rhymes" and isinstance(d.get("rhymes"), dict):
        out.append("rhymes: " + ", ".join(map(str, (d["rhymes"].get("all") or [])[:15])))
    if info == "pronunciation" and d.get("pronunciation"):
        p = d["pronunciation"]
        out.append("pronunciation: " + (", ".join(f"{k} {v}" for k, v in p.items()) if isinstance(p, dict) else str(p)))
    if info == "syllables" and isinstance(d.get("syllables"), dict):
        out.append(f"syllables: {d['syllables'].get('count', '?')} ({'-'.join(d['syllables'].get('list', []))})")
    if info == "frequency" and d.get("frequency") is not None:
        out.append(f"frequency: {d['frequency']}")
    if info == "all":
        syn = sorted({s for r in (d.get("results") or []) if isinstance(r, dict) for s in (r.get("synonyms") or [])})[:12]
        if syn:
            out.append("synonyms: " + ", ".join(syn))
    if not out:
        return f"nothing found for {w!r}"
    return f"{d.get('word', w)}:\n" + "\n".join(out)


# -- the moon (computed here) ---------------------------------------------------------------------------------------------------------
SYNODIC_DAYS = 29.530588853                                   # mean length of a lunation
_REFERENCE_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
_PHASES = ((0.0339, "New Moon"), (0.2161, "Waxing Crescent"), (0.2839, "First Quarter"), (0.4661, "Waxing Gibbous"),
           (0.5339, "Full Moon"), (0.7161, "Waning Gibbous"), (0.7839, "Last Quarter"), (0.9661, "Waning Crescent"), (1.01, "New Moon"))


def moon(when: datetime) -> dict:
    """The moon's phase from the mean lunar cycle. Against eight real new and full moons it was off by 13 hours at worst, so it is
    right to about half a day; around the exact moment of a phase the name may be one day early or late."""
    age = ((when - _REFERENCE_NEW_MOON).total_seconds() / 86400) % SYNODIC_DAYS
    frac = age / SYNODIC_DAYS
    name = next(n for limit, n in _PHASES if frac < limit)
    return {"age_days": age, "phase": name, "illumination": (1 - math.cos(2 * math.pi * frac)) / 2,
            "next_new": when + timedelta(days=SYNODIC_DAYS - age),
            "next_full": when + timedelta(days=(SYNODIC_DAYS / 2 - age) % SYNODIC_DAYS)}


def moon_phase(ctx: ServiceCtx, date: str = "") -> str:
    text = str(date).strip()
    if text:
        try:
            when = datetime.strptime(text, "%Y-%m-%d").replace(hour=12, tzinfo=timezone.utc)
        except ValueError as e:
            raise PalError("give the date as YYYY-MM-DD, or leave it empty for today") from e
        label = text
    else:
        when = datetime.now(timezone.utc)
        label = when.strftime("%Y-%m-%d")
    m = moon(when)
    return (f"Moon on {label} (UTC): {m['phase']}, about {m['illumination'] * 100:.0f}% lit, {m['age_days']:.1f} days into the cycle. "
            f"Next new moon around {m['next_new']:%Y-%m-%d}, next full moon around {m['next_full']:%Y-%m-%d} "
            "(a simple model, right to about half a day).")


SERVICES = [
    ServiceSpec("currency_convert", "currency_convert(amount, from_currency, to_currency)",
                (param("amount", float), param("from_currency", str), param("to_currency", str)), currency_convert,
                "convert money at central bank reference rates (Frankfurter)",
                re.compile(r"\bcurrency\b|\bexchange rate\b|\bforex\b|\b(?:usd|eur|inr|gbp|jpy|aud|cad|chf|cny)\b|\b(?:dollars?|euros?|rupees?|pounds?|yen)\b", _C),
                None, sends="the two currency codes go to Frankfurter"),
    ServiceSpec("words_api", "words_api(word, info)", (param("word", str), param("info", str, "definitions")), words_api,
                "definitions, synonyms, antonyms, rhymes, examples (WordsAPI)",
                re.compile(r"\bsynonyms?\b|\bantonyms?\b|\bdefinition of\b|\bdefine\b|\bmeaning of\b|\brhymes? with\b|\bpronunciation\b|\bsyllables?\b|\bwhat does \w+ mean\b", _C),
                RAPID, sends="the word goes to WordsAPI"),
    ServiceSpec("moon_phase", "moon_phase(date)", (param("date", str, ""),), moon_phase,
                "the moon's phase for a date, computed locally (no key, no network)",
                re.compile(r"\bmoon\b|\blunar\b|\bfull moon\b|\bnew moon\b", _C), None, sends="nothing leaves this machine"),
]

__all__ = ["SERVICES", "moon", "SYNODIC_DAYS"]
