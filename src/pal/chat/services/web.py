"""Search and lookup services: Google results and WolframAlpha.

Google's Custom Search JSON API is closed to new customers and ends on 1 January 2027, so the "Google search" tool reads the results
page through SerpApi, a third-party results service, which needs its own key and sends your query to it.
"""
from __future__ import annotations

import re
from urllib.parse import urlencode

from pal.chat.services.base import ServiceCtx, ServiceSpec, clamp, clean_text, param
from pal.errors import PalError
from pal.redact import redact

SERPAPI = "SERPAPI_API_KEY"
WOLFRAM = "WOLFRAM_APP_ID"


def _serp(ctx: ServiceCtx, query: str, num: int) -> str:
    q = clean_text(query, "a search query", 300)
    n = clamp(num, 1, 10)
    url = "https://serpapi.com/search.json?" + urlencode({"engine": "google", "q": q, "num": n, "api_key": ctx.key(SERPAPI)})
    data = ctx.fetch_json("SerpApi", url, key_env=SERPAPI)
    if isinstance(data, dict) and data.get("error"):
        raise PalError("SerpApi: " + redact(str(data["error"]))[:200])
    rows = (data.get("organic_results") if isinstance(data, dict) else None) or []
    lines = []
    for i, r in enumerate(rows[:n], 1):
        if isinstance(r, dict) and r.get("title"):
            snippet = " ".join(str(r.get("snippet", "")).split())[:240]
            lines.append(f"{i}. {r['title']}" + (f" - {snippet}" if snippet else "") + (f" ({r['link']})" if r.get("link") else ""))
    return "\n".join(lines) or f"no google results for {q!r}"


def google_search(ctx: ServiceCtx, query: str, num: int = 5) -> str:
    return _serp(ctx, query, num)


def wolfram_alpha(ctx: ServiceCtx, query: str) -> str:
    q = clean_text(query, "a question or calculation in plain words", 300)
    url = "https://www.wolframalpha.com/api/v1/llm-api?" + urlencode({"input": q, "maxchars": 2000})
    r = ctx.net(url, headers={"Authorization": "Bearer " + ctx.key(WOLFRAM)})
    if r["status"] in (400, 501):
        raise PalError("Wolfram|Alpha did not understand that input; try rephrasing it")
    if r["status"] >= 400:
        from pal.chat.services.base import explain
        raise PalError(explain("Wolfram|Alpha", r["status"], r.get("text", ""), WOLFRAM))
    return (r.get("text") or "").strip() or "Wolfram|Alpha returned no answer"


_C = re.I
SERVICES = [
    ServiceSpec("google_search", "google_search(query, num)", (param("query", str), param("num", int, 5)), google_search,
                "Google results for a query (through SerpApi)", re.compile(r"\bgoogle\b|\bweb search\b|\bsearch (?:the )?(?:web|internet|online)\b|\blatest news\b|\bnews about\b", _C),
                SERPAPI, sends="your query goes to SerpApi"),
    ServiceSpec("wolfram_alpha", "wolfram_alpha(query)", (param("query", str),), wolfram_alpha,
                "computed answers: maths, units, science, facts (Wolfram|Alpha LLM API)",
                re.compile(r"\bwolfram\b|\bintegral\b|\bderivative\b|\bsolve\b|\bequation\b|\bhow (?:far|tall|heavy|old|big)\b|\bdistance (?:from|between)\b|\bpopulation of\b|\bconvert\b", _C),
                WOLFRAM, sends="your query goes to Wolfram"),
]
