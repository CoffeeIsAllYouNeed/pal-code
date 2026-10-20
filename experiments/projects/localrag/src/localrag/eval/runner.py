"""Running a golden set against an engine and averaging the ranking metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from localrag.engine import LocalRAG
from localrag.eval import metrics
from localrag.eval.golden import GoldenQuery


@dataclass
class QueryOutcome:
    query: str
    relevant: set[str]
    ranked_sources: list[str]
    first_relevant_rank: int          # 0 when nothing relevant was found


@dataclass
class EvalReport:
    k: int
    outcomes: list[QueryOutcome] = field(default_factory=list)
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    map: float = 0.0
    ndcg: float = 0.0
    seconds: float = 0.0

    def failures(self) -> list[QueryOutcome]:
        """The queries for which no relevant source was found in the first k results."""
        return [o for o in self.outcomes if o.first_relevant_rank == 0]


def evaluate(engine: LocalRAG, golden: list[GoldenQuery], k: int = 5) -> EvalReport:
    """Search for every golden query and average recall, precision, MRR, MAP and nDCG over them (ranking by source, best first)."""
    report, started = EvalReport(k), time.perf_counter()
    for g in golden:
        sources: list[str] = []
        for hit in engine.search(g.query, k).hits:
            source = str(hit.chunk.metadata.get("source", hit.chunk.doc_id))
            if source not in sources:
                sources.append(source)
        rank = next((i for i, s in enumerate(sources, start=1) if s in g.relevant), 0)
        report.outcomes.append(QueryOutcome(g.query, g.relevant, sources, rank))
    n = len(report.outcomes) or 1
    for name, fn in (("recall", lambda o: metrics.recall_at_k(o.relevant, o.ranked_sources, k)),
                     ("precision", lambda o: metrics.precision_at_k(o.relevant, o.ranked_sources, k)),
                     ("mrr", lambda o: metrics.reciprocal_rank(o.relevant, o.ranked_sources)),
                     ("map", lambda o: metrics.average_precision(o.relevant, o.ranked_sources)),
                     ("ndcg", lambda o: metrics.ndcg_at_k(o.relevant, o.ranked_sources, k))):
        setattr(report, name, sum(fn(o) for o in report.outcomes) / n)
    report.seconds = time.perf_counter() - started
    return report
