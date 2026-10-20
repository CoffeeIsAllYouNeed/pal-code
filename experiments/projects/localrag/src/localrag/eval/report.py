"""Showing an evaluation: a readable summary and the failed queries."""
from __future__ import annotations

import json

from localrag.eval.runner import EvalReport


def format_report(report: EvalReport) -> str:
    """The metrics as a short table, then up to five queries that found nothing."""
    lines = [f"Evaluated {len(report.outcomes)} queries at k={report.k} in {report.seconds:.2f}s",
             f"  recall@{report.k}     {report.recall:.3f}", f"  precision@{report.k}  {report.precision:.3f}", f"  MRR          {report.mrr:.3f}",
             f"  MAP          {report.map:.3f}", f"  nDCG@{report.k}      {report.ndcg:.3f}"]
    failures = report.failures()
    if failures:
        lines.append(f"  {len(failures)} queries found nothing relevant, for example:")
        lines += [f"    - {o.query!r} (wanted {sorted(o.relevant)}, got {o.ranked_sources[:3]})" for o in failures[:5]]
    return "\n".join(lines)


def to_json(report: EvalReport) -> str:
    return json.dumps({"k": report.k, "queries": len(report.outcomes), "recall": round(report.recall, 4), "precision": round(report.precision, 4),
                       "mrr": round(report.mrr, 4), "map": round(report.map, 4), "ndcg": round(report.ndcg, 4),
                       "failures": len(report.failures())}, indent=2)
