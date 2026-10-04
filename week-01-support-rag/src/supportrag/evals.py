"""Evals: measure retrieval and answers separately.

Two questions, two evals, because they fail for different reasons:

1. Retrieval: did the right page come back? Needs no model call, runs in
   milliseconds, and is the number you watch while changing chunking,
   ranking or embedding models.
2. Answers: given the tools, did the agent behave correctly? Slower and
   costs tokens, so it is run less often.

If answers are bad, look at retrieval first. A model cannot cite a passage
it was never shown.

Every grader here is plain code. Code graders are cheap, deterministic and
cannot be argued with; they check behaviour (cited the right page, escalated
when it should). Judging whether prose is *correct* needs a model as judge,
which is Week 4.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from supportrag.agent import AgentResult
from supportrag.retrieval import Hit


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file: one JSON object per line, blank lines ignored."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


@dataclass
class RetrievalReport:
    k: int
    rows: list[dict[str, Any]] = field(default_factory=list)

    def _subset(self, kind: str | None) -> list[dict[str, Any]]:
        return [r for r in self.rows if kind is None or r["kind"] == kind]

    def hit_rate(self, at: int, kind: str | None = None) -> float:
        """Share of questions with an expected page in the top `at` results."""
        rows = self._subset(kind)
        return sum(1 for r in rows if r["rank"] is not None and r["rank"] <= at) / len(rows) if rows else 0.0

    def mrr(self, kind: str | None = None) -> float:
        """Mean reciprocal rank: 1/rank of the first correct result, averaged.

        Unlike hit rate, it rewards putting the right page first rather than
        fifth. A miss counts as 0.
        """
        rows = self._subset(kind)
        return sum(1.0 / r["rank"] for r in rows if r["rank"] is not None) / len(rows) if rows else 0.0

    def kinds(self) -> list[str]:
        return sorted({r["kind"] for r in self.rows})


def first_relevant_rank(hits: list[Hit], expected_docs: list[str]) -> int | None:
    """Rank (1-based) of the first hit from an expected page, else None.

    Relevance is judged per page, not per chunk. Chunk ids change every time
    the chunker changes; page paths do not, so the eval set survives the
    experiments it exists to measure.
    """
    for position, hit in enumerate(hits, start=1):
        if hit.chunk.doc_path in expected_docs:
            return position
    return None


def run_retrieval_eval(
    cases: list[dict[str, Any]], search: Callable[[str, int], list[Hit]], k: int = 5
) -> RetrievalReport:
    report = RetrievalReport(k=k)
    for case in cases:
        hits = search(case["question"], k)
        report.rows.append(
            {
                "id": case["id"],
                "kind": case.get("kind", "direct"),
                "question": case["question"],
                "expected": case["expected_docs"],
                "rank": first_relevant_rank(hits, case["expected_docs"]),
                "got": [hit.chunk.doc_path for hit in hits],
            }
        )
    return report


def format_retrieval_report(report: RetrievalReport, label: str) -> str:
    lines = [f"Retrieval eval: {label}, {len(report.rows)} questions", ""]
    lines.append(f"{'subset':<12}{'n':>4}{'hit@1':>8}{'hit@3':>8}{f'hit@{report.k}':>8}{'MRR':>8}")
    for kind in [None, *report.kinds()]:
        rows = report._subset(kind)
        lines.append(
            f"{kind or 'all':<12}{len(rows):>4}"
            f"{report.hit_rate(1, kind):>8.2f}{report.hit_rate(3, kind):>8.2f}"
            f"{report.hit_rate(report.k, kind):>8.2f}{report.mrr(kind):>8.2f}"
        )
    misses = [r for r in report.rows if r["rank"] is None]
    if misses:
        lines += ["", f"Misses ({len(misses)}):"]
        for row in misses:
            lines.append(f"  {row['id']} [{row['kind']}] {row['question']}")
            lines.append(f"      expected {row['expected']}")
            lines.append(f"      got      {list(dict.fromkeys(row['got']))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


def mentions(answer: str, phrase: str) -> bool:
    """Whole-phrase, case-insensitive match, ignoring citation markers.

    Two traps this avoids: "5" matching inside "15", and "5" matching the
    citation "[5]".
    """
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", answer)
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


def grade_answer(case: dict[str, Any], result: AgentResult) -> dict[str, bool]:
    """Apply every check that is relevant to this case. True means passed."""
    checks: dict[str, bool] = {}
    should_escalate = case.get("should_escalate", False)

    # Escalating when it should, and only then. Both directions are failures:
    # a missed escalation is a made-up answer, a needless one is a wasted human.
    checks["escalation_correct"] = result.escalated == should_escalate
    # Never cite a source number that no search returned.
    checks["citations_valid"] = not result.invalid_citations
    checks["within_budget"] = not result.stopped_early

    if not should_escalate:
        checks["has_citation"] = bool(result.cited)
        expected = case.get("expected_docs", [])
        if expected:
            checks["cites_expected_page"] = any(s.chunk.doc_path in expected for s in result.cited)
        # `must_mention` is a list of groups; each group passes if any one of
        # its phrases appears. It catches an answer that cites the right page
        # and still leaves out the fact the customer asked for.
        for index, group in enumerate(case.get("must_mention", []), start=1):
            checks[f"mentions_{index}"] = any(mentions(result.answer, phrase) for phrase in group)
    return checks


def run_answer_eval(cases: list[dict[str, Any]], ask: Callable[[str], AgentResult]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        result = ask(case["question"])
        checks = grade_answer(case, result)
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "passed": all(checks.values()),
                "checks": checks,
                "answer": result.answer,
                "escalated": result.escalated,
                "steps": result.steps,
                "searches": [e["query"] for e in result.trace if e["tool"] == "search_docs" and "query" in e],
                "cited": [s.chunk.doc_path for s in result.cited],
                "usage": result.usage,
            }
        )
    return rows


def format_answer_report(rows: list[dict[str, Any]]) -> str:
    passed = sum(1 for r in rows if r["passed"])
    lines = [f"Answer eval: {passed}/{len(rows)} cases passed", ""]

    # Per-check pass rates say *what* is failing, which the headline cannot.
    totals: dict[str, list[int]] = {}
    for row in rows:
        for name, ok in row["checks"].items():
            key = "mentions_required_fact" if name.startswith("mentions_") else name
            bucket = totals.setdefault(key, [0, 0])
            bucket[0] += int(ok)
            bucket[1] += 1
    for name, (ok, total) in totals.items():
        lines.append(f"  {name:<24}{ok:>3}/{total}")

    failures = [r for r in rows if not r["passed"]]
    if failures:
        lines += ["", "Failures:"]
        for row in failures:
            failed = [name for name, ok in row["checks"].items() if not ok]
            lines.append(f"  {row['id']}: {', '.join(failed)}")
            lines.append(f"      Q: {row['question']}")
            lines.append(f"      searches: {row['searches']}  cited: {row['cited']}  escalated: {row['escalated']}")
            lines.append(f"      A: {row['answer'][:240].replace(chr(10), ' ')}")
    tokens_in = sum(r["usage"].get("input_tokens", 0) for r in rows)
    tokens_out = sum(r["usage"].get("output_tokens", 0) for r in rows)
    lines += ["", f"Tokens: {tokens_in} in, {tokens_out} out"]
    return "\n".join(lines)
