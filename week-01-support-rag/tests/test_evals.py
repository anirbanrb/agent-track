from fakes import FakeClient, message, response, tool_call

from supportrag.agent import SupportAgent
from supportrag.evals import (
    format_answer_report,
    format_retrieval_report,
    grade_answer,
    mentions,
    run_answer_eval,
    run_retrieval_eval,
)


def test_retrieval_eval_scores_by_page_not_chunk(retriever):
    cases = [
        {"id": "r1", "kind": "direct", "question": "cancel subscription", "expected_docs": ["cancel-subscription.md"]},
        {"id": "r2", "kind": "identifier", "question": "x-plausible-dropped", "expected_docs": ["events-api.md"]},
        {"id": "r3", "kind": "paraphrase", "question": "zeppelin", "expected_docs": ["excluding.md"]},
    ]
    report = run_retrieval_eval(cases, lambda q, k: retriever.search(q, k=k), k=3)

    assert [row["rank"] for row in report.rows] == [1, 1, None]
    assert report.hit_rate(1) == 2 / 3
    assert report.mrr() == 2 / 3
    assert report.hit_rate(3, kind="paraphrase") == 0.0
    text = format_retrieval_report(report, "bm25")
    assert "Misses (1):" in text and "r3" in text


def test_mentions_matches_whole_phrases_and_ignores_citations():
    assert mentions("You can import 5 properties [3].", "5")
    assert not mentions("Sessions last 15 days [5].", "5")
    assert mentions("Check the X-Plausible-Dropped header.", "x-plausible-dropped")


def run(retriever, script, question="How do I cancel?"):
    return SupportAgent(FakeClient(script), retriever, model="fake").answer(question)


def test_grading_a_correct_answer(retriever):
    result = run(
        retriever,
        [
            response(tool_call("search_docs", {"query": "cancel subscription"})),
            response(message("Go to Account Settings and click Cancel plan [1].")),
        ],
    )
    case = {"expected_docs": ["cancel-subscription.md"], "must_mention": [["Account Settings"], ["Cancel plan", "cancel"]]}
    assert all(grade_answer(case, result).values())


def test_grading_catches_a_confident_answer_that_should_have_escalated(retriever):
    result = run(
        retriever,
        [
            response(tool_call("search_docs", {"query": "refund"})),
            response(message("Refunds are available within 30 days [1].")),
        ],
    )
    checks = grade_answer({"should_escalate": True}, result)
    assert checks["escalation_correct"] is False


def test_grading_catches_needless_escalation_and_missing_fact(retriever):
    result = run(
        retriever,
        [
            response(tool_call("escalate_to_human", {"reason": "other", "summary": "?"})),
            response(message("Someone will be in touch.")),
        ],
    )
    checks = grade_answer({"expected_docs": ["cancel-subscription.md"], "must_mention": [["Account Settings"]]}, result)
    assert checks["escalation_correct"] is False
    assert checks["has_citation"] is False and checks["mentions_1"] is False


def test_answer_eval_report_lists_failures(retriever):
    scripts = iter(
        [
            [response(tool_call("search_docs", {"query": "cancel"})), response(message("Click Cancel plan [1]."))],
            [response(message("It is sunny."))],
        ]
    )
    cases = [
        {"id": "a1", "question": "How do I cancel?", "expected_docs": ["cancel-subscription.md"]},
        {"id": "e1", "question": "Refund me", "should_escalate": True},
    ]
    rows = run_answer_eval(cases, lambda q: run(retriever, next(scripts), q))

    assert [row["passed"] for row in rows] == [True, False]
    text = format_answer_report(rows)
    assert "1/2 cases passed" in text and "e1: escalation_correct" in text
