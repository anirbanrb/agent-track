from fakes import FakeClient, message, response, tool_call

from supportrag.agent import TOOLS, SupportAgent, parse_citations


def make_agent(retriever, script, max_steps=5):
    client = FakeClient(script)
    return SupportAgent(client, retriever, model="fake", max_steps=max_steps, top_k=3), client


def test_search_then_answer_with_citation(retriever):
    agent, client = make_agent(
        retriever,
        [
            response(tool_call("search_docs", {"query": "cancel subscription"})),
            response(message("Open Account Settings and click Cancel plan [1].")),
        ],
    )
    result = agent.answer("How do I cancel?")

    assert result.answer.endswith("[1].")
    assert result.cited[0].chunk.doc_path == "cancel-subscription.md"
    assert result.grounded and not result.escalated and not result.stopped_early
    assert result.steps == 2
    assert result.usage == {"input_tokens": 200, "output_tokens": 40}
    assert result.trace[0]["query"] == "cancel subscription"


def test_tool_output_is_sent_back_with_the_matching_call_id(retriever):
    agent, client = make_agent(
        retriever,
        [
            response(tool_call("search_docs", {"query": "cancel"}, call_id="call_abc")),
            response(message("Done [1].")),
        ],
    )
    agent.answer("How do I cancel?")

    second_request = client.responses.requests[1]["input"]
    # user message, the model's function_call, our function_call_output
    assert second_request[0] == {"role": "user", "content": "How do I cancel?"}
    assert second_request[1].type == "function_call"
    assert second_request[2]["type"] == "function_call_output"
    assert second_request[2]["call_id"] == "call_abc"
    assert '<source number="1"' in second_request[2]["output"]


def test_source_numbers_are_stable_across_searches(retriever):
    agent, _ = make_agent(
        retriever,
        [
            response(tool_call("search_docs", {"query": "cancel subscription"})),
            response(tool_call("search_docs", {"query": "cancel plan billing period"})),
            response(message("You keep access until the period ends [2].")),
        ],
    )
    result = agent.answer("What happens when I cancel?")

    numbers = [s.number for s in result.sources]
    assert numbers == sorted(set(numbers))  # no duplicates, no gaps in ordering
    chunk_ids = [s.chunk.chunk_id for s in result.sources]
    assert len(chunk_ids) == len(set(chunk_ids))  # the same passage is registered once


def test_citation_of_a_source_that_was_never_retrieved_is_flagged(retriever):
    agent, _ = make_agent(
        retriever,
        [
            response(tool_call("search_docs", {"query": "cancel"})),
            response(message("Refunds take 5 days [9].")),
        ],
    )
    result = agent.answer("Refund?")
    assert result.invalid_citations == [9]
    assert not result.grounded


def test_answer_without_citations_is_not_grounded(retriever):
    agent, _ = make_agent(retriever, [response(message("Probably in the settings somewhere."))])
    result = agent.answer("How do I cancel?")
    assert result.cited == [] and not result.grounded


def test_escalation_is_recorded_from_the_tool_call(retriever):
    agent, _ = make_agent(
        retriever,
        [
            response(tool_call("search_docs", {"query": "refund policy"})),
            response(tool_call("escalate_to_human", {"reason": "not_in_docs", "summary": "Wants a refund."})),
            response(message("I've passed this to our team; someone will follow up.")),
        ],
    )
    result = agent.answer("Can I have a refund?")
    assert result.escalated
    assert result.escalation == {"reason": "not_in_docs", "summary": "Wants a refund."}
    assert result.grounded  # an escalation makes no claims, so it needs no citations


def test_claiming_to_escalate_without_calling_the_tool_does_not_count(retriever):
    agent, _ = make_agent(retriever, [response(message("I have escalated this to a human."))])
    assert not agent.answer("Refund?").escalated


def test_step_budget_forces_a_final_answer_without_tools(retriever):
    script = [response(tool_call("search_docs", {"query": f"cancel {n}"})) for n in range(2)]
    script.append(response(message("Best effort answer [1].")))
    agent, client = make_agent(retriever, script, max_steps=2)

    result = agent.answer("How do I cancel?")

    assert result.stopped_early and result.steps == 3
    choices = [request["tool_choice"] for request in client.responses.requests]
    assert choices == ["auto", "auto", "none"]
    # The tool list itself never changes between requests.
    assert all(request["tools"] is TOOLS for request in client.responses.requests)


def test_tool_errors_are_returned_to_the_model_not_raised(retriever):
    agent, client = make_agent(
        retriever,
        [
            response(
                tool_call("search_docs", "{not json", call_id="c1"),
                tool_call("search_docs", {"query": "  "}, call_id="c2"),
                tool_call("delete_everything", {}, call_id="c3"),
            ),
            response(message("Sorry, let me try again.")),
        ],
    )
    result = agent.answer("anything")

    outputs = [item["output"] for item in client.responses.requests[1]["input"][-3:]]
    assert outputs[0].startswith("Error: arguments were not valid JSON")
    assert outputs[1] == "Error: `query` must not be empty."
    assert outputs[2].startswith("Error: unknown tool 'delete_everything'")
    assert [event.get("error") for event in result.trace] == ["invalid JSON arguments", "empty query", "unknown tool"]


def test_search_with_no_results_tells_the_model_what_to_do(retriever):
    agent, client = make_agent(
        retriever,
        [response(tool_call("search_docs", {"query": "zeppelin"})), response(message("I could not find that."))],
    )
    agent.answer("zeppelin?")
    assert client.responses.requests[1]["input"][-1]["output"].startswith("No results.")


def test_parse_citations():
    assert parse_citations("A [1]. B [2][3]. C [1, 4].") == [1, 2, 3, 4]
    assert parse_citations("No citations, but an array index a[i] and [text].") == []


def test_tool_schemas_meet_strict_mode_requirements():
    # Strict mode: every property required, additionalProperties false.
    for tool in TOOLS:
        params = tool["parameters"]
        assert tool["strict"] is True
        assert params["additionalProperties"] is False
        assert sorted(params["required"]) == sorted(params["properties"])
