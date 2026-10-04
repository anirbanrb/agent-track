"""Wire-level tests: the real OpenAI SDK against a fake HTTP server.

The other agent tests replace the client object. These keep the real client
and replace only the network, so they check what actually goes over the
wire: endpoint paths, the JSON body of each request, and that the SDK can
parse a response shaped like the documented one.
"""

import json

import numpy as np

try:  # OpenAI SDK 3.x ships on httpx2; older versions use httpx.
    import httpx2 as httpx
except ImportError:
    import httpx
from openai import OpenAI

from supportrag.agent import SupportAgent
from supportrag.embeddings import OpenAIEmbedder


def envelope(output):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-test",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 50,
            "output_tokens": 10,
            "total_tokens": 60,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "search_docs",
    "arguments": json.dumps({"query": "cancel subscription"}),
    "status": "completed",
}
FINAL_MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Click Cancel plan [1].", "annotations": []}],
}


def client_with(handler) -> OpenAI:
    return OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)


def test_agent_round_trip_through_the_real_sdk(retriever):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=envelope([FUNCTION_CALL] if len(bodies) == 1 else [FINAL_MESSAGE]))

    result = SupportAgent(client_with(handler), retriever, model="gpt-test").answer("How do I cancel?")

    assert result.answer == "Click Cancel plan [1]." and result.grounded
    assert result.usage == {"input_tokens": 100, "output_tokens": 20}

    first, second = bodies
    assert first["model"] == "gpt-test" and first["tool_choice"] == "auto"
    assert [tool["name"] for tool in first["tools"]] == ["search_docs", "escalate_to_human"]
    assert first["input"] == [{"role": "user", "content": "How do I cancel?"}]

    # Second request: the model's own function_call is echoed back, followed
    # by our output for the same call_id.
    echoed, output = second["input"][1], second["input"][2]
    assert echoed["type"] == "function_call" and echoed["call_id"] == "call_1"
    assert output["type"] == "function_call_output" and output["call_id"] == "call_1"
    assert "Cancel plan" in output["output"]


def test_embedder_through_the_real_sdk():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        body = json.loads(request.content)
        assert body["model"] == "embed-test"
        data = [{"object": "embedding", "index": i, "embedding": [3.0, 4.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"object": "list", "model": "embed-test", "data": data, "usage": {"prompt_tokens": 2, "total_tokens": 2}})

    vectors = OpenAIEmbedder(client_with(handler), "embed-test").embed(["a", "b"])
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors, [[0.6, 0.8], [0.6, 0.8]])
