"""Test doubles. No test in this suite touches the network."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)

from supportrag.embeddings import normalize


class HashEmbedder:
    """Deterministic bag-of-words vectors.

    Not semantic: it only knows that texts sharing words are similar. That is
    enough to exercise the vector code path (storage, normalisation, ranking,
    fusion) without a model.
    """

    model = "hash-test"

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim
                matrix[row, bucket] += 1.0
        return normalize(matrix)


# The fakes below are built from the SDK's own response types, so a field
# renamed in the SDK breaks these tests instead of breaking production.


def tool_call(name: str, arguments: dict[str, Any] | str, call_id: str = "call_1") -> ResponseFunctionToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ResponseFunctionToolCall(type="function_call", name=name, arguments=raw, call_id=call_id)


def message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def response(*output: Any) -> Response:
    return Response(
        id="resp_1",
        created_at=0,
        model="fake",
        object="response",
        output=list(output),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        # model_construct skips validation: the nested token-detail fields
        # differ between SDK versions and the agent only reads the two totals.
        usage=ResponseUsage.model_construct(input_tokens=100, output_tokens=20, total_tokens=120),
    )


class _Responses:
    def __init__(self, script: list[Response]) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Response:
        # Snapshot the input: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        if not self._script:
            raise AssertionError("The agent made more model calls than the test scripted.")
        return self._script.pop(0)


class FakeClient:
    """Stands in for `OpenAI()`: replays a scripted list of responses."""

    def __init__(self, script: list[Response]) -> None:
        self.responses = _Responses(script)
