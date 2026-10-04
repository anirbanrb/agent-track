"""Embeddings: text in, unit-length vectors out.

`Embedder` is a Protocol, not a base class. Anything with a `model` name and
an `embed` method qualifies, which is what lets the tests run the whole
vector path with a fake and no network.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim), rows unit length."""
        ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1.

    With unit vectors, cosine similarity is a plain dot product, so search
    becomes one matrix multiplication. Do it once at write time, not on
    every query.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class OpenAIEmbedder:
    def __init__(self, client, model: str, batch_size: int = 64) -> None:
        self._client = client
        self.model = model
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        # Batching is the difference between 15 requests and 925 for this corpus.
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.embeddings.create(model=self.model, input=batch)
            # The API returns one item per input, each tagged with its index.
            rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return normalize(np.array(rows, dtype=np.float32))
