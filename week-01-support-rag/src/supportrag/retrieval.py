"""Retrieval: keyword search, vector search, and a fusion of the two.

The two methods fail in opposite ways, which is why they are combined:

- Keyword search (BM25) is exact. It finds "x-plausible-dropped" or "SAML"
  instantly, and finds nothing when the user says "stop paying" and the
  docs say "cancel your subscription".
- Vector search matches meaning. It handles the paraphrase, and blurs exact
  identifiers, error codes and product names into their neighbours.

Reciprocal rank fusion merges the two rankings using only rank positions,
so it needs no score calibration between two unrelated scoring systems.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from supportrag.chunking import Chunk
from supportrag.embeddings import Embedder
from supportrag.store import Store

Mode = Literal["bm25", "vector", "hybrid"]

# Function words that carry no topic. Dropped from keyword queries so that
# "how do I cancel" searches for "cancel", not for every chunk containing "how".
STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from how i if in is it its me my of on or
    our so that the their then there this to was we what when where which who why will with you your""".split()
)

RRF_K = 60  # The constant from the original RRF paper; damps the top ranks.


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    # Rank of this chunk in each underlying ranking (1 = best). Kept for
    # debugging: it shows which retriever found the chunk.
    ranks: dict[str, int] = field(default_factory=dict)


def keyword_query(question: str) -> str:
    """Turn free text into a safe FTS5 expression.

    FTS5 has its own query syntax, so raw user text can be a syntax error
    ("what's" contains a quote) or change meaning ("NOT", "*", ":").
    Extracting plain terms, quoting each and joining with OR removes both
    problems. OR, not AND: BM25 already rewards chunks matching more terms,
    and AND would return nothing as soon as one word is absent.
    """
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", question.lower()) if t not in STOPWORDS]
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse several rankings of ids into one, best first.

    Each ranking contributes 1 / (k + rank) for every id it contains. An id
    ranked highly by both lists beats an id ranked first by only one.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ranking in rankings.items():
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
            ranks.setdefault(item, {})[name] = position
    ordered = sorted(scores, key=lambda item: (-scores[item], item))
    return [(item, scores[item], ranks[item]) for item in ordered]


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        mode: Mode = "hybrid",
        candidates: int = 20,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self.mode: Mode = mode
        self._candidates = candidates
        self._rowids: np.ndarray | None = None
        self._matrix: np.ndarray | None = None

    # -- the two base rankings ------------------------------------------------

    def _keyword_ranking(self, question: str, limit: int) -> list[int]:
        expr = keyword_query(question)
        return self._store.keyword_search(expr, limit) if expr else []

    def _vector_ranking(self, question: str, limit: int) -> list[int]:
        if self._embedder is None:
            raise RuntimeError("Vector search needs an embedder. Use mode='bm25' or pass one.")
        if self._matrix is None:
            self._rowids, self._matrix = self._store.load_embeddings()
            if len(self._rowids) == 0:
                raise RuntimeError("No embeddings in the index. Run `supportrag embed` first.")
            indexed_with = self._store.get_meta("embedding_model")
            if indexed_with != self._embedder.model:
                raise RuntimeError(
                    f"Index was embedded with {indexed_with!r} but queries use "
                    f"{self._embedder.model!r}. Vectors from different models are not comparable."
                )
        query = self._embedder.embed([question])[0]
        scores = self._matrix @ query  # cosine similarity: both sides are unit length
        limit = min(limit, len(scores))
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [int(self._rowids[i]) for i in top]

    # -- public ---------------------------------------------------------------

    def search(self, question: str, k: int = 5, mode: Mode | None = None) -> list[Hit]:
        mode = mode or self.mode
        rankings: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid"):
            rankings["bm25"] = self._keyword_ranking(question, self._candidates)
        if mode in ("vector", "hybrid"):
            rankings["vector"] = self._vector_ranking(question, self._candidates)

        fused = reciprocal_rank_fusion(rankings)[:k]
        chunks = self._store.get_chunks([rowid for rowid, _, _ in fused])
        return [Hit(chunk=chunks[rowid], score=score, ranks=ranks) for rowid, score, ranks in fused]
