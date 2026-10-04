"""Storage: one SQLite file holds the chunks, the keyword index and the vectors.

Why not a vector database? At 925 chunks, comparing a query against every
vector is one matrix multiplication and takes well under a millisecond.
A vector database earns its place when the vectors stop fitting in memory or
you need filtering and concurrent writes at scale. Until then it is
infrastructure with nothing to do.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from supportrag.chunking import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    chunk_id  TEXT UNIQUE NOT NULL,
    doc_path  TEXT NOT NULL,
    title     TEXT NOT NULL,
    heading   TEXT NOT NULL,
    url       TEXT NOT NULL,
    text      TEXT NOT NULL
);

-- FTS5 is SQLite's built-in full-text index. "External content" means it
-- indexes the chunks table instead of storing a second copy of the text.
-- The porter tokenizer stems words, so "cancelling" matches "cancel".
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, heading, text,
    content='chunks', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_rowid INTEGER PRIMARY KEY REFERENCES chunks(id),
    vector      BLOB NOT NULL  -- float32 bytes
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: Path | str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    # -- writes ------------------------------------------------------------

    def replace_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Rebuild the index from scratch.

        Re-ingesting everything is the simplest correct behaviour: no stale
        chunks, no orphaned vectors. Incremental updates are an optimisation
        to add when a full rebuild gets slow, and not before.
        """
        rows = [(c.chunk_id, c.doc_path, c.title, c.heading, c.url, c.text) for c in chunks]
        with self._db:
            self._db.execute("DELETE FROM embeddings")
            self._db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('delete-all')")
            self._db.execute("DELETE FROM chunks")
            self._db.executemany(
                "INSERT INTO chunks (chunk_id, doc_path, title, heading, url, text) VALUES (?,?,?,?,?,?)",
                rows,
            )
            self._db.execute(
                "INSERT INTO chunks_fts (rowid, title, heading, text) SELECT id, title, heading, text FROM chunks"
            )
            self._db.execute("DELETE FROM meta WHERE key = 'embedding_model'")
        return len(rows)

    def set_embeddings(self, model: str, rowids: Sequence[int], matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=np.float32)
        with self._db:
            self._db.execute("DELETE FROM embeddings")
            self._db.executemany(
                "INSERT INTO embeddings (chunk_rowid, vector) VALUES (?, ?)",
                [(int(rowid), row.tobytes()) for rowid, row in zip(rowids, matrix)],
            )
        # Vectors from different models are not comparable. Record which model
        # produced them so a query embedded with another model fails loudly.
        self.set_meta("embedding_model", model)

    def set_meta(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # -- reads -------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def count_chunks(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def all_chunks(self) -> list[tuple[int, Chunk]]:
        rows = self._db.execute(
            "SELECT id, chunk_id, doc_path, title, heading, url, text FROM chunks ORDER BY id"
        ).fetchall()
        return [(row[0], Chunk(*row[1:])) for row in rows]

    def get_chunks(self, rowids: Sequence[int]) -> dict[int, Chunk]:
        if not rowids:
            return {}
        marks = ",".join("?" * len(rowids))
        rows = self._db.execute(
            f"SELECT id, chunk_id, doc_path, title, heading, url, text FROM chunks WHERE id IN ({marks})",
            [int(r) for r in rowids],
        ).fetchall()
        return {row[0]: Chunk(*row[1:]) for row in rows}

    def load_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (rowids, matrix). The matrix is empty if nothing is embedded."""
        rows = self._db.execute("SELECT chunk_rowid, vector FROM embeddings ORDER BY chunk_rowid").fetchall()
        if not rows:
            return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        rowids = np.array([r[0] for r in rows], dtype=np.int64)
        matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return rowids, matrix

    def keyword_search(self, match_expr: str, limit: int) -> list[int]:
        """Row ids ranked by BM25, best first.

        The three weights apply to the title, heading and text columns: a
        query term in a heading is stronger evidence than one in a body.
        """
        rows = self._db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts, 2.0, 3.0, 1.0) LIMIT ?",
            (match_expr, limit),
        ).fetchall()
        return [row[0] for row in rows]
