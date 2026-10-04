from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import HashEmbedder  # noqa: E402

from supportrag.chunking import chunk_documents  # noqa: E402
from supportrag.corpus import Document  # noqa: E402
from supportrag.retrieval import Retriever  # noqa: E402
from supportrag.store import Store  # noqa: E402

DOCS = [
    Document(
        path="cancel-subscription.md",
        title="Cancel your subscription plan",
        url="https://example.test/docs/cancel-subscription",
        description="",
        body=(
            "Go to Account Settings, open the Subscription section and click Cancel plan.\n\n"
            "## What happens after you cancel\n\n"
            "You keep full access until the end of your billing period.\n"
        ),
    ),
    Document(
        path="excluding.md",
        title="Exclude your own visits",
        url="https://example.test/docs/excluding",
        description="",
        body=(
            "## Exclude visits by IP address\n\n"
            "Open site settings, choose Shields, then IP Addresses, and add your address.\n\n"
            "## Exclude visits by browser\n\n"
            "Set the `localStorage.plausible_ignore` flag in the browser console.\n"
        ),
    ),
    Document(
        path="events-api.md",
        title="Events API reference",
        url="https://example.test/docs/events-api",
        description="",
        body=(
            "## Debugging\n\n"
            "The API returns HTTP 202 even when an event is dropped. "
            "Check the `x-plausible-dropped` response header.\n"
        ),
    ),
]


@pytest.fixture
def store() -> Store:
    store = Store(":memory:")
    store.replace_chunks(chunk_documents(DOCS))
    return store


@pytest.fixture
def embedded_store(store: Store) -> Store:
    rows = store.all_chunks()
    embedder = HashEmbedder()
    store.set_embeddings(embedder.model, [rowid for rowid, _ in rows], embedder.embed([c.embed_text for _, c in rows]))
    return store


@pytest.fixture
def retriever(store: Store) -> Retriever:
    return Retriever(store, mode="bm25")
