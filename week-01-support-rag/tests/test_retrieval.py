import numpy as np
import pytest
from fakes import HashEmbedder

from supportrag.embeddings import OpenAIEmbedder, normalize
from supportrag.retrieval import Retriever, keyword_query, reciprocal_rank_fusion


def test_keyword_query_drops_stopwords_and_quotes_terms():
    assert keyword_query("How do I cancel my subscription?") == '"cancel" OR "subscription"'


def test_keyword_query_neutralises_fts_syntax():
    # Quotes, operators and punctuation in user text must not reach FTS5 as syntax.
    assert keyword_query('what\'s "NOT" x-plausible-dropped: *') == '"s" OR "not" OR "x" OR "plausible" OR "dropped"'
    assert keyword_query("???") == ""


def test_bm25_finds_exact_identifier(retriever):
    hits = retriever.search("x-plausible-dropped", k=3)
    assert hits[0].chunk.doc_path == "events-api.md"


def test_bm25_stems_words(retriever):
    # "cancelling" should match "cancel" through the porter tokenizer.
    assert retriever.search("cancelling", k=1)[0].chunk.doc_path == "cancel-subscription.md"


def test_bm25_returns_nothing_for_unknown_words(retriever):
    assert retriever.search("zeppelin", k=3) == []
    assert retriever.search("???", k=3) == []


def test_rrf_prefers_items_ranked_well_by_both_lists():
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 2, 9]})
    order = [item for item, _, _ in fused]
    # 2 and 3 appear in both lists; 1 and 9 lead one list each but appear only once.
    assert set(order[:2]) == {2, 3}
    assert fused[0][2] in ({"a": 2, "b": 2}, {"a": 3, "b": 1})


def test_rrf_with_one_ranking_keeps_its_order():
    assert [item for item, _, _ in reciprocal_rank_fusion({"only": [7, 5, 6]})] == [7, 5, 6]


def test_vector_search_ranks_by_cosine_similarity(embedded_store):
    retriever = Retriever(embedded_store, HashEmbedder(), mode="vector")
    hits = retriever.search("exclude visits by IP address", k=2)
    assert hits[0].chunk.heading == "Exclude visits by IP address"
    assert list(hits[0].ranks) == ["vector"]


def test_hybrid_records_which_retriever_found_each_hit(embedded_store):
    retriever = Retriever(embedded_store, HashEmbedder(), mode="hybrid")
    hits = retriever.search("cancel plan subscription", k=3)
    assert hits[0].chunk.doc_path == "cancel-subscription.md"
    assert set(hits[0].ranks) == {"bm25", "vector"}


def test_vector_search_without_embeddings_explains_what_to_do(store):
    with pytest.raises(RuntimeError, match="supportrag embed"):
        Retriever(store, HashEmbedder(), mode="vector").search("anything")


def test_vector_search_rejects_a_different_embedding_model(embedded_store):
    other = HashEmbedder()
    other.model = "another-model"
    with pytest.raises(RuntimeError, match="not comparable"):
        Retriever(embedded_store, other, mode="vector").search("anything")


def test_reindexing_discards_stale_embeddings(embedded_store):
    embedded_store.replace_chunks([c for _, c in embedded_store.all_chunks()])
    rowids, _ = embedded_store.load_embeddings()
    assert len(rowids) == 0 and embedded_store.get_meta("embedding_model") is None


def test_normalize_makes_unit_rows_and_tolerates_zero_rows():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_openai_embedder_batches_and_keeps_input_order():
    class Item:
        def __init__(self, index, embedding):
            self.index, self.embedding = index, embedding

    class Embeddings:
        calls: list[list[str]] = []

        def create(self, model, input):
            self.calls.append(input)
            # Return items out of order to prove the embedder sorts by index.
            items = [Item(i, [float(len(text)), 1.0]) for i, text in enumerate(input)]
            return type("R", (), {"data": list(reversed(items))})()

    class Client:
        embeddings = Embeddings()

    vectors = OpenAIEmbedder(Client(), "m", batch_size=2).embed(["a", "bbb", "cc"])
    assert [len(call) for call in Client.embeddings.calls] == [2, 1]
    assert vectors.shape == (3, 2)
    assert np.allclose(vectors[1], normalize(np.array([[3.0, 1.0]]))[0])
