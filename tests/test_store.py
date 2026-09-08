"""Contract tests for the vector store and the offline embedder.

These run against ``InMemoryStore`` so the default suite needs no credentials.
``QdrantStore`` implements the same protocol; the live smoke test in
``test_live_smoke.py`` exercises it against the real cluster.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.chunking import chunk_corpus
from app.rag.embedder import Embedder, HashingEmbedder, OpenRouterEmbedder
from app.rag.store import InMemoryStore, QdrantStore, SearchHit, VectorStore

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="module")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture()
def populated(embedder: HashingEmbedder) -> InMemoryStore:
    store = InMemoryStore()
    store.ensure_collection(embedder.dim)
    chunks = chunk_corpus(CORPUS_DIR)
    store.upsert(chunks, embedder.embed_documents([c.embedding_text() for c in chunks]))
    return store


# --- protocol conformance ------------------------------------------------


def test_both_stores_satisfy_the_protocol() -> None:
    assert isinstance(InMemoryStore(), VectorStore)
    assert issubclass(QdrantStore, VectorStore.__mro__[0]) or hasattr(
        QdrantStore, "search"
    )
    for method in ("ensure_collection", "upsert", "search", "count", "ping"):
        assert callable(getattr(QdrantStore, method))


def test_both_embedders_satisfy_the_protocol() -> None:
    assert isinstance(HashingEmbedder(), Embedder)
    for method in ("embed_documents", "embed_query"):
        assert callable(getattr(OpenRouterEmbedder, method))


# --- embedder ------------------------------------------------------------


def test_hashing_embedder_is_deterministic(embedder: HashingEmbedder) -> None:
    assert embedder.embed_query("riba") == embedder.embed_query("riba")


def test_hashing_embedder_returns_unit_vectors(embedder: HashingEmbedder) -> None:
    vec = embedder.embed_query("guaranteed fixed return on a deposit")
    assert pytest.approx(sum(v * v for v in vec), abs=1e-9) == 1.0


def test_hashing_embedder_handles_empty_text(embedder: HashingEmbedder) -> None:
    vec = embedder.embed_query("!!!")
    assert len(vec) == embedder.dim
    assert pytest.approx(sum(v * v for v in vec), abs=1e-9) == 1.0


def test_hashing_embedder_captures_lexical_similarity(
    embedder: HashingEmbedder,
) -> None:
    def cos(a: str, b: str) -> float:
        va, vb = embedder.embed_query(a), embedder.embed_query(b)
        return sum(x * y for x, y in zip(va, vb, strict=True))

    related = cos("guaranteed fixed return deposit", "fixed return guaranteed deposit")
    unrelated = cos("guaranteed fixed return deposit", "leasing maintenance obligations")
    assert related > unrelated


def test_openrouter_embedder_rejects_missing_key() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbedder(api_key="", model="m", dim=1536)


# --- store contract ------------------------------------------------------


def test_upsert_stores_every_chunk(populated: InMemoryStore) -> None:
    assert populated.count() == len(chunk_corpus(CORPUS_DIR))


def test_upsert_is_idempotent(embedder: HashingEmbedder, populated: InMemoryStore) -> None:
    before = populated.count()
    chunks = chunk_corpus(CORPUS_DIR)
    populated.upsert(chunks, embedder.embed_documents([c.embedding_text() for c in chunks]))
    assert populated.count() == before, "re-ingesting must not duplicate points"


def test_upsert_rejects_mismatched_counts(populated: InMemoryStore) -> None:
    chunks = chunk_corpus(CORPUS_DIR)
    with pytest.raises(ValueError, match="mismatch"):
        populated.upsert(chunks, [[0.0] * 256])


def test_ensure_collection_rejects_dimension_change(populated: InMemoryStore) -> None:
    with pytest.raises(RuntimeError, match="dim"):
        populated.ensure_collection(1536)


def test_recreate_clears_the_store(populated: InMemoryStore) -> None:
    populated.ensure_collection(256, recreate=True)
    assert populated.count() == 0


def test_search_respects_top_k(embedder: HashingEmbedder, populated: InMemoryStore) -> None:
    hits = populated.search(embedder.embed_query("riba"), top_k=3)
    assert len(hits) == 3


def test_search_returns_descending_scores(
    embedder: HashingEmbedder, populated: InMemoryStore
) -> None:
    hits = populated.search(embedder.embed_query("murabaha ownership"), top_k=5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_threshold_filters(
    embedder: HashingEmbedder, populated: InMemoryStore
) -> None:
    query = embedder.embed_query("ijarah maintenance")
    assert populated.search(query, top_k=10, score_threshold=0.99) == []


def test_hits_are_directly_citable(
    embedder: HashingEmbedder, populated: InMemoryStore
) -> None:
    hit = populated.search(embedder.embed_query("riba deposit"), top_k=1)[0]
    assert isinstance(hit, SearchHit)
    assert hit.doc_id.startswith("SFS-")
    assert hit.citation.startswith(hit.doc_id)
    assert hit.text, "a hit must carry its own text; the container ships no corpus"
    assert hit.to_log()["citation"] == hit.citation


def test_empty_store_searches_cleanly() -> None:
    store = InMemoryStore()
    store.ensure_collection(8)
    assert store.search([0.0] * 8, top_k=5) == []
    assert store.count() == 0
    assert store.ping() is True


# --- retrieval quality (offline proxy) -----------------------------------


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("guaranteed fixed return on a savings deposit", "SFS-001"),
        ("cost plus sale ownership disclosure", "SFS-002"),
        ("lease maintenance and risk of loss", "SFS-003"),
        ("gambling speculation and conventional insurance", "SFS-004"),
        ("profit sharing ratio and loss attribution", "SFS-005"),
        ("debt ratio screening threshold for a fund", "SFS-006"),
    ],
)
def test_golden_queries_retrieve_the_right_document(
    embedder: HashingEmbedder, populated: InMemoryStore, query: str, expected_doc: str
) -> None:
    """Lexical proxy for retrieval quality.

    The hashing embedder is not semantic, so this guards chunking and plumbing
    rather than embedding quality; the live smoke test covers the real model.
    """
    hits = populated.search(embedder.embed_query(query), top_k=3)
    assert expected_doc in {h.doc_id for h in hits}, (
        f"{query!r} retrieved {[h.citation for h in hits]}"
    )
