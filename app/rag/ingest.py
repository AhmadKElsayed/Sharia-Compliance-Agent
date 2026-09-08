"""Corpus ingestion: load, chunk, embed, and upsert into the vector store.

Also holds the factories that assemble an embedder and store from settings, so
the API and the ingest CLI construct them the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.rag.chunking import Chunk, chunk_corpus
from app.rag.embedder import Embedder, OpenRouterEmbedder
from app.rag.store import QdrantStore, VectorStore

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Construct the configured embedder."""
    s = settings or get_settings()
    return OpenRouterEmbedder(
        api_key=s.openrouter_api_key,
        model=s.openrouter_embedding_model,
        dim=s.embedding_dim,
        base_url=s.openrouter_base_url,
        timeout=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,
    )


def build_store(settings: Settings | None = None) -> VectorStore:
    """Construct the configured vector store."""
    s = settings or get_settings()
    return QdrantStore(
        url=s.qdrant_url,
        api_key=s.qdrant_api_key,
        collection=s.qdrant_collection,
        timeout=s.qdrant_timeout_seconds,
    )


@dataclass
class IngestReport:
    """Outcome of an ingest run."""

    documents: int
    chunks: int
    upserted: int
    total_in_store: int
    embedding_model: str
    collection: str
    recreated: bool

    def summary(self) -> str:
        action = "recreated" if self.recreated else "updated"
        return (
            f"{action} collection {self.collection!r}: "
            f"{self.documents} documents -> {self.chunks} chunks -> "
            f"{self.upserted} upserted ({self.total_in_store} total in store) "
            f"using {self.embedding_model}"
        )


def ingest(
    store: VectorStore,
    embedder: Embedder,
    corpus_dir: Path = CORPUS_DIR,
    recreate: bool = False,
) -> IngestReport:
    """Chunk the corpus, embed it, and write it to the store.

    Idempotent: point IDs derive from chunk IDs, so re-running replaces in place
    rather than duplicating.
    """
    chunks: list[Chunk] = chunk_corpus(corpus_dir)
    if not chunks:
        raise RuntimeError(f"no chunks produced from {corpus_dir}")

    store.ensure_collection(embedder.dim, recreate=recreate)
    vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
    upserted = store.upsert(chunks, vectors)

    return IngestReport(
        documents=len({c.doc_id for c in chunks}),
        chunks=len(chunks),
        upserted=upserted,
        total_in_store=store.count(),
        embedding_model=getattr(embedder, "model", type(embedder).__name__),
        collection=getattr(store, "collection", "in-memory"),
        recreated=recreate,
    )
