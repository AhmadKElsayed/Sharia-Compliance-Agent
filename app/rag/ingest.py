"""Corpus ingestion: load, chunk, embed, and upsert into the vector store.

Also holds the factories that assemble an embedder and store from settings, so
the API and the ingest CLI construct them the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.rag.chunking import Chunk, chunk_documents
from app.rag.embedder import Embedder, OpenRouterEmbedder
from app.rag.pdf_loader import load_pdf_corpus
from app.rag.store import QdrantStore, VectorStore

# The pipeline ingests the PDFs, which are the corpus as it is circulated.
# corpus/source holds the markdown they are built from.
CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus" / "pdf"

# The AAOIFI volume is licensed and git-ignored, so a fresh clone will not have
# it -- download it from https://aaoifi.com/shariah-standards-3/?lang=en and drop
# it here. Ingest prefers it when present and falls back to the demo corpus,
# which keeps the repository reproducible for anyone without a copy.
AAOIFI_PDF = CORPUS_DIR / "Shariaa-Standards-ENG.pdf"
DEMO_DIR = Path(__file__).resolve().parents[2] / "corpus" / "demo"


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> tuple[list[Chunk], str, int]:
    """Load corpus chunks, returning them with the source name and doc count."""
    aaoifi = corpus_dir / AAOIFI_PDF.name
    if aaoifi.exists():
        from app.rag.aaoifi import extract_chunks

        chunks = extract_chunks(aaoifi)
        return chunks, "AAOIFI Shari'ah Standards (EN)", len({c.doc_id for c in chunks})

    documents = load_pdf_corpus(DEMO_DIR)
    return chunk_documents(documents), "demo corpus", len(documents)


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
    source: str = ""

    def summary(self) -> str:
        action = "recreated" if self.recreated else "updated"
        return (
            f"{action} collection {self.collection!r}: "
            f"{self.source}: {self.documents} documents -> {self.chunks} chunks -> "
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
    chunks, source, doc_count = load_chunks(corpus_dir)
    if not chunks:
        raise RuntimeError(f"no chunks produced from {corpus_dir}")

    store.ensure_collection(embedder.dim, recreate=recreate)
    vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
    upserted = store.upsert(chunks, vectors)

    return IngestReport(
        documents=doc_count,
        chunks=len(chunks),
        upserted=upserted,
        total_in_store=store.count(),
        embedding_model=getattr(embedder, "model", type(embedder).__name__),
        collection=getattr(store, "collection", "in-memory"),
        recreated=recreate,
        source=source,
    )
