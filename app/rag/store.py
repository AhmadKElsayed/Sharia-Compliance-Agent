"""Vector store abstraction with Qdrant and in-memory implementations.

The protocol exists for two reasons. It keeps the default test suite
credential-free and fast — ``InMemoryStore`` runs the same contract as Qdrant —
and it means swapping the backing store later touches one file rather than the
agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.rag.chunking import Chunk


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk, carrying everything needed to cite it.

    The payload is denormalised deliberately: a retrieval result must be
    directly citable without a second lookup against the corpus files, which the
    deployed container does not ship.
    """

    chunk_id: str
    score: float
    doc_id: str
    title: str
    heading: str
    section_label: str
    citation: str
    text: str

    def to_log(self) -> dict[str, Any]:
        """Compact form for structured logs."""
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "doc_id": self.doc_id,
            "citation": self.citation,
        }


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "topic": chunk.topic,
        "heading": chunk.heading,
        "section_label": chunk.section_label,
        "citation": chunk.citation,
        "text": chunk.text,
    }


def _hit(payload: dict[str, Any], score: float) -> SearchHit:
    return SearchHit(
        chunk_id=payload.get("chunk_id", ""),
        score=score,
        doc_id=payload.get("doc_id", ""),
        title=payload.get("title", ""),
        heading=payload.get("heading", ""),
        section_label=payload.get("section_label", ""),
        citation=payload.get("citation", ""),
        text=payload.get("text", ""),
    )


@runtime_checkable
class VectorStore(Protocol):
    """Storage and similarity search over corpus chunks."""

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        """Create the collection if absent, validating its dimension."""
        ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Insert or replace chunks. Returns the number written."""
        ...

    def search(
        self, vector: list[float], top_k: int, score_threshold: float | None = None
    ) -> list[SearchHit]:
        """Return the nearest chunks by cosine similarity."""
        ...

    def count(self) -> int:
        """Number of stored vectors."""
        ...

    def ping(self) -> bool:
        """True if the store is reachable."""
        ...


class QdrantStore:
    """Vector store backed by a Qdrant Cloud collection."""

    def __init__(
        self,
        url: str,
        api_key: str,
        collection: str,
        timeout: float = 20.0,
        client: QdrantClient | None = None,
    ) -> None:
        self._collection = collection
        self._client = client or QdrantClient(
            url=url, api_key=api_key, timeout=int(timeout)
        )

    @property
    def collection(self) -> str:
        return self._collection

    def _exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        if recreate and self._exists():
            self._client.delete_collection(self._collection)

        if not self._exists():
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=dim, distance=qmodels.Distance.COSINE
                ),
            )
            return

        # A dimension mismatch would otherwise surface as silently poor
        # retrieval rather than an error, so fail loudly instead.
        info = self._client.get_collection(self._collection)
        params = info.config.params.vectors
        existing = params.size if hasattr(params, "size") else None
        if existing is not None and existing != dim:
            raise RuntimeError(
                f"collection {self._collection!r} has {existing}-dim vectors but the "
                f"configured embedding model produces {dim}. Re-run with --recreate "
                f"to rebuild it."
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )
        if not chunks:
            return 0

        points = [
            qmodels.PointStruct(
                id=chunk.point_id(), vector=vector, payload=_payload(chunk)
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    def search(
        self, vector: list[float], top_k: int, score_threshold: float | None = None
    ) -> list[SearchHit]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [_hit(p.payload or {}, p.score) for p in response.points]

    def count(self) -> int:
        if not self._exists():
            return 0
        return self._client.count(self._collection, exact=True).count

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


class InMemoryStore:
    """In-process vector store used by tests and offline development.

    Exact cosine search over a Python list. At corpus scale (~30 chunks) this is
    not merely adequate but faster than a network call.
    """

    def __init__(self) -> None:
        self._dim: int | None = None
        self._points: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        if recreate:
            self._points.clear()
            self._dim = None
        if self._dim is not None and self._dim != dim:
            raise RuntimeError(
                f"store holds {self._dim}-dim vectors but {dim} was requested"
            )
        self._dim = dim

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )
        for chunk, vector in zip(chunks, vectors, strict=True):
            if self._dim is not None and len(vector) != self._dim:
                raise RuntimeError(
                    f"vector for {chunk.chunk_id} has {len(vector)} dims, expected {self._dim}"
                )
            self._points[chunk.point_id()] = (list(vector), _payload(chunk))
        return len(chunks)

    def search(
        self, vector: list[float], top_k: int, score_threshold: float | None = None
    ) -> list[SearchHit]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for stored, payload in self._points.values():
            score = sum(a * b for a, b in zip(vector, stored, strict=True))
            if score_threshold is None or score >= score_threshold:
                scored.append((score, payload))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_hit(payload, score) for score, payload in scored[:top_k]]

    def count(self) -> int:
        return len(self._points)

    def ping(self) -> bool:
        return True
