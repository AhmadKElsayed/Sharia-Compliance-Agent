"""Embedding backends.

Embeddings go through OpenRouter's OpenAI-compatible ``/embeddings`` endpoint,
so the same key and SDK serve both chat and retrieval. Nothing is hosted
locally, which keeps the deployed container free of model weights.

``HashingEmbedder`` is the offline stand-in used by tests. It produces genuine
lexical similarity rather than noise, so retrieval logic can be exercised end to
end without a network call or an API key.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from openai import OpenAI

# OpenRouter accepts batched input; 64 keeps request bodies comfortably small
# while cutting the 30-chunk corpus down to a single round trip.
BATCH_SIZE = 64

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors produced."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for search."""
        ...


class OpenRouterEmbedder:
    """Embedder backed by OpenRouter's ``/embeddings`` endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dim: int,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required to embed text")
        self._model = model
        self._dim = dim
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> str:
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = self._client.embeddings.create(model=self._model, input=batch)
            # The API does not guarantee ordering, so sort by index rather than
            # trusting response order.
            ordered = sorted(response.data, key=lambda d: d.index)
            if len(ordered) != len(batch):
                raise RuntimeError(
                    f"embedding count mismatch: sent {len(batch)}, got {len(ordered)}"
                )
            vectors.extend(list(item.embedding) for item in ordered)

        actual = len(vectors[0])
        if actual != self._dim:
            raise RuntimeError(
                f"embedding model {self._model!r} returned {actual}-dim vectors but "
                f"EMBEDDING_DIM is {self._dim}. Update EMBEDDING_DIM and rebuild the "
                f"collection with `python scripts/ingest.py --recreate`."
            )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class HashingEmbedder:
    """Deterministic offline embedder for tests.

    A hashed bag-of-words projection: each token is hashed to a bucket and
    accumulated, then the vector is L2-normalised. Documents sharing vocabulary
    land near each other under cosine similarity, so tests can assert real
    retrieval behaviour without credentials.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # An empty or purely non-alphanumeric string still needs a valid
            # unit vector, or cosine similarity is undefined.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
