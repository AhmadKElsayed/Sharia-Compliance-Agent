"""Cross-encoder reranking over retrieved candidates.

Vector search uses a bi-encoder: the query and the clause are embedded
separately, so similarity is an approximation that never sees the two together.
A cross-encoder scores the pair jointly and is markedly better at judging
whether a clause actually answers the question.

That distinction matters here. Retrieval showed similarity scores that were
*anti-correlated* with relevance on one production query -- irrelevant clauses
at 0.56, the governing ones at 0.42 -- because generic legal phrasing matches
generic clause language everywhere. A cross-encoder is the standard remedy.

Reranking is a quality improvement, never a dependency: if the service is slow
or unavailable the fused vector ordering is returned unchanged, because a
degraded ranking is much better than a failed assessment.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import httpx

log = logging.getLogger("sharia.rerank")


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidates by relevance to the query."""

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Return ``(original_index, relevance_score)`` best first."""
        ...


class OpenRouterReranker:
    """Reranker backed by OpenRouter's ``/rerank`` endpoint.

    The endpoint is not listed in ``/models`` but is live and proxies to Voyage,
    so it needs no additional credential -- the same OpenRouter key covers chat,
    embeddings, and reranking.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyageai/rerank-2.5",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required to rerank")
        self._model = model
        self._url = f"{base_url.rstrip('/')}/rerank"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        try:
            response = httpx.post(
                self._url,
                headers=self._headers,
                json={
                    "model": self._model,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as exc:  # noqa: BLE001
            # Never fail an assessment because reranking was unavailable.
            log.warning(
                "rerank.failed",
                extra={"detail": f"{type(exc).__name__}: {exc}", "model": self._model},
            )
            return [(i, 0.0) for i in range(min(top_n, len(documents)))]

        ordered: list[tuple[int, float]] = []
        for item in results:
            index = item.get("index")
            if isinstance(index, int) and 0 <= index < len(documents):
                ordered.append((index, float(item.get("relevance_score", 0.0))))

        if not ordered:
            log.warning("rerank.empty", extra={"model": self._model})
            return [(i, 0.0) for i in range(min(top_n, len(documents)))]

        return ordered[:top_n]
