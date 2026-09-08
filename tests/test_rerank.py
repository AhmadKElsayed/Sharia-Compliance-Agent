"""Tests for cross-encoder reranking.

No network. The central property: reranking improves ranking but is never a
dependency -- if it fails, retrieval must still return usable results.
"""

from __future__ import annotations

import httpx
import pytest

from app.agent.nodes import AgentDeps, retrieve
from app.rag.rerank import OpenRouterReranker, Reranker
from app.rag.store import SearchHit


def hit(cid: str, score: float, doc: str = "SS-01") -> SearchHit:
    return SearchHit(
        chunk_id=cid, score=score, doc_id=doc, title="t", heading="h",
        section_label="1", citation=cid, text=f"text of {cid}",
        breadcrumb=f"Standard > {cid}",
    )


class StubStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits

    def search(self, vector, top_k, score_threshold=None):  # noqa: ANN001, ANN201
        return self._hits[:top_k]


class OneVec:
    dim = 4

    def embed_query(self, text: str):  # noqa: ANN201
        return [1.0, 0.0, 0.0, 0.0]

    def embed_documents(self, texts):  # noqa: ANN001, ANN201
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class ReverseReranker:
    """Reverses the candidate order, so its effect is unmistakable."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def rerank(self, query, documents, top_n):  # noqa: ANN001, ANN201
        self.seen = list(documents)
        order = list(range(len(documents)))[::-1]
        return [(i, 1.0 - n * 0.01) for n, i in enumerate(order)][:top_n]


class BrokenReranker:
    def rerank(self, query, documents, top_n):  # noqa: ANN001, ANN201
        raise RuntimeError("rerank service down")


def _state(query: str = "a compliance question here") -> dict:
    return {"query": query, "sub_queries": [query], "node_path": [], "errors": []}


# --- protocol ------------------------------------------------------------


def test_openrouter_reranker_satisfies_the_protocol() -> None:
    assert isinstance(
        OpenRouterReranker(api_key="k"), Reranker
    )


def test_reranker_requires_a_key() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterReranker(api_key="")


# --- effect on retrieval -------------------------------------------------


def test_reranker_reorders_results() -> None:
    hits = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]
    reranker = ReverseReranker()
    deps = AgentDeps(
        llm=None, embedder=OneVec(), store=StubStore(hits), top_k=5,
        reranker=reranker,
    )
    state = _state()
    retrieve(state, deps)

    assert [h.chunk_id for h in state["hits"]] == ["c", "b", "a"]
    assert state["reranked"] is True


def test_reranker_sees_the_breadcrumb() -> None:
    """AAOIFI clauses are elliptical; without lineage a cross-encoder cannot
    judge relevance."""
    reranker = ReverseReranker()
    deps = AgentDeps(
        llm=None, embedder=OneVec(),
        store=StubStore([hit("a", 0.9), hit("b", 0.8)]),
        top_k=5, reranker=reranker,
    )
    retrieve(_state(), deps)
    assert "Standard > a" in reranker.seen[0]
    assert "text of a" in reranker.seen[0]


def test_retrieval_survives_a_failing_reranker() -> None:
    """A degraded ranking is far better than a failed assessment."""
    hits = [hit("a", 0.9), hit("b", 0.8)]
    deps = AgentDeps(
        llm=None, embedder=OneVec(), store=StubStore(hits), top_k=5,
        reranker=BrokenReranker(),
    )
    state = _state()
    retrieve(state, deps)

    assert [h.chunk_id for h in state["hits"]] == ["a", "b"], "fused order kept"
    assert "reranked" not in state
    assert any("rerank failed" in e for e in state["errors"])


def test_http_failure_falls_back_to_original_order(monkeypatch) -> None:
    """The real client must swallow transport errors, not propagate them."""

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(httpx, "post", boom)
    reranker = OpenRouterReranker(api_key="k")
    order = reranker.rerank("q", ["one", "two", "three"], top_n=2)
    assert order == [(0, 0.0), (1, 0.0)], "preserves the incoming order"


def test_empty_document_list_is_handled() -> None:
    assert OpenRouterReranker(api_key="k").rerank("q", [], top_n=5) == []


def test_retrieval_without_a_reranker_is_unchanged() -> None:
    hits = [hit("a", 0.9), hit("b", 0.8)]
    deps = AgentDeps(llm=None, embedder=OneVec(), store=StubStore(hits), top_k=5)
    state = _state()
    retrieve(state, deps)
    assert [h.chunk_id for h in state["hits"]] == ["a", "b"]
    assert "reranked" not in state


def test_out_of_range_indices_are_ignored() -> None:
    class BadIndex:
        def rerank(self, query, documents, top_n):  # noqa: ANN001, ANN201
            return [(99, 1.0), (0, 0.9)]

    deps = AgentDeps(
        llm=None, embedder=OneVec(),
        store=StubStore([hit("a", 0.9), hit("b", 0.8)]),
        top_k=5, reranker=BadIndex(),
    )
    state = _state()
    retrieve(state, deps)
    assert [h.chunk_id for h in state["hits"]] == ["a"]
