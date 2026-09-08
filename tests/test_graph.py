"""End-to-end graph tests with a stubbed LLM and in-memory store.

No network. These assert the control flow the design depends on: node order,
the broadening retry edge, the out-of-scope short circuit, and that a
hallucinated citation cannot produce a NON_COMPLIANT verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.graph import build_graph, initial_state
from app.agent.llm import LLMClient
from app.agent.nodes import AgentDeps
from app.agent.verdict import Verdict
from app.rag.chunking import chunk_documents
from app.rag.pdf_loader import load_pdf_corpus
from app.rag.embedder import HashingEmbedder
from app.rag.store import InMemoryStore

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus" / "demo"


class ScriptedLLM(LLMClient):
    """LLM client returning canned JSON payloads in order."""

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, max_tokens: int = 2048):  # type: ignore[override]
        self.calls.append((system, user))
        payload = self.payloads.pop(0) if self.payloads else {}
        response = SimpleNamespace(
            content=json.dumps(payload),
            model="stub",
            latency_ms=1,
            repaired=False,
            attempts=["stub"],
            to_log=lambda include_content: {"model": "stub", "latency_ms": 1},
        )
        return payload, response


@pytest.fixture(scope="module")
def store_and_embedder() -> tuple[InMemoryStore, HashingEmbedder]:
    embedder = HashingEmbedder(dim=256)
    store = InMemoryStore()
    store.ensure_collection(embedder.dim)
    chunks = chunk_documents(load_pdf_corpus(CORPUS_DIR))
    store.upsert(chunks, embedder.embed_documents([c.embedding_text() for c in chunks]))
    return store, embedder


def make_deps(payloads: list[dict], store_and_embedder, threshold: float = 0.05):
    store, embedder = store_and_embedder
    llm = ScriptedLLM(payloads)
    return AgentDeps(llm, embedder, store, top_k=5, score_threshold=threshold), llm


def parse_payload(**kw) -> dict:
    return {
        "in_scope": kw.get("in_scope", True),
        "product_type": kw.get("product_type", "savings account"),
        "summary": kw.get("summary", "A deposit paying a fixed return."),
        "features": kw.get("features", ["guarantees a fixed 3% annual return"]),
        "concepts": kw.get("concepts", ["riba"]),
    }


def assess_payload(chunk_id: str, severity: str = "PROHIBITED") -> dict:
    return {
        "findings": [
            {
                "issue": "riba",
                "severity": severity,
                "explanation": "A guaranteed return on a deposit is riba.",
                "citations": [{"chunk_id": chunk_id, "quote": "guaranteed return"}],
            }
        ],
        "summary": "The proposal guarantees a fixed return.",
        "recommended_actions": ["Restructure as a Mudarabah account."],
    }


QUERY = "Can Mal offer a fixed-return savings account?"


def run(deps, query: str = QUERY):
    return build_graph(deps).invoke(initial_state(query, "trace-1"))


def retrieved_chunk_id(store_and_embedder, query: str = QUERY) -> str:
    """A chunk id the retriever will actually return for this query.

    Picking an arbitrary stored chunk instead would be a latent bug: outside
    top-k it gets stripped by the citation gate and the verdict silently
    downgrades, which is exactly what the safety tests are meant to detect.
    """
    store, embedder = store_and_embedder
    hits = store.search(embedder.embed_query(query), top_k=5)
    assert hits, "fixture corpus produced no hits"
    return hits[0].chunk_id


# --- happy path ----------------------------------------------------------


def test_full_path_produces_non_compliant(store_and_embedder) -> None:
    real_id = retrieved_chunk_id(store_and_embedder)
    deps, llm = make_deps(
        [parse_payload(), assess_payload(real_id)], store_and_embedder
    )
    state = run(deps)

    assert state["decision"].verdict is Verdict.NON_COMPLIANT
    assert state["decision"].rule == "grounded_prohibition"
    assert state["node_path"] == [
        "parse_query",
        "plan_retrieval",
        "retrieve",
        "assess",
        "verify_citations",
        "decide_verdict",
    ]
    assert len(llm.calls) == 2, "exactly two LLM calls per assessment"


def test_permissible_findings_yield_compliant(store_and_embedder) -> None:
    real_id = retrieved_chunk_id(store_and_embedder)
    deps, _ = make_deps(
        [parse_payload(), assess_payload(real_id, "PERMISSIBLE")], store_and_embedder
    )
    assert run(deps)["decision"].verdict is Verdict.COMPLIANT


# --- control flow --------------------------------------------------------


def test_out_of_scope_skips_assessment(store_and_embedder) -> None:
    deps, llm = make_deps([parse_payload(in_scope=False)], store_and_embedder)
    state = run(deps, "What is the weather in Cairo?")

    assert state["decision"].verdict is Verdict.NEEDS_REVIEW
    assert state["decision"].rule == "out_of_scope"
    assert "assess" not in state["node_path"]
    assert len(llm.calls) == 1, "no assessment call for an out-of-scope query"


def test_weak_retrieval_triggers_one_broadened_retry(store_and_embedder) -> None:
    """An unreachable threshold must cause exactly one retry, then proceed."""
    real_id = retrieved_chunk_id(store_and_embedder)
    deps, _ = make_deps(
        [parse_payload(), assess_payload(real_id)], store_and_embedder, threshold=1.1
    )
    state = run(deps)

    assert state["node_path"].count("plan_retrieval") == 2
    assert state["node_path"].count("retrieve") == 2
    assert state["broadened"] is True
    assert state["retrieval_round"] == 2
    # Coverage still fails after broadening, so the verdict must not be confident.
    assert state["decision"].rule == "insufficient_corpus_coverage"


def test_broadened_queries_differ_from_first_round(store_and_embedder) -> None:
    real_id = retrieved_chunk_id(store_and_embedder)
    deps, _ = make_deps(
        [parse_payload(), assess_payload(real_id)], store_and_embedder, threshold=1.1
    )
    state = run(deps)
    assert any("prohibited features" in q for q in state["sub_queries"])


def test_sub_queries_are_derived_from_features(store_and_embedder) -> None:
    deps, _ = make_deps(
        [
            parse_payload(features=["guarantees a fixed return"], concepts=["riba"]),
            {"findings": [], "summary": "", "recommended_actions": []},
        ],
        store_and_embedder,
    )
    state = run(deps)
    joined = " ".join(state["sub_queries"])
    assert "guarantees a fixed return" in joined
    assert "riba" in joined
    assert len(state["sub_queries"]) > 1, "must not rely on a single query"


# --- the safety property -------------------------------------------------


def test_hallucinated_citation_cannot_convict(store_and_embedder) -> None:
    deps, _ = make_deps(
        [parse_payload(), assess_payload("SFS-999#9.9")], store_and_embedder
    )
    state = run(deps)

    assert state["rejected_citations"] == ["SFS-999#9.9"]
    assert state["decision"].verdict is Verdict.NEEDS_REVIEW
    assert state["decision"].rule == "prohibited_findings_lost_all_citations"


def test_empty_findings_do_not_yield_compliant(store_and_embedder) -> None:
    deps, _ = make_deps(
        [parse_payload(), {"findings": [], "summary": "", "recommended_actions": []}],
        store_and_embedder,
    )
    assert run(deps)["decision"].verdict is Verdict.NEEDS_REVIEW


def test_unknown_severity_degrades_to_unresolved(store_and_embedder) -> None:
    real_id = retrieved_chunk_id(store_and_embedder)
    deps, _ = make_deps(
        [parse_payload(), assess_payload(real_id, "SOMEWHAT_DUBIOUS")],
        store_and_embedder,
    )
    state = run(deps)
    assert state["decision"].verdict is Verdict.NEEDS_REVIEW
    assert state["decision"].rule == "unresolved_findings"


def test_retrieval_hits_are_deduplicated(store_and_embedder) -> None:
    deps, _ = make_deps(
        [parse_payload(), {"findings": [], "summary": "", "recommended_actions": []}],
        store_and_embedder,
    )
    ids = [h.chunk_id for h in run(deps)["hits"]]
    assert len(ids) == len(set(ids)), "a chunk matched by several sub-queries appears once"


# --- retrieval planning and fusion ---------------------------------------


def test_sub_queries_are_grounded_in_the_product() -> None:
    """Grounding is what made retrieval work: on a measured failing case
    "<product> <feature>" retrieved 4/5 relevant clauses where the bare feature
    retrieved 0/5."""
    from app.agent.nodes import _ground

    assert _ground("guarantees a fixed 4% return", "savings account") == (
        "savings account guarantees a fixed 4% return"
    )
    assert _ground("riba", "savings account", template="{text} in {product}") == (
        "riba in savings account"
    )


def test_grounding_does_not_repeat_the_product() -> None:
    """Blind concatenation produced "Murabaha in Murabaha" -- a wasted call."""
    from app.agent.nodes import _ground

    assert _ground("Murabaha", "Murabaha", template="{text} in {product}") == "Murabaha"
    assert _ground("savings account for depositors", "savings account") == (
        "savings account for depositors"
    )
    assert _ground("anything", "") == "anything"


def test_degenerate_sub_queries_are_dropped(store_and_embedder) -> None:
    """A two-word sub-query matches generic clause language everywhere."""
    deps, _ = make_deps(
        [
            parse_payload(product_type="", features=["ok"], concepts=["x"]),
            {"findings": [], "summary": "", "recommended_actions": []},
        ],
        store_and_embedder,
    )
    state = run(deps)
    assert all(len(q.split()) >= 3 for q in state["sub_queries"])
    assert state["sub_queries"], "the raw query must always survive filtering"


def test_retrieval_uses_rank_fusion_not_raw_score(store_and_embedder) -> None:
    """Raw scores are not comparable across sub-queries.

    Measured on a real failure: a vague sub-query scored 0.56 on irrelevant
    clauses while the user's own question scored 0.42 on the governing ones, so
    max-score merging let the wrong results displace the right ones. Fusing on
    rank makes a sub-query's ordering count and its absolute scores irrelevant.
    """
    from app.agent.nodes import AgentDeps, retrieve
    from app.rag.store import SearchHit

    def hit(cid: str, score: float) -> SearchHit:
        return SearchHit(
            chunk_id=cid, score=score, doc_id=cid, title="t", heading="h",
            section_label="1", citation=cid, text="x",
        )

    class SplitStore:
        """First sub-query returns the right answer at a low score; the second
        returns junk at high scores."""

        def __init__(self) -> None:
            self.calls = 0

        def search(self, vector, top_k, score_threshold=None):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                return [hit("RIGHT", 0.42), hit("also", 0.41)]
            return [hit("JUNK-A", 0.56), hit("JUNK-B", 0.55), hit("RIGHT", 0.30)]

    class OneVec:
        dim = 4

        def embed_query(self, text: str):  # noqa: ANN201
            return [1.0, 0.0, 0.0, 0.0]

        def embed_documents(self, texts):  # noqa: ANN001, ANN201
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    deps = AgentDeps(llm=None, embedder=OneVec(), store=SplitStore(), top_k=5)
    state = {"sub_queries": ["query one here", "query two here"], "node_path": []}
    retrieve(state, deps)

    ids = [h.chunk_id for h in state["hits"]]
    # RIGHT is rank 1 in one query and rank 3 in the other, so fusion promotes
    # it above JUNK-A, which ranks first only once. Max-score merging would put
    # JUNK-A first on its 0.56.
    assert ids[0] == "RIGHT", f"rank fusion should promote RIGHT, got {ids}"
    # The raw similarity is preserved for the coverage threshold.
    assert state["hits"][0].score == 0.42
