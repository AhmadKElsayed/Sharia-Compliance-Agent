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
from app.rag.chunking import chunk_corpus
from app.rag.embedder import HashingEmbedder
from app.rag.store import InMemoryStore

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"


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
    chunks = chunk_corpus(CORPUS_DIR)
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


def run(deps, query: str = "Can Mal offer a fixed-return savings account?"):
    return build_graph(deps).invoke(initial_state(query, "trace-1"))


# --- happy path ----------------------------------------------------------


def test_full_path_produces_non_compliant(store_and_embedder) -> None:
    store, _ = store_and_embedder
    real_id = next(iter(store._points.values()))[1]["chunk_id"]  # noqa: SLF001
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
    store, _ = store_and_embedder
    real_id = next(iter(store._points.values()))[1]["chunk_id"]  # noqa: SLF001
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
    store, _ = store_and_embedder
    real_id = next(iter(store._points.values()))[1]["chunk_id"]  # noqa: SLF001
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
    store, _ = store_and_embedder
    real_id = next(iter(store._points.values()))[1]["chunk_id"]  # noqa: SLF001
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
    store, _ = store_and_embedder
    real_id = next(iter(store._points.values()))[1]["chunk_id"]  # noqa: SLF001
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
