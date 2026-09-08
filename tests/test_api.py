"""API contract tests.

The graph is stubbed, so these run offline and assert the HTTP surface: status
codes, response shape, the trace header, the error envelope, and that /health
reports readiness honestly rather than optimistically.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agent.llm import LLMError
from app.agent.verdict import Citation, Finding, Severity, VerdictDecision, Verdict
from app.observability.trace import TRACE_HEADER
from app.observability.traces import TRACES
from app.rag.store import SearchHit


def make_state(**over: Any) -> dict[str, Any]:
    hit = SearchHit(
        chunk_id="SS-08/3/1/1",
        score=0.679,
        doc_id="SS-08",
        title="Murabahah",
        heading="3. Acquisition",
        section_label="3/1/1",
        citation="Shari'ah Standard No. 8 (Murabahah), clause 3/1/1",
        text="The Institution must acquire ownership...",
    )
    state: dict[str, Any] = {
        "query": "Must the bank own the asset first?",
        "decision": VerdictDecision(
            Verdict.NON_COMPLIANT, "grounded_prohibition", "Prohibited under: ownership.", 0.9
        ),
        "summary": "Ownership is required before onward sale.",
        "findings": [
            Finding(
                issue="ownership",
                severity=Severity.PROHIBITED,
                explanation="No ownership is taken.",
                citations=[
                    Citation(
                        chunk_id="SS-08/3/1/1",
                        doc_id="SS-08",
                        section="3/1/1",
                        title="Murabahah",
                        quote="must acquire ownership",
                    )
                ],
            )
        ],
        "open_questions": ["What is the interval between purchase and resale?"],
        "recommended_actions": ["Take real ownership before the onward sale."],
        "hits": [hit],
        "rejected_citations": [],
        "llm_calls": [{"node": "assess", "model": "stub-model"}],
        "errors": [],
    }
    state.update(over)
    return state


class StubGraph:
    def __init__(self, state: dict[str, Any] | None = None, raises: Exception | None = None):
        self.state = state if state is not None else make_state()
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def invoke(self, initial: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(initial)
        if self.raises:
            raise self.raises
        return {**self.state, "query": initial["query"]}


class StubStore:
    def __init__(self, reachable: bool = True, count: int = 1973):
        self._reachable, self._count = reachable, count

    def ping(self) -> bool:
        return self._reachable

    def count(self) -> int:
        if not self._reachable:
            raise RuntimeError("unreachable")
        return self._count


class StubDeps:
    def __init__(self, store: StubStore):
        self.store = store


@pytest.fixture()
def client(monkeypatch):
    """A client with lifespan bypassed and stubs installed."""
    from app.config import Settings

    settings = Settings(
        openrouter_api_key="k", qdrant_url="https://x", qdrant_api_key="q",
        _env_file=None,
    )
    monkeypatch.setitem(main._state, "settings", settings)
    monkeypatch.setitem(main._state, "missing", [])
    monkeypatch.setitem(main._state, "deps", StubDeps(StubStore()))
    monkeypatch.setitem(main._state, "graph", StubGraph())

    # TestClient normally runs lifespan; these tests install state directly.
    with TestClient(main.app) as c:
        monkeypatch.setitem(main._state, "settings", settings)
        monkeypatch.setitem(main._state, "missing", [])
        monkeypatch.setitem(main._state, "deps", StubDeps(StubStore()))
        monkeypatch.setitem(main._state, "graph", StubGraph())
        yield c


# --- POST /assess --------------------------------------------------------


def test_assess_returns_structured_verdict(client) -> None:
    r = client.post("/assess", json={"query": "Must the bank own the asset first?"})
    assert r.status_code == 200
    body = r.json()

    assert body["verdict"] == "NON_COMPLIANT"
    assert body["rule"] == "grounded_prohibition"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["findings"][0]["severity"] == "PROHIBITED"
    assert body["findings"][0]["citations"][0]["chunk_id"] == "SS-08/3/1/1"
    assert body["retrieved_chunks"][0]["doc_id"] == "SS-08"
    assert body["latency_ms"] >= 0
    assert body["model"] == "stub-model"


def test_assess_surfaces_open_questions_separately_from_findings(client) -> None:
    body = client.post("/assess", json={"query": "a valid query"}).json()
    assert body["open_questions"], "open questions must reach the reviewer"
    # They must not appear as findings, which would change the verdict.
    assert all("interval" not in f["issue"] for f in body["findings"])


def test_assess_carries_a_disclaimer(client) -> None:
    body = client.post("/assess", json={"query": "a valid query"}).json()
    assert "not a Sharia ruling" in body["disclaimer"]


def test_assess_returns_trace_header_matching_body(client) -> None:
    r = client.post("/assess", json={"query": "a valid query"})
    assert r.headers[TRACE_HEADER]
    assert r.json()["trace_id"] == r.headers[TRACE_HEADER]


def test_inbound_trace_id_is_adopted(client) -> None:
    r = client.post(
        "/assess", json={"query": "a valid query"}, headers={TRACE_HEADER: "abc-123-xyz"}
    )
    assert r.headers[TRACE_HEADER] == "abc-123-xyz"
    assert r.json()["trace_id"] == "abc-123-xyz"


def test_malformed_inbound_trace_id_is_replaced(client) -> None:
    """An unvalidated header would let a caller inject into logs and headers."""
    r = client.post(
        "/assess", json={"query": "a valid query"}, headers={TRACE_HEADER: "bad id\nwith newline"}
    )
    assert r.headers[TRACE_HEADER] != "bad id\nwith newline"
    assert len(r.headers[TRACE_HEADER]) == 32


@pytest.mark.parametrize("bad", [{}, {"query": ""}, {"query": "ab"}, {"query": "x" * 5000}])
def test_invalid_requests_are_rejected(client, bad) -> None:
    assert client.post("/assess", json=bad).status_code == 422


def test_llm_failure_returns_502_with_error_envelope(client, monkeypatch) -> None:
    monkeypatch.setitem(main._state, "graph", StubGraph(raises=LLMError("all models failed")))
    r = client.post("/assess", json={"query": "a valid query"})
    assert r.status_code == 502
    body = r.json()
    assert body["error"]["code"] == "upstream_llm_error"
    assert body["trace_id"]


def test_unexpected_failure_returns_500_with_error_envelope(client, monkeypatch) -> None:
    monkeypatch.setitem(main._state, "graph", StubGraph(raises=RuntimeError("boom")))
    r = client.post("/assess", json={"query": "a valid query"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "assessment_failed"
    # The internal message must not leak to the caller.
    assert "boom" not in r.text


def test_assess_returns_503_when_unconfigured(client, monkeypatch) -> None:
    monkeypatch.setitem(main._state, "graph", None)
    r = client.post("/assess", json={"query": "a valid query"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "not_ready"


# --- GET /health ---------------------------------------------------------


def test_health_ok_when_everything_reachable(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["corpus_chunks"] == 1973
    assert all(d["ok"] for d in body["dependencies"])


def test_health_reports_503_when_store_unreachable(client, monkeypatch) -> None:
    """A green check while the vector store is down is worse than no check."""
    monkeypatch.setitem(main._state, "deps", StubDeps(StubStore(reachable=False)))
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert any(d["name"] == "qdrant" and not d["ok"] for d in r.json()["dependencies"])


def test_health_reports_503_when_collection_empty(client, monkeypatch) -> None:
    monkeypatch.setitem(main._state, "deps", StubDeps(StubStore(count=0)))
    r = client.get("/health")
    assert r.status_code == 503


def test_health_reports_missing_config(client, monkeypatch) -> None:
    monkeypatch.setitem(main._state, "deps", None)
    monkeypatch.setitem(main._state, "missing", ["OPENROUTER_API_KEY"])
    r = client.get("/health")
    assert r.status_code == 503
    assert "OPENROUTER_API_KEY" in r.json()["missing_config"]


# --- GET /corpus ---------------------------------------------------------


def test_corpus_reports_index_without_serving_text(client) -> None:
    body = client.get("/corpus").json()
    assert body["chunks"] == 1973
    assert "clause-level" in body["chunking"]
    # Licensed text must not be served in bulk from a public endpoint.
    assert "text" not in body


# --- GET /traces ---------------------------------------------------------


def test_trace_is_recorded_and_replayable(client) -> None:
    r = client.post("/assess", json={"query": "a valid query"})
    trace_id = r.headers[TRACE_HEADER]

    t = client.get(f"/traces/{trace_id}")
    assert t.status_code == 200
    body = t.json()
    assert body["trace_id"] == trace_id
    assert body["event_count"] >= 2
    events = {e["event"] for e in body["events"]}
    assert "request.received" in events
    assert "request.completed" in events


def test_completed_event_records_the_verdict(client) -> None:
    r = client.post("/assess", json={"query": "a valid query"})
    body = client.get(f"/traces/{r.headers[TRACE_HEADER]}").json()
    completed = next(e for e in body["events"] if e["event"] == "request.completed")
    assert completed["data"]["verdict"] == "NON_COMPLIANT"
    assert completed["data"]["rule"] == "grounded_prohibition"


def test_unknown_trace_returns_404_envelope(client) -> None:
    r = client.get("/traces/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "trace_not_found"


def test_trace_store_is_bounded() -> None:
    from app.observability.traces import TraceStore

    store = TraceStore(max_traces=3, max_events=2)
    for i in range(5):
        store.add(f"t{i}", {"event": "a"})
        store.add(f"t{i}", {"event": "b"})
        store.add(f"t{i}", {"event": "dropped"})

    assert len(store) == 3, "oldest traces evicted"
    assert store.get("t0") is None
    assert len(store.get("t4")) == 2, "events capped per trace"


# --- misc ----------------------------------------------------------------


def test_root_redirects_to_docs(client) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/docs"


def test_openapi_schema_is_served(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/assess" in schema["paths"]
    assert "/health" in schema["paths"]
