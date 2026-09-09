"""Request and response models for the REST API.

These are the public contract. Internal types (``Finding``, ``SearchHit``,
``VerdictDecision``) are mapped into them explicitly rather than serialised
directly, so that refactoring the agent cannot silently change the API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.agent.verdict import Verdict


class AssessRequest(BaseModel):
    query: str = Field(
        min_length=3,
        max_length=4000,
        description="Plain-English description of the proposed product or transaction.",
        examples=[
            "How should profit be shared between the bank and depositors in a "
            "Mudarabah account?"
        ],
    )


class CitationOut(BaseModel):
    chunk_id: str = Field(description="Identifier of the retrieved clause.")
    doc_id: str
    section: str = Field(description="Clause number within the standard, e.g. 2/1/2.")
    title: str
    quote: str = Field(default="", description="Verbatim span supporting the finding.")


class FindingOut(BaseModel):
    issue: str
    severity: Literal["PROHIBITED", "CONDITIONAL", "UNRESOLVED", "PERMISSIBLE"]
    explanation: str
    citations: list[CitationOut] = Field(
        default_factory=list,
        description="Only citations verified against retrieved text; fabricated "
        "references are stripped before this point.",
    )


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    score: float
    doc_id: str
    citation: str


class AssessResponse(BaseModel):
    trace_id: str
    query: str
    verdict: Literal["COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW", "IRRELEVANT"]
    """The three assessment outcomes, plus IRRELEVANT when the message was not a
    compliance question at all and nothing was assessed."""
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Support for the verdict. A coarse signal, not a probability.",
    )
    rule: str = Field(description="The deterministic rule that produced the verdict.")
    rationale: str
    summary: str = ""
    findings: list[FindingOut] = Field(default_factory=list)
    open_questions: list[str] = Field(
        default_factory=list,
        description="Information the proposal does not state. Surfaced to the "
        "reviewer but excluded from the verdict.",
    )
    recommended_actions: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkOut] = Field(default_factory=list)
    rejected_citations: list[str] = Field(
        default_factory=list,
        description="Citations dropped because they did not match retrieved text.",
    )
    model: str = ""
    latency_ms: int = 0
    disclaimer: str = ""


class DependencyHealth(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    corpus_chunks: int = 0
    collection: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    dependencies: list[DependencyHealth] = Field(default_factory=list)
    missing_config: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    ts: float
    level: str
    event: str
    logger: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(BaseModel):
    trace_id: str
    event_count: int
    events: list[TraceEvent]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Errors use the same envelope shape as success, never a bare string."""

    trace_id: str
    error: ErrorDetail


DISCLAIMER = (
    "Decision support only. This assessment is generated from retrieved "
    "standards text and must be reviewed by a qualified Sharia reviewer. It is "
    "not a Sharia ruling and does not substitute for Sharia Supervisory Board "
    "approval."
)

# An IRRELEVANT response assessed nothing, so the standard disclaimer would be a
# false statement about it -- there is no retrieved standards text behind a
# greeting, and telling a reviewer to review one wastes their attention.
NO_ASSESSMENT_NOTE = (
    "No compliance assessment was performed for this message."
)


def to_assess_response(
    state: AgentState, trace_id: str, latency_ms: int
) -> AssessResponse:
    """Map terminal agent state onto the public response contract."""
    decision = state["decision"]
    llm_calls = state.get("llm_calls") or []

    return AssessResponse(
        trace_id=trace_id,
        query=state["query"],
        verdict=Verdict(decision.verdict).value,
        confidence=decision.confidence,
        rule=decision.rule,
        rationale=decision.rationale,
        summary=state.get("summary", "") or "",
        findings=[
            FindingOut(
                issue=f.issue,
                severity=f.severity.value,
                explanation=f.explanation,
                citations=[
                    CitationOut(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        section=c.section,
                        title=c.title,
                        quote=c.quote,
                    )
                    for c in f.citations
                ],
            )
            for f in state.get("findings", []) or []
        ],
        open_questions=state.get("open_questions") or [],
        recommended_actions=state.get("recommended_actions") or [],
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_id=h.chunk_id,
                score=round(h.score, 4),
                doc_id=h.doc_id,
                citation=h.citation,
            )
            for h in state.get("hits", []) or []
        ],
        rejected_citations=state.get("rejected_citations") or [],
        model=(llm_calls[-1].get("model", "") if llm_calls else ""),
        latency_ms=latency_ms,
        disclaimer=(
            NO_ASSESSMENT_NOTE
            if decision.verdict == Verdict.IRRELEVANT
            else DISCLAIMER
        ),
    )
