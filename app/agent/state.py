"""Shared state threaded through the LangGraph nodes."""

from __future__ import annotations

from typing import Any, TypedDict

from app.agent.verdict import Finding, VerdictDecision
from app.rag.store import SearchHit


class AgentState(TypedDict, total=False):
    """State passed between nodes.

    ``total=False`` because nodes populate it progressively; each node reads what
    its predecessors wrote and adds its own keys.
    """

    # Input
    query: str
    trace_id: str

    # parse_query
    # "greeting" when the input is only a hello, set without an LLM call.
    # Absent for everything else, which is the ordinary assessment path.
    intent: str
    in_scope: bool
    product_type: str
    proposal_summary: str
    features: list[str]
    concepts: list[str]

    # plan_retrieval
    sub_queries: list[str]
    retrieval_round: int
    broadened: bool

    # retrieve
    hits: list[SearchHit]
    top_score: float
    reranked: bool

    # assess
    findings: list[Finding]
    summary: str
    recommended_actions: list[str]
    # Information the proposal does not mention. Surfaced to the reviewer but
    # deliberately excluded from the verdict: a short proposal is silent on many
    # details, and letting each one force NEEDS_REVIEW makes every verdict
    # NEEDS_REVIEW.
    open_questions: list[str]

    # verify_citations
    rejected_citations: list[str]

    # decide_verdict
    decision: VerdictDecision

    # Bookkeeping
    llm_calls: list[dict[str, Any]]
    node_path: list[str]
    errors: list[str]
