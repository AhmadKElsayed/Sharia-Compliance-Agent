"""The agent's node functions.

Each node is an ordinary Python function taking and returning ``AgentState``.
Nothing is hidden behind a framework abstraction: the retrieval planning, the
citation gate, and the verdict rules are all readable here and in
``verdict.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.llm import LLMClient, LLMError
from app.agent.prompts import (
    ASSESS_SYSTEM,
    PARSE_SYSTEM,
    render_assess_user,
    render_parse_user,
)
from app.agent.state import AgentState
from app.agent.verdict import (
    Citation,
    Finding,
    Severity,
    decide,
    verify_citations,
)
from app.rag.embedder import Embedder
from app.rag.rerank import Reranker
from app.rag.store import SearchHit, VectorStore

log = logging.getLogger("sharia.agent")

MAX_RETRIEVAL_ROUNDS = 2
MAX_SUB_QUERIES = 4
MAX_EXCERPTS = 8

# Sub-queries shorter than this match generic clause language rather than a
# topic, and their inflated scores crowd out correct results.
MIN_SUB_QUERY_WORDS = 3

# Reciprocal-rank-fusion constant. 60 is the value from the original RRF paper
# and is deliberately large so that ranks 1-10 stay close together, letting
# agreement across sub-queries matter more than a single top placement.
RRF_K = 60


class AgentDeps:
    """Collaborators the nodes need.

    Passed explicitly rather than resolved globally so tests can substitute a
    stub LLM and an in-memory store.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.35,
        max_tokens: int = 12000,
        reranker: "Reranker | None" = None,
        rerank_candidates: int = 24,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold
        # Shared budget for both calls. It must cover the reasoning pass as well
        # as the answer; too small and the model returns empty content.
        self.max_tokens = max_tokens
        # Optional; retrieval degrades to fused vector ordering without it.
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates


def _record_call(state: AgentState, node: str, response: Any) -> None:
    calls = state.setdefault("llm_calls", [])
    calls.append({"node": node, **response.to_log(include_content=False)})


def _mark(state: AgentState, node: str) -> None:
    state.setdefault("node_path", []).append(node)


def _error(state: AgentState, message: str) -> None:
    state.setdefault("errors", []).append(message)
    log.warning("agent.node.error", extra={"detail": message})


# --- 1. parse_query ------------------------------------------------------

# Intents the parse step can return. This node is the router: it classifies the
# message and the conditional edge below acts on the classification.
#
# The LLM does the classifying rather than a keyword list. A word list was tried
# first and rejected: it cannot cover greetings in other languages, transliterated
# salutations, typos or informal phrasing without growing indefinitely, and every
# word added widens the chance of swallowing a real query. The model already
# reads this message for structure, so classification is free -- no extra call.
INTENT_GREETING = "GREETING"
INTENT_ASSESSMENT = "ASSESSMENT"
INTENT_OTHER = "OTHER"
VALID_INTENTS = frozenset({INTENT_GREETING, INTENT_ASSESSMENT, INTENT_OTHER})


def _intent(parsed: dict[str, Any]) -> str:
    """Read the classification, defaulting to the safe branch.

    An unrecognised value routes to ASSESSMENT rather than GREETING. Treating an
    unknown intent as a greeting would answer a real compliance question with a
    welcome message; treating it as an assessment merely costs a retrieval.
    """
    value = str(parsed.get("intent", "") or "").strip().upper()
    return value if value in VALID_INTENTS else INTENT_ASSESSMENT


def parse_query(state: AgentState, deps: AgentDeps) -> AgentState:
    """Classify the message and extract product structure.

    Doubles as the router: ``state["intent"]`` is what ``should_retrieve``
    branches on.
    """
    _mark(state, "parse_query")

    try:
        parsed, response = deps.llm.complete_json(
            PARSE_SYSTEM, render_parse_user(state["query"]), max_tokens=deps.max_tokens
        )
        _record_call(state, "parse_query", response)
    except LLMError as exc:
        # Without structure we can still retrieve on the raw query, so degrade
        # rather than abort; the verdict rules will handle thin evidence. The
        # intent defaults to ASSESSMENT for the same reason: a failed classifier
        # must not silently turn a compliance question into a welcome message.
        _error(state, f"parse_query failed: {exc}")
        state["intent"] = INTENT_ASSESSMENT
        state["in_scope"] = True
        state["proposal_summary"] = state["query"]
        state["features"] = []
        state["concepts"] = []
        return state

    intent = _intent(parsed)
    state["intent"] = intent
    # A greeting is never in scope, whatever the model put in the flag.
    state["in_scope"] = intent == INTENT_ASSESSMENT and bool(
        parsed.get("in_scope", True)
    )
    state["product_type"] = str(parsed.get("product_type", "") or "")
    state["proposal_summary"] = str(parsed.get("summary", "") or state["query"])
    state["features"] = [str(f) for f in parsed.get("features", []) if str(f).strip()]
    state["concepts"] = [str(c) for c in parsed.get("concepts", []) if str(c).strip()]
    return state


# --- 2. plan_retrieval ---------------------------------------------------


def _ground(text: str, product: str, template: str = "{product} {text}") -> str:
    """Attach the product type to a feature or concept, without repeating it.

    Grounding is what made these sub-queries work: measured on a failing case,
    "<product> <feature>" retrieved 4/5 relevant clauses where the bare feature
    retrieved 0/5.

    The product name is often already inside the extracted text, and blind
    concatenation produced "Murabaha in Murabaha" and "savings account savings
    account for depositors" -- wasted embedding calls retrieving the same thing
    twice.
    """
    text = " ".join(text.split())
    if not product:
        return text
    if set(product.lower().split()).issubset(set(text.lower().split())):
        return text
    return template.format(product=product, text=text)


def plan_retrieval(state: AgentState, deps: AgentDeps) -> AgentState:
    """Derive targeted sub-queries from the parsed structure.

    Pure Python, no LLM call. One embedding of the raw question tends to match
    whichever document shares its surface vocabulary; querying separately per
    feature and per concept surfaces clauses a single query would miss.
    """
    _mark(state, "plan_retrieval")

    round_no = state.get("retrieval_round", 0) + 1
    state["retrieval_round"] = round_no
    query = state["query"]

    product = (state.get("product_type") or "").strip()

    if round_no == 1:
        # Sub-queries are grounded in the product rather than dressed in generic
        # legal phrasing. Measured on a failing case: "<product> <feature>"
        # returned 4/5 relevant clauses and "<concept> in <product>" 3/5, while
        # the same feature alone, the bare concept, and the previous templates
        # ("... permissibility ruling", "... definition and prohibition
        # criteria") each returned 0/5.
        #
        # The templates were worse than useless. Generic legal phrasing matches
        # generic clause language across the whole corpus, so it scored *higher*
        # (0.55) than correct hits (0.42) while retrieving nothing relevant.
        sub: list[str] = [query]
        for feature in state.get("features", [])[:2]:
            sub.append(_ground(feature, product))
        for concept in state.get("concepts", [])[:2]:
            sub.append(_ground(concept, product, template="{text} in {product}"))
    else:
        # Broadened pass: drop the specifics that failed and reach for the
        # general principles instead.
        state["broadened"] = True
        sub = [
            query,
            f"{state.get('product_type', 'financial product')} compliance assessment criteria",
            "prohibited features in Islamic financial products",
            "permissible alternative structures",
        ]

    deduped: list[str] = []
    for item in sub:
        cleaned = " ".join(item.split())
        # A sub-query too thin to carry topic ends up matching generic clause
        # language everywhere. One observed failure derived "Applies to
        # depositors permissibility ruling" from a vague feature; it scored
        # 0.56 against wholly irrelevant clauses and displaced correct hits.
        if len(cleaned.split()) < MIN_SUB_QUERY_WORDS:
            continue
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)

    # The raw query is always kept, even if the filter emptied everything else.
    state["sub_queries"] = deduped[:MAX_SUB_QUERIES] or [query]
    return state


# --- 3. retrieve ---------------------------------------------------------


def _embed_sub_queries(
    sub_queries: list[str], state: AgentState, deps: AgentDeps
) -> list[tuple[str, list[float] | None]]:
    """Embed every sub-query, preferring one batched call.

    Sub-queries were embedded one at a time, which cost four sequential round
    trips where one would do. Measured against the live endpoint: 1,904 ms
    sequential versus 679 ms batched, so this is worth roughly 1.2 s on every
    assessment. It costs nothing, because embeddings are billed per token and
    the token count is identical either way; the win is round trips, plus the
    rate-limit headroom of 50k requests a day instead of 200k.

    Falls back to per-query calls if the batch fails, because the loop it
    replaced isolated errors: one malformed sub-query must not cost us the other
    three. That property is worth more than the extra round trips on the rare
    path where it matters.
    """
    if not sub_queries:
        return []

    try:
        vectors = deps.embedder.embed_documents(sub_queries)
        if len(vectors) == len(sub_queries):
            return list(zip(sub_queries, vectors, strict=True))
        _error(
            state,
            f"batch embedding returned {len(vectors)} vectors for "
            f"{len(sub_queries)} sub-queries; falling back to individual calls",
        )
    except Exception as exc:  # noqa: BLE001
        _error(state, f"batch embedding failed, falling back: {exc}")

    out: list[tuple[str, list[float] | None]] = []
    for sub_query in sub_queries:
        try:
            out.append((sub_query, deps.embedder.embed_query(sub_query)))
        except Exception as exc:  # noqa: BLE001 - one bad sub-query must not abort
            _error(state, f"embedding failed for {sub_query!r}: {exc}")
            out.append((sub_query, None))
    return out


def retrieve(state: AgentState, deps: AgentDeps) -> AgentState:
    """Search the store for each sub-query, then merge and rank the results."""
    _mark(state, "retrieve")

    best: dict[str, SearchHit] = {}
    fused: dict[str, float] = {}

    for sub_query, vector in _embed_sub_queries(
        list(state.get("sub_queries", [])), state, deps
    ):
        if vector is None:
            continue
        try:
            hits = deps.store.search(vector, top_k=deps.top_k)
        except Exception as exc:  # noqa: BLE001 - one bad sub-query must not abort
            _error(state, f"retrieval failed for {sub_query!r}: {exc}")
            continue

        for rank, hit in enumerate(hits, start=1):
            # Reciprocal rank fusion. Ordering by raw similarity across
            # different sub-queries compares scores that are not on a common
            # scale: measured on one failure, a vague sub-query scored 0.56 on
            # irrelevant clauses while the user's own question scored 0.42 on
            # the governing ones, so the wrong results displaced the right ones.
            # Fusing on rank makes a sub-query's *ordering* count and its
            # absolute scores irrelevant, so no single query can dominate.
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)

            # The raw similarity is still kept, for display and for the
            # coverage threshold that decides NEEDS_REVIEW.
            existing = best.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                best[hit.chunk_id] = hit

    candidates = [
        best[chunk_id]
        for chunk_id in sorted(fused, key=lambda c: (-fused[c], -best[c].score))
    ]

    # Rerank a wider pool than we finally use. Fusion fixes the comparability of
    # scores across sub-queries but is still bi-encoder similarity underneath;
    # a cross-encoder judges the query and clause together and is far better at
    # telling a governing rule from text that merely reads like one.
    ranked = candidates[:MAX_EXCERPTS]

    if deps.reranker is not None and len(candidates) > 1:
        pool = candidates[: deps.rerank_candidates]
        try:
            order = deps.reranker.rerank(
                query=state["query"],
                documents=[c.embedding_context() for c in pool],
                top_n=MAX_EXCERPTS,
            )
        except Exception as exc:  # noqa: BLE001
            # Reranking is a quality improvement, never a dependency. The
            # supplied ordering is already usable, so a failure here degrades
            # the ranking rather than the assessment.
            _error(state, f"rerank failed, using fused order: {exc}")
            order = []

        reordered = [pool[i] for i, _ in order if 0 <= i < len(pool)]
        if reordered:
            ranked = reordered
            state["reranked"] = True
    state["hits"] = ranked
    state["top_score"] = ranked[0].score if ranked else 0.0

    log.info(
        "retrieval.result",
        extra={
            "sub_queries": state.get("sub_queries", []),
            "chunks": [h.to_log() for h in ranked],
            "top_score": state["top_score"],
        },
    )
    return state


def should_retrieve(state: AgentState) -> str:
    """Conditional edge: act on the intent ``parse_query`` classified.

    Only an ASSESSMENT reaches retrieval. A greeting and an off-topic question
    both go straight to the verdict, where they get different answers.

    The scope check used to live only in ``should_broaden``, one node too late:
    the query was embedded, searched and *reranked* before anything consulted
    the flag. Three reasons that was worth fixing beyond the ~1.3s.

    The reranker bills for work already known to be pointless. The query text
    reaches a second upstream provider after the system has decided it has no
    business processing it, which is exactly the egress §5 Risk 1 is about. And
    the nonsense hits were written into the trace and the logs — a greeting
    retrieved Salam and Online Dealings clauses at a top score of 0.11 — where
    they read as though they had been considered.

    Returns the name of the next node.
    """
    if state.get("intent") == INTENT_GREETING:
        return "decide_verdict"
    if state.get("in_scope") is False:
        return "decide_verdict"
    return "plan_retrieval"


def should_broaden(state: AgentState, deps: AgentDeps) -> str:
    """Conditional edge: retry retrieval once when the first pass is weak.

    Returns the name of the next node.
    """
    # Retained as a guard rather than removed. ``should_retrieve`` now catches
    # this before retrieval, but a node that decides a verdict's fate should not
    # depend on an upstream edge having done its job.
    if state.get("in_scope") is False:
        return "decide_verdict"

    weak = state.get("top_score", 0.0) < deps.score_threshold
    if weak and state.get("retrieval_round", 1) < MAX_RETRIEVAL_ROUNDS:
        return "plan_retrieval"
    return "assess"


# --- 4. assess -----------------------------------------------------------


_SEVERITY_ALIASES = {
    "PROHIBITED": Severity.PROHIBITED,
    "NON_COMPLIANT": Severity.PROHIBITED,
    "CONDITIONAL": Severity.CONDITIONAL,
    "UNRESOLVED": Severity.UNRESOLVED,
    "UNCLEAR": Severity.UNRESOLVED,
    "PERMISSIBLE": Severity.PERMISSIBLE,
    "COMPLIANT": Severity.PERMISSIBLE,
}


def _coerce_severity(raw: Any) -> Severity:
    """Map a model's severity string onto the enum.

    Unrecognised values become UNRESOLVED rather than raising: a malformed
    severity is exactly the case where a human should look.
    """
    return _SEVERITY_ALIASES.get(str(raw).strip().upper(), Severity.UNRESOLVED)


def _parse_findings(raw: Any, hits: list[SearchHit]) -> list[Finding]:
    by_id = {h.chunk_id: h for h in hits}
    findings: list[Finding] = []

    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue

        citations: list[Citation] = []
        for cite in item.get("citations", []) or []:
            if not isinstance(cite, dict):
                continue
            chunk_id = str(cite.get("chunk_id", "")).strip()
            hit = by_id.get(chunk_id)
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    doc_id=hit.doc_id if hit else "",
                    section=hit.section_label if hit else "",
                    title=hit.title if hit else "",
                    quote=str(cite.get("quote", ""))[:400],
                )
            )

        findings.append(
            Finding(
                issue=str(item.get("issue", "unspecified")).strip() or "unspecified",
                severity=_coerce_severity(item.get("severity")),
                explanation=str(item.get("explanation", "")).strip(),
                citations=citations,
            )
        )

    return findings


def assess(state: AgentState, deps: AgentDeps) -> AgentState:
    """Ask the model for per-issue findings grounded in the retrieved excerpts."""
    _mark(state, "assess")

    hits = state.get("hits", [])
    if not hits:
        state["findings"] = []
        state["summary"] = ""
        state["recommended_actions"] = []
        state["open_questions"] = []
        return state

    user = render_assess_user(
        state["query"],
        state.get("proposal_summary", state["query"]),
        state.get("features", []),
        hits,
    )

    try:
        parsed, response = deps.llm.complete_json(
            ASSESS_SYSTEM, user, max_tokens=deps.max_tokens
        )
        _record_call(state, "assess", response)
    except LLMError as exc:
        _error(state, f"assess failed: {exc}")
        state["findings"] = []
        state["summary"] = ""
        state["recommended_actions"] = []
        state["open_questions"] = []
        return state

    state["findings"] = _parse_findings(parsed.get("findings"), hits)
    state["summary"] = str(parsed.get("summary", "")).strip()
    state["recommended_actions"] = [
        str(a).strip()
        for a in parsed.get("recommended_actions", []) or []
        if str(a).strip()
    ]
    state["open_questions"] = [
        str(q).strip()
        for q in parsed.get("open_questions", []) or []
        if str(q).strip()
    ]
    return state


# --- 5. verify_citations -------------------------------------------------


def verify(state: AgentState, deps: AgentDeps) -> AgentState:
    """Strip citations that do not correspond to a retrieved chunk."""
    _mark(state, "verify_citations")

    retrieved_ids = {h.chunk_id for h in state.get("hits", [])}
    cleaned, rejected = verify_citations(state.get("findings", []), retrieved_ids)

    state["findings"] = cleaned
    state["rejected_citations"] = rejected

    if rejected:
        log.warning(
            "citations.rejected",
            extra={"rejected": rejected, "retrieved": sorted(retrieved_ids)},
        )
    return state


# --- 6. decide_verdict ---------------------------------------------------


def decide_verdict(state: AgentState, deps: AgentDeps) -> AgentState:
    """Apply the deterministic rules to the verified findings."""
    _mark(state, "decide_verdict")

    decision = decide(
        state.get("findings", []),
        state.get("top_score", 0.0),
        out_of_scope=state.get("in_scope") is False,
        greeting=state.get("intent") == INTENT_GREETING,
        coverage_threshold=deps.score_threshold,
    )
    state["decision"] = decision

    log.info(
        "verdict.decided",
        extra={
            "verdict": decision.verdict.value,
            "rule": decision.rule,
            "confidence": decision.confidence,
        },
    )
    return state
