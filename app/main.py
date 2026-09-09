"""FastAPI application.

Endpoints:

- ``POST /assess``           submit a query, receive a structured verdict
- ``GET  /health``           liveness and readiness
- ``GET  /corpus``           what is indexed
- ``GET  /traces/{id}``      replay an assessment for audit
- ``GET  /docs``             OpenAPI UI

Dependencies are built once during lifespan and reused. The agent is
synchronous and an assessment takes 9-20s, so each request holds a worker for
that duration -- acceptable for a demonstrator, and the first thing to change
for production (see DOCUMENTATION.md §2.3).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.agent.graph import build_graph, initial_state
from app.agent.llm import LLMClient, LLMError
from app.agent.nodes import AgentDeps
from app.config import Settings, get_settings
from app.observability.logging_setup import configure_logging
from app.observability.trace import TraceIDMiddleware, get_trace_id
from app.observability.traces import TRACES
from app.schemas import (
    AssessRequest,
    AssessResponse,
    CorpusResponse,
    DependencyHealth,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    TraceEvent,
    TraceResponse,
    to_assess_response,
)

log = logging.getLogger("sharia.api")

VERSION = "0.1.0"

# Populated during lifespan.
_state: dict[str, Any] = {}


def build_dependencies(settings: Settings) -> AgentDeps:
    """Construct the agent's collaborators from configuration."""
    from app.rag.ingest import build_embedder, build_store
    from app.rag.rerank import OpenRouterReranker

    llm = LLMClient(
        api_key=settings.openrouter_api_key,
        models=settings.model_chain,
        base_url=settings.openrouter_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        provider_sort=settings.openrouter_provider_sort,
        reasoning_effort=settings.llm_reasoning_effort,
        log_prompts=settings.log_prompts,
    )
    reranker = (
        OpenRouterReranker(
            api_key=settings.openrouter_api_key,
            model=settings.rerank_model,
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
        )
        if settings.rerank_enabled
        else None
    )
    return AgentDeps(
        llm=llm,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
        embedder=build_embedder(settings),
        store=build_store(settings),
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
        max_tokens=settings.llm_max_tokens,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    settings = get_settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)

    missing = settings.missing_required()
    _state["settings"] = settings
    _state["missing"] = missing

    if missing:
        # Start anyway and report the problem on /health. A container that dies
        # on a missing variable is harder to diagnose than one that explains
        # itself.
        log.error("startup.incomplete_config", extra={"missing": missing})
        _state["deps"] = None
        _state["graph"] = None
    else:
        deps = build_dependencies(settings)
        _state["deps"] = deps
        _state["graph"] = build_graph(deps)
        log.info(
            "startup.ready",
            extra={
                "collection": settings.qdrant_collection,
                "chat_model": settings.openrouter_model,
                "embedding_model": settings.openrouter_embedding_model,
            },
        )

    yield
    log.info("shutdown")
    _state.clear()


app = FastAPI(
    title="Sharia Compliance Agent",
    version=VERSION,
    description=(
        "Assesses whether a proposed financial product or transaction is "
        "Sharia-compliant, returning a structured verdict with citations into "
        "the AAOIFI Shari'ah Standards. Decision support for a qualified "
        "reviewer; not a Sharia ruling."
    ),
    lifespan=lifespan,
)
app.add_middleware(TraceIDMiddleware)


def _error(code: str, message: str, http_status: int) -> JSONResponse:
    body = ErrorResponse(
        trace_id=get_trace_id(),
        error=ErrorDetail(code=code, message=message),
    )
    return JSONResponse(status_code=http_status, content=body.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):  # noqa: ANN201
    log.exception("request.unhandled_error", extra={"path": request.url.path})
    return _error(
        "internal_error",
        "An unexpected error occurred. Quote the trace ID when reporting this.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness",
    responses={503: {"model": HealthResponse}},
)
async def health() -> JSONResponse:
    """Report readiness honestly.

    A 200 means the service can actually answer an assessment. If Qdrant is
    unreachable or configuration is incomplete it returns 503, because a green
    check while a dependency is down is worse than no check at all.
    """
    settings: Settings | None = _state.get("settings")
    missing: list[str] = _state.get("missing") or []
    deps: AgentDeps | None = _state.get("deps")

    dependencies: list[DependencyHealth] = []
    chunks = 0

    if deps is None:
        dependencies.append(
            DependencyHealth(name="config", ok=False, detail="incomplete configuration")
        )
    else:
        try:
            reachable = deps.store.ping()
            chunks = deps.store.count() if reachable else 0
            dependencies.append(
                DependencyHealth(
                    name="qdrant",
                    ok=reachable and chunks > 0,
                    detail=(
                        f"{chunks} vectors"
                        if reachable and chunks
                        else "reachable but empty"
                        if reachable
                        else "unreachable"
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            dependencies.append(
                DependencyHealth(name="qdrant", ok=False, detail=type(exc).__name__)
            )
        dependencies.append(
            DependencyHealth(
                name="openrouter",
                ok=bool(settings and settings.openrouter_api_key),
                detail="key configured" if settings and settings.openrouter_api_key else "no key",
            )
        )

    ok = bool(deps) and not missing and all(d.ok for d in dependencies)
    body = HealthResponse(
        status="ok" if ok else "degraded",
        version=VERSION,
        corpus_chunks=chunks,
        collection=settings.qdrant_collection if settings else "",
        chat_model=settings.openrouter_model if settings else "",
        embedding_model=settings.openrouter_embedding_model if settings else "",
        dependencies=dependencies,
        missing_config=missing,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body.model_dump(),
    )


@app.post(
    "/assess",
    response_model=AssessResponse,
    summary="Assess a proposed product or transaction",
    responses={
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def assess(payload: AssessRequest) -> Any:
    """Run the compliance graph over a plain-English query."""
    graph = _state.get("graph")
    if graph is None:
        return _error(
            "not_ready",
            "The service is not configured. Check /health.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    trace_id = get_trace_id()
    query = payload.query.strip()
    started = time.perf_counter()

    log.info("request.received", extra={"path": "/assess", "query": query})

    try:
        # The graph is synchronous; run it off the event loop so the server can
        # continue serving health checks during a 9-20s assessment.
        from anyio import to_thread

        state = await to_thread.run_sync(graph.invoke, initial_state(query, trace_id))
    except LLMError as exc:
        log.error("assess.llm_failed", extra={"detail": str(exc)})
        return _error(
            "upstream_llm_error",
            "The language model could not be reached or returned no usable "
            "response. Retry shortly.",
            status.HTTP_502_BAD_GATEWAY,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("assess.failed", extra={"detail": type(exc).__name__})
        return _error(
            "assessment_failed",
            "The assessment could not be completed. Quote the trace ID when "
            "reporting this.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    response = to_assess_response(state, trace_id, latency_ms)

    log.info(
        "request.completed",
        extra={
            "verdict": response.verdict,
            "rule": response.rule,
            "confidence": response.confidence,
            "latency_ms": latency_ms,
            "findings": len(response.findings),
            "rejected_citations": response.rejected_citations,
            "errors": state.get("errors") or [],
        },
    )
    return response


@app.get(
    "/corpus",
    response_model=CorpusResponse,
    summary="What is indexed",
    responses={503: {"model": ErrorResponse}},
)
async def corpus() -> Any:
    """Summarise the indexed corpus without exposing full clause text.

    Deliberately returns counts and identifiers only: the standards are
    licensed, so a public endpoint should not serve the text back in bulk.
    """
    deps: AgentDeps | None = _state.get("deps")
    settings: Settings | None = _state.get("settings")
    if deps is None or settings is None:
        return _error(
            "not_ready", "The service is not configured.", status.HTTP_503_SERVICE_UNAVAILABLE
        )

    try:
        count = deps.store.count()
    except Exception:  # noqa: BLE001
        return _error(
            "store_unavailable",
            "The vector store is unreachable.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return CorpusResponse(
        collection=settings.qdrant_collection,
        chunks=count,
        embedding_model=settings.openrouter_embedding_model,
        embedding_dim=settings.embedding_dim,
        chunking="clause-level with hierarchical contextual headers",
        source="AAOIFI Shari'ah Standards (English edition)",
        note=(
            "Clause text is licensed and is not served in bulk. Assessments "
            "return short quotes with citations."
        ),
    )


@app.get(
    "/traces/{trace_id}",
    response_model=TraceResponse,
    summary="Replay an assessment",
    responses={404: {"model": ErrorResponse}},
)
async def get_trace(trace_id: str) -> Any:
    """Return the recorded events for one assessment.

    In-process and bounded, so traces are lost on restart and are not shared
    across instances. Sufficient for debugging a demonstrator; a regulated
    deployment needs durable, tamper-evident storage.
    """
    events = TRACES.get(trace_id)
    if events is None:
        return _error(
            "trace_not_found",
            "No trace with that ID is retained. Traces are held in memory and "
            "are lost on restart.",
            status.HTTP_404_NOT_FOUND,
        )

    known = {"ts", "level", "event", "logger"}
    return TraceResponse(
        trace_id=trace_id,
        event_count=len(events),
        events=[
            TraceEvent(
                ts=e.get("ts", 0.0),
                level=e.get("level", ""),
                event=e.get("event", ""),
                logger=e.get("logger", ""),
                data={k: v for k, v in e.items() if k not in known},
            )
            for e in events
        ],
    )
