"""Request-scoped trace identity.

A trace ID is minted per request and stored in a ``ContextVar`` so that every
module -- the graph nodes, the LLM client, the store -- can attach it to a log
record without it being threaded through every call signature.

The ID is accepted from an inbound ``X-Trace-Id`` header when present, so a
trace can be correlated across an upstream caller and this service, and is
always returned on the response.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

TRACE_HEADER = "X-Trace-Id"

SAFE_TRACE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    return uuid.uuid4().hex


def get_trace_id() -> str:
    """The current trace ID, or an empty string outside a request."""
    return _trace_id.get()


def set_trace_id(value: str) -> None:
    _trace_id.set(value)


def adopt_or_mint(inbound: str | None) -> str:
    """Use a caller-supplied trace ID when it is well formed, else mint one."""
    if inbound and SAFE_TRACE_ID.match(inbound):
        return inbound
    return new_trace_id()


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Assign a trace ID to every request and return it on the response."""

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        trace_id = adopt_or_mint(request.headers.get(TRACE_HEADER))
        set_trace_id(trace_id)
        # Also on request.state so handlers can read it without the contextvar.
        request.state.trace_id = trace_id

        response = await call_next(request)
        response.headers[TRACE_HEADER] = trace_id
        return response
