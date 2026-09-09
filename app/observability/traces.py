"""In-process trace collection for replay.

Every log record carrying a trace ID is appended to a bounded per-trace buffer,
so ``GET /traces/{id}`` can show a reviewer exactly what the agent retrieved,
what prompt it sent, and which rule produced the verdict.

Collection works by attaching a logging handler rather than by instrumenting
call sites: the nodes already log ``retrieval.result`` and ``verdict.decided``,
so those become trace events for free and cannot drift out of sync with the
logs.

**This is deliberately a demonstrator, not the production audit trail.** The
buffer is in-process and bounded, so traces are lost on restart and invisible to
other instances. A regulated deployment needs an append-only, tamper-evident
store with a retention policy -- see DOCUMENTATION.md §4.2 and §5.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

DEFAULT_MAX_TRACES = 500
DEFAULT_MAX_EVENTS = 200


class TraceStore:
    """Bounded, thread-safe store of recent traces.

    Eviction is least-recently-inserted. Bounded on both axes so a long-running
    process cannot grow without limit, and so one pathological request cannot
    evict every other trace.
    """

    def __init__(
        self,
        max_traces: int = DEFAULT_MAX_TRACES,
        max_events: int = DEFAULT_MAX_EVENTS,
    ) -> None:
        self._traces: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._max_traces = max_traces
        self._max_events = max_events
        self._lock = threading.Lock()

    def add(self, trace_id: str, event: dict[str, Any]) -> None:
        if not trace_id:
            return
        with self._lock:
            events = self._traces.get(trace_id)
            if events is None:
                events = []
                self._traces[trace_id] = events
                self._traces.move_to_end(trace_id)
                while len(self._traces) > self._max_traces:
                    self._traces.popitem(last=False)
            if len(events) < self._max_events:
                events.append(event)

    def get(self, trace_id: str) -> list[dict[str, Any]] | None:
        with self._lock:
            events = self._traces.get(trace_id)
            return list(events) if events is not None else None

    def recent(self, limit: int = 20) -> list[str]:
        with self._lock:
            return list(reversed(list(self._traces.keys())))[:limit]

    def __len__(self) -> int:
        with self._lock:
            return len(self._traces)


TRACES = TraceStore()


class TraceCollectorHandler(logging.Handler):
    """Route log records into the trace store, keyed by trace ID."""

    def __init__(self, store: TraceStore | None = None) -> None:
        super().__init__()
        self.store = store or TRACES

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from app.observability.logging_setup import record_payload

            payload = record_payload(record)
            trace_id = payload.pop("trace_id", "")
            self.store.add(trace_id, payload)
        except Exception:  # noqa: BLE001
            # A telemetry failure must never break the request it is describing.
            self.handleError(record)
