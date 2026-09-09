"""Structured JSON-lines logging.

One JSON object per line to stdout and to a rotating file. Every record carries
the current trace ID, so a whole assessment can be reconstructed by filtering on
one field.

Anything passed via ``extra=`` is merged into the record, which is what lets
``log.info("retrieval.result", extra={"chunks": [...]})`` in the nodes produce a
structured event without a bespoke schema per call site.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from app.observability.trace import get_trace_id

_STANDARD = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName
    relativeCreated stack_info thread threadName taskName""".split()
)

MAX_VALUE_CHARS = 20_000


def _safe(value: Any) -> Any:
    """Coerce a value into something JSON-serialisable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


def record_payload(record: logging.LogRecord) -> dict[str, Any]:
    """Build the structured payload for a log record."""
    payload: dict[str, Any] = {
        "ts": record.created,
        "level": record.levelname,
        "logger": record.name,
        "event": record.getMessage(),
        "trace_id": getattr(record, "trace_id", "") or get_trace_id(),
    }

    for key, value in record.__dict__.items():
        if key in _STANDARD or key.startswith("_") or key == "trace_id":
            continue
        payload[key] = _safe(value)

    if record.exc_info:
        payload["exception"] = logging.Formatter().formatException(record.exc_info)

    return payload


class JSONFormatter(logging.Formatter):
    """Render a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = record_payload(record)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        if len(line) > MAX_VALUE_CHARS:
            trimmed = {
                k: payload[k]
                for k in ("ts", "level", "logger", "event", "trace_id")
                if k in payload
            }
            trimmed["truncated"] = True
            trimmed["original_size"] = len(line)
            line = json.dumps(trimmed, ensure_ascii=False, default=str)
        return line


def configure_logging(
    level: str = "INFO",
    log_file: str | None = "logs/app.jsonl",
    collect_traces: bool = True,
) -> None:
    """Install JSON logging on the root logger.

    Idempotent: re-running replaces handlers rather than stacking them, which
    matters under a reloader that imports the module more than once.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = JSONFormatter()

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(formatter)
    root.addHandler(stdout)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)

    if collect_traces:
        from app.observability.traces import TraceCollectorHandler

        root.addHandler(TraceCollectorHandler())

    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
