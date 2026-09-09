"""Shared pytest configuration.

Isolates the suite from the developer's environment. Without this, tests that
start the app through ``TestClient`` read the real ``.env`` and inherit the
production log path, so every run appended its stubbed output to
``logs/app.jsonl``.

That mattered for more than tidiness. ``logs/app.jsonl`` is the artefact you
read to see what the observability layer actually produces, and one test run
wrote 168 lines of scripted-LLM events, fixture trace IDs and deliberate
negative-test errors into it. The file stopped resembling the system it was
supposed to demonstrate.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def isolate_logging(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the suite's logging at a temp file, not the real log.

    Session-scoped and autouse: this must be in place before the first
    ``TestClient`` triggers the app lifespan, and no individual test should have
    to remember it.
    """
    log_dir = tmp_path_factory.mktemp("logs")
    os.environ["LOG_FILE"] = str(log_dir / "test.jsonl")

    # Settings are cached, so a value read before this fixture ran would persist
    # for the whole session.
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_stray_log_writes() -> None:
    """Fail loudly if a test writes to the production log path.

    A regression here is silent -- the suite still passes, the log file just
    quietly fills with test output again -- so it is worth asserting rather
    than trusting the fixture above to stay wired up.
    """
    real = Path("logs/app.jsonl")
    before = real.stat().st_size if real.exists() else None
    yield
    after = real.stat().st_size if real.exists() else None
    if before is not None and after is not None and after != before:
        pytest.fail(
            "a test wrote to logs/app.jsonl; tests must log to a temp path "
            "(see the isolate_logging fixture)"
        )
