"""Configuration validation tests."""

from __future__ import annotations

import pytest

from app.config import Settings


def _settings(**kw) -> Settings:
    base = dict(
        openrouter_api_key="k", qdrant_url="https://x", qdrant_api_key="q",
        _env_file=None,
    )
    return Settings(**{**base, **kw})


@pytest.mark.parametrize("effort", ["low", "medium", "high", ""])
def test_valid_reasoning_efforts(effort: str) -> None:
    assert _settings(llm_reasoning_effort=effort).llm_reasoning_effort == effort


def test_reasoning_effort_is_normalised() -> None:
    assert _settings(llm_reasoning_effort="  HIGH ").llm_reasoning_effort == "high"


def test_invalid_reasoning_effort_is_rejected() -> None:
    with pytest.raises(ValueError, match="llm_reasoning_effort"):
        _settings(llm_reasoning_effort="maximum")


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="log_level"):
        _settings(log_level="verbose")


def test_qdrant_url_trailing_slash_is_stripped() -> None:
    assert _settings(qdrant_url="https://x/").qdrant_url == "https://x"


def test_model_chain_dedupes_and_orders() -> None:
    s = _settings(openrouter_model="a", openrouter_fallback_models="b, a ,c")
    assert s.model_chain == ["a", "b", "c"]


def test_missing_required_reports_each_gap() -> None:
    s = Settings(openrouter_api_key="", qdrant_url="", qdrant_api_key="", _env_file=None)
    assert s.missing_required() == ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]


def test_complete_config_reports_nothing_missing() -> None:
    assert _settings().missing_required() == []
