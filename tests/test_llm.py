"""Tests for JSON extraction and the LLM client's fallback behaviour.

No network: the client is driven through a stubbed OpenAI-shaped object.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.llm import JSONParseError, LLMClient, LLMError, extract_json


# --- extraction ----------------------------------------------------------


def test_plain_json() -> None:
    assert extract_json('{"verdict": "COMPLIANT"}') == {"verdict": "COMPLIANT"}


def test_fenced_json() -> None:
    text = 'Here is the result:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert extract_json(text) == {"a": 1}


def test_unlabelled_fence() -> None:
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_json_with_surrounding_prose() -> None:
    """A reasoning model often narrates before answering."""
    text = 'Let me think about this. The answer is {"verdict": "NEEDS_REVIEW"} overall.'
    assert extract_json(text) == {"verdict": "NEEDS_REVIEW"}


def test_nested_objects_survive() -> None:
    text = 'result: {"findings": [{"issue": "riba", "cites": {"doc": "SFS-001"}}]}'
    assert extract_json(text)["findings"][0]["cites"]["doc"] == "SFS-001"


def test_braces_inside_strings_do_not_break_balancing() -> None:
    text = 'noise {"explanation": "a } brace { in text", "ok": true} tail'
    parsed = extract_json(text)
    assert parsed["ok"] is True
    assert parsed["explanation"] == "a } brace { in text"


def test_escaped_quote_inside_string() -> None:
    text = r'{"quote": "he said \"riba\" clearly", "n": 1}'
    assert extract_json(text)["n"] == 1


def test_empty_response_raises() -> None:
    with pytest.raises(JSONParseError, match="empty"):
        extract_json("   ")


def test_prose_without_json_raises() -> None:
    with pytest.raises(JSONParseError, match="no JSON object"):
        extract_json("I cannot answer that question.")


def test_bare_array_is_rejected() -> None:
    """The agent contract is an object; a list is a malformed reply."""
    with pytest.raises(JSONParseError):
        extract_json("[1, 2, 3]")


# --- client fallback -----------------------------------------------------


def _completion(content: str, model: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop"
            )
        ],
        model=model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class _StubCompletions:
    def __init__(self, script: list[object]) -> None:
        self.script = script
        self.calls: list[str] = []

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs["model"])
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _completion(str(outcome), kwargs["model"])


def _client(script: list[object], models: list[str]) -> tuple[LLMClient, _StubCompletions]:
    client = LLMClient(api_key="test-key", models=models)
    stub = _StubCompletions(script)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))  # noqa: SLF001
    return client, stub


def test_first_model_wins() -> None:
    client, stub = _client(['{"ok": true}'], ["model-a", "model-b"])
    parsed, response = client.complete_json("sys", "user")
    assert parsed == {"ok": True}
    assert stub.calls == ["model-a"]
    assert response.repaired is False


def test_falls_back_to_second_model() -> None:
    client, stub = _client(
        [RuntimeError("503 upstream"), '{"ok": true}'], ["model-a", "model-b"]
    )
    parsed, response = client.complete_json("sys", "user")
    assert parsed == {"ok": True}
    assert stub.calls == ["model-a", "model-b"]
    assert response.attempts == ["model-a", "model-b"]


def test_all_models_failing_raises_with_detail() -> None:
    client, _ = _client(
        [RuntimeError("boom-a"), RuntimeError("boom-b")], ["model-a", "model-b"]
    )
    with pytest.raises(LLMError, match="all models failed"):
        client.complete_json("sys", "user")


def test_repair_pass_recovers_malformed_json() -> None:
    client, stub = _client(["not json at all", '{"fixed": true}'], ["model-a"])
    parsed, response = client.complete_json("sys", "user")
    assert parsed == {"fixed": True}
    assert response.repaired is True
    assert len(stub.calls) == 2, "exactly one repair attempt"


def test_repair_failing_twice_raises_rather_than_looping() -> None:
    client, stub = _client(["still not json", "nor is this"], ["model-a"])
    with pytest.raises(JSONParseError):
        client.complete_json("sys", "user")
    assert len(stub.calls) == 2, "must not retry indefinitely"


def test_empty_content_is_reported_as_token_exhaustion() -> None:
    """A reasoning model can burn the whole budget and return nothing.

    Observed in practice: 3072/3072 completion tokens were reasoning tokens and
    the content was "". The error must name that cause rather than surfacing as
    a confusing JSON parse failure.
    """
    client, _ = _client(["   "], ["model-a"])
    with pytest.raises(LLMError, match="empty content"):
        client.complete_json("sys", "user")


def test_empty_content_falls_through_to_the_next_model() -> None:
    client, stub = _client(["", '{"ok": true}'], ["model-a", "model-b"])
    parsed, _ = client.complete_json("sys", "user")
    assert parsed == {"ok": True}
    assert stub.calls == ["model-a", "model-b"]


def test_client_requires_key_and_models() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LLMClient(api_key="", models=["m"])
    with pytest.raises(ValueError, match="at least one model"):
        LLMClient(api_key="k", models=[])


# --- reasoning and provider routing --------------------------------------


class _CapturingCompletions(_StubCompletions):
    def __init__(self) -> None:
        super().__init__(['{"ok": true}'])
        self.kwargs: dict = {}

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        self.kwargs = kwargs
        return super().create(**kwargs)


def _capturing(**client_kw) -> tuple[LLMClient, _CapturingCompletions]:
    client = LLMClient(api_key="k", models=["m"], **client_kw)
    stub = _CapturingCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))  # noqa: SLF001
    return client, stub


def test_reasoning_effort_is_sent() -> None:
    client, stub = _capturing(reasoning_effort="low")
    client.complete_json("s", "u")
    assert stub.kwargs["extra_body"]["reasoning"] == {"effort": "low"}


def test_empty_effort_disables_reasoning_explicitly() -> None:
    """Explicit disable, not omission: the provider default is what caused
    a reasoning pass to consume the entire token budget."""
    client, stub = _capturing(reasoning_effort="")
    client.complete_json("s", "u")
    assert stub.kwargs["extra_body"]["reasoning"] == {"enabled": False}


def test_provider_sort_is_sent() -> None:
    client, stub = _capturing(provider_sort="throughput")
    client.complete_json("s", "u")
    assert stub.kwargs["extra_body"]["provider"] == {"sort": "throughput"}


def test_max_tokens_is_forwarded() -> None:
    client, stub = _capturing()
    client.complete_json("s", "u", max_tokens=12000)
    assert stub.kwargs["max_tokens"] == 12000
