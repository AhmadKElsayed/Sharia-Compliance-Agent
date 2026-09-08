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


# --- prompt logging ------------------------------------------------------


def _events(caplog, name: str) -> list:
    return [r for r in caplog.records if r.getMessage() == name]


def test_prompt_is_logged_in_full(caplog) -> None:
    """The brief requires logging the prompt sent; a trace without it cannot
    attribute a wrong verdict to retrieval, prompt, or model."""
    caplog.set_level("INFO", logger="sharia.llm")
    client, _ = _capturing(log_prompts=True)
    client.complete_json("SYSTEM-MARKER", "USER-MARKER")

    req = _events(caplog, "llm.request")
    assert len(req) == 1
    assert req[0].system_prompt == "SYSTEM-MARKER"
    assert req[0].user_prompt == "USER-MARKER"
    assert req[0].model == "m"


def test_response_content_is_logged(caplog) -> None:
    caplog.set_level("INFO", logger="sharia.llm")
    client, _ = _capturing(log_prompts=True)
    client.complete_json("s", "u")

    resp = _events(caplog, "llm.response")
    assert len(resp) == 1
    assert resp[0].content == '{"ok": true}'
    assert resp[0].finish_reason == "stop"
    assert resp[0].latency_ms >= 0


def test_log_prompts_false_omits_text_but_keeps_metadata(caplog) -> None:
    """Sensitive query text must be suppressible without losing the audit
    skeleton -- see DOCUMENTATION.md §5, Risk 3."""
    caplog.set_level("INFO", logger="sharia.llm")
    client, _ = _capturing(log_prompts=False)
    client.complete_json("SYSTEM-MARKER", "USER-MARKER")

    req = _events(caplog, "llm.request")[0]
    resp = _events(caplog, "llm.response")[0]

    assert not hasattr(req, "system_prompt")
    assert not hasattr(req, "user_prompt")
    assert not hasattr(resp, "content")
    # Metadata still present, so the call remains auditable.
    assert req.system_chars == len("SYSTEM-MARKER")
    assert req.user_chars == len("USER-MARKER")
    assert resp.completion_tokens == 5


def test_oversized_prompts_are_clipped_not_dropped(caplog) -> None:
    from app.agent.llm import MAX_LOGGED_PROMPT_CHARS

    caplog.set_level("INFO", logger="sharia.llm")
    client, _ = _capturing(log_prompts=True)
    huge = "x" * (MAX_LOGGED_PROMPT_CHARS + 5000)
    client.complete_json("s", huge)

    logged = _events(caplog, "llm.request")[0].user_prompt
    assert len(logged) < len(huge)
    assert "clipped" in logged
    assert str(len(huge)) in logged


def test_request_is_logged_before_the_call_so_failures_are_reproducible(caplog) -> None:
    caplog.set_level("INFO", logger="sharia.llm")
    client, _ = _client([RuntimeError("upstream down")], ["model-a"])
    client._log_prompts = True  # noqa: SLF001
    with pytest.raises(LLMError):
        client.complete_json("SYSTEM-MARKER", "USER-MARKER")

    req = _events(caplog, "llm.request")
    assert len(req) == 1, "a failed call must still record what was sent"
    assert req[0].user_prompt == "USER-MARKER"
    assert not _events(caplog, "llm.response")


def test_repair_pass_is_logged_separately(caplog) -> None:
    caplog.set_level("INFO", logger="sharia.llm")
    client, _ = _client(["not json", '{"fixed": true}'], ["model-a"])
    client._log_prompts = True  # noqa: SLF001
    client.complete_json("s", "u")

    assert len(_events(caplog, "llm.request")) == 2, "repair call must be visible"
