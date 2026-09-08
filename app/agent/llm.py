"""OpenRouter chat client.

Wraps the OpenAI-compatible endpoint with the three things the agent actually
needs: a model fallback chain, defensive JSON extraction, and a single repair
attempt when a model returns something that is nearly-but-not-quite JSON.

The repair path matters more than it looks. The configured model is a reasoning
model, which sometimes wraps its answer in prose or a fenced code block even
when asked not to. Rather than failing the request, the extractor pulls the
JSON object out; only if that fails does it spend a second call asking for a
correction.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

# Matches a ```json ... ``` fence, capturing the body.
FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """Raised when every model in the chain fails."""


class JSONParseError(LLMError):
    """Raised when a response could not be coerced into JSON."""


@dataclass
class LLMResponse:
    """A completed chat call, with everything the trace log needs."""

    content: str
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    attempts: list[str] = field(default_factory=list)
    repaired: bool = False

    def to_log(self, include_content: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "attempts": self.attempts,
            "repaired": self.repaired,
        }
        if include_content:
            payload["content"] = self.content
        return payload


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Tries, in order: the whole string, a fenced code block, then the outermost
    brace-balanced span. Raises ``JSONParseError`` if none yields an object.
    """
    if not text or not text.strip():
        raise JSONParseError("empty response")

    candidates: list[str] = [text.strip()]

    fence = FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())

    span = _balanced_object(text)
    if span:
        candidates.append(span)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise JSONParseError(f"no JSON object found in response: {text[:300]!r}")


def _balanced_object(text: str) -> str | None:
    """Return the first brace-balanced ``{...}`` span, ignoring braces in strings."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class LLMClient:
    """Chat client with a model fallback chain and JSON coercion."""

    def __init__(
        self,
        api_key: str,
        models: list[str],
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        provider_sort: str = "throughput",
        reasoning_enabled: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        if not models:
            raise ValueError("at least one model must be configured")
        self._models = models
        self._temperature = temperature
        self._provider_sort = provider_sort
        self._reasoning_enabled = reasoning_enabled
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def models(self) -> list[str]:
        return list(self._models)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        response_format_json: bool = False,
    ) -> LLMResponse:
        """Call the first model in the chain that succeeds."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        attempts: list[str] = []
        errors: list[str] = []

        for model in self._models:
            attempts.append(model)
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": self._temperature,
                }
                if response_format_json:
                    kwargs["response_format"] = {"type": "json_object"}
                extra: dict[str, Any] = {}
                if self._provider_sort:
                    # Without this, OpenRouter may route to whichever provider is
                    # cheapest rather than fastest. On this model that is the
                    # difference between a 2 second and a 30 second call.
                    extra["provider"] = {"sort": self._provider_sort}
                if not self._reasoning_enabled:
                    # A reasoning pass can consume the whole max_tokens budget
                    # and leave no content at all. See config.llm_reasoning_enabled.
                    extra["reasoning"] = {"enabled": False}
                if extra:
                    kwargs["extra_body"] = extra

                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                usage = response.usage

                content = choice.message.content or ""
                if not content.strip():
                    # Empty content with a length stop is the token-exhaustion
                    # signature. Naming it here beats a downstream JSON error
                    # that looks like a model quality problem.
                    reason = choice.finish_reason or "unknown"
                    raise LLMError(
                        f"{model} returned empty content (finish_reason={reason}); "
                        "the token budget was likely consumed before any answer was "
                        "written — raise max_tokens or disable reasoning"
                    )
                return LLMResponse(
                    content=content,
                    model=response.model or model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    finish_reason=choice.finish_reason or "",
                    attempts=attempts.copy(),
                )
            except Exception as exc:  # noqa: BLE001 - fall through to the next model
                errors.append(f"{model}: {type(exc).__name__}: {exc}")

        raise LLMError("all models failed -> " + " | ".join(errors))

    def complete_json(
        self, system: str, user: str, max_tokens: int = 2048
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Call the model and coerce the reply into a JSON object.

        On a parse failure, one repair call is made that hands the model its own
        malformed output back. A second failure raises rather than looping,
        since a model that cannot produce JSON twice will not manage it on a
        third attempt either.
        """
        response = self.complete(
            system, user, max_tokens=max_tokens, response_format_json=True
        )
        try:
            return extract_json(response.content), response
        except JSONParseError:
            pass

        repair = self.complete(
            system=(
                "You fix malformed JSON. Reply with the corrected JSON object and "
                "nothing else. No prose, no code fences, no explanation."
            ),
            user=(
                "The following was meant to be a single JSON object but could not "
                f"be parsed. Return only the corrected object.\n\n{response.content}"
            ),
            max_tokens=max_tokens,
            response_format_json=True,
        )
        repair.repaired = True
        repair.attempts = response.attempts + repair.attempts
        return extract_json(repair.content), repair
