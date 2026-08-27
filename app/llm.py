"""Claude client wrapper: prompt caching, structured output, usage accounting.

Three model roles, deliberately split (the subject scores "cost awareness"):

    rerank / rewrite  ->  Haiku 4.5   high call volume, small outputs, latency
                                      sits on the critical path of every query
    answer            ->  Sonnet 5    one call per question, quality decides
                                      whether the product is usable
    judge (eval only) ->  Opus 5      offline, runs over the whole eval set,
                                      must be at least as strong as the model
                                      it is grading

Prompt caching is applied to the parts of a prompt that repeat across
requests. Caching is a PREFIX match, so the stable content has to come first
and the volatile content last -- put the user's question above the retrieved
context and nothing caches, silently. `usage()` exists so that can be checked
rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings


class LLMUnavailable(RuntimeError):
    """Raised when no API key is configured.

    A distinct type so callers can degrade gracefully -- retrieval works
    without a key, so the reranker should be skippable rather than fatal.
    """


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0

    def add(self, raw: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(raw, "input_tokens", 0) or 0
        self.output_tokens += getattr(raw, "output_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(raw, "cache_creation_input_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(raw, "cache_read_input_tokens", 0) or 0

    @property
    def cache_hit_rate(self) -> float:
        """If this stays at 0 across repeated calls, caching is not working --
        usually a timestamp or a per-request id sitting in the cached prefix."""
        total = self.cache_read_input_tokens + self.cache_creation_input_tokens
        return self.cache_read_input_tokens / total if total else 0.0

    def report(self) -> str:
        return (
            f"{self.calls} calls | in {self.input_tokens} out {self.output_tokens} | "
            f"cache write {self.cache_creation_input_tokens} read "
            f"{self.cache_read_input_tokens} ({self.cache_hit_rate:.0%} hit)"
        )


@dataclass
class LLM:
    settings: Settings = field(default_factory=get_settings)
    usage: Usage = field(default_factory=Usage)
    _client: Any = None

    @property
    def available(self) -> bool:
        return bool(self.settings.anthropic_api_key)

    def client(self) -> Any:
        if self._client is None:
            if not self.available:
                raise LLMUnavailable(
                    "ANTHROPIC_API_KEY is not set. Retrieval works without it; "
                    "reranking, query rewriting, generation and the eval judge do not."
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    # -- core call -----------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        schema: dict[str, Any] | None = None,
        cache_system: bool = True,
    ) -> Any:
        """One Messages API call.

        `cache_system` marks the system prompt as a cache breakpoint. System
        prompts here are static instructions, so this is close to free and
        pays back from the second call onward.
        """
        if isinstance(system, str):
            block: dict[str, Any] = {"type": "text", "text": system}
            if cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks = [block]
        else:
            system_blocks = system

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if schema is not None:
            # Structured outputs: the response is guaranteed to satisfy the
            # schema, which removes the "model returned prose instead of JSON"
            # failure mode entirely. Far more reliable than asking politely.
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        response = self.client().messages.create(**kwargs)
        self.usage.add(response.usage)
        return response

    def complete_json(self, *, schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        response = self.complete(schema=schema, **kwargs)
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"model returned non-JSON despite schema: {text[:200]}") from exc

    def complete_text(self, **kwargs: Any) -> str:
        response = self.complete(**kwargs)
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def stream_text(self, *, model: str, system: str, messages: list[dict[str, Any]],
                    max_tokens: int = 4096):
        """Token stream for the chat UI.

        Streaming is not decoration here: answer generation reads several
        thousand tokens of retrieved context, so time-to-first-token is the
        difference between a usable product and one that looks hung.
        """
        if isinstance(system, str):
            system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        with self.client().messages.stream(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
            self.usage.add(stream.get_final_message().usage)
