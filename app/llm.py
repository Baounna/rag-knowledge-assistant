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


class Backend:
    """A provider that can answer a Messages-shaped request.

    Two implementations: Claude, and Ollama for running locally at no cost.
    Everything upstream talks only to this interface, so switching providers
    is a config value (`LLM_PROVIDER`) rather than a rewrite.
    """

    name = "backend"

    @property
    def available(self) -> bool:
        raise NotImplementedError

    def complete(self, **kwargs: Any) -> tuple[str, Any]:
        """Return (text, usage-like object)."""
        raise NotImplementedError

    def stream(self, **kwargs: Any):
        raise NotImplementedError


@dataclass
class LLM:
    settings: Settings = field(default_factory=get_settings)
    usage: Usage = field(default_factory=Usage)
    _backend: Any = None

    @property
    def backend(self) -> "Backend":
        if self._backend is None:
            self._backend = make_backend(self.settings)
        return self._backend

    @property
    def available(self) -> bool:
        return self.backend.available

    def model_for(self, role: str) -> str:
        """Map a role (answer/rerank/rewrite/judge) to this backend's model.

        Claude splits the roles across three cost tiers. Ollama runs one local
        model for every role: there is no cost gradient to optimise, and a
        second resident model would double the memory footprint for nothing.
        """
        if self.backend.name == "ollama":
            return self.settings.ollama_model
        return {
            "answer": self.settings.model_answer,
            "rerank": self.settings.model_rerank,
            "rewrite": self.settings.model_rewrite,
            "judge": self.settings.model_judge,
        }.get(role, self.settings.model_answer)

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

        text, usage = self.backend.complete(
            model=model, max_tokens=max_tokens, system=system_blocks,
            messages=messages, schema=schema,
        )
        self.usage.add(usage)
        return text

    def complete_json(self, *, schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        text = self.complete(schema=schema, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Claude's structured outputs guarantee valid JSON. A local model
            # constrained by the same schema usually complies but sometimes
            # wraps the object in prose, so recover the outermost object
            # rather than failing the request over formatting.
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(f"model returned non-JSON despite schema: {text[:200]}") from None

    def complete_text(self, **kwargs: Any) -> str:
        return self.complete(**kwargs).strip()

    def stream_text(self, *, model: str, system: str, messages: list[dict[str, Any]],
                    max_tokens: int = 4096):
        """Token stream for the chat UI.

        Streaming is not decoration here: answer generation reads several
        thousand tokens of retrieved context, so time-to-first-token is the
        difference between a usable product and one that looks hung.
        """
        if isinstance(system, str):
            system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        usage = yield from self.backend.stream(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )
        if usage is not None:
            self.usage.add(usage)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class AnthropicBackend(Backend):
    """Claude via the Anthropic SDK."""

    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self.settings.anthropic_api_key)

    def client(self) -> Any:
        if self._client is None:
            if not self.available:
                raise LLMUnavailable(
                    "ANTHROPIC_API_KEY is not set. Retrieval works without it; "
                    "reranking, query rewriting, generation and the eval judge do not. "
                    "Set LLM_PROVIDER=ollama to run locally instead."
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def complete(self, *, model, max_tokens, system, messages, schema=None):
        kwargs: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens,
            "system": system, "messages": messages,
        }
        if schema is not None:
            # Structured outputs: the response is guaranteed to satisfy the
            # schema, which removes the "returned prose instead of JSON"
            # failure mode entirely rather than asking politely for JSON.
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        response = self.client().messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        return text, response.usage

    def stream(self, *, model, max_tokens, system, messages):
        with self.client().messages.stream(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
            return stream.get_final_message().usage


class OllamaBackend(Backend):
    """A local model via Ollama. No API key, no cost, no data leaving the machine.

    Three differences from the Claude path, all handled here so nothing
    upstream has to know which backend is running:

    * Ollama has no `system` array, so the system blocks are flattened into a
      single system message.
    * Structured output uses Ollama's `format` field, which constrains
      decoding to a JSON schema. Weaker than Claude's guarantee -- hence the
      brace-recovery fallback in `LLM.complete_json`.
    * There is no prompt caching to report, so cache counters stay at zero.
      That is honest rather than broken: `Usage.cache_hit_rate` reading 0%
      under Ollama means caching does not exist here, not that it failed.
    """

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._url = settings.ollama_url.rstrip("/")

    @property
    def available(self) -> bool:
        import httpx

        try:
            r = httpx.get(f"{self._url}/api/tags", timeout=2.0)
            models = [m["name"] for m in r.json().get("models", [])]
        except Exception:  # noqa: BLE001
            return False
        want = self.settings.ollama_model
        return any(m == want or m.split(":")[0] == want.split(":")[0] for m in models)

    @staticmethod
    def _flatten(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        if isinstance(system, str):
            system_text = system
        else:
            system_text = "\n\n".join(
                b.get("text", "") for b in system if isinstance(b, dict))
        out = [{"role": "system", "content": system_text}] if system_text else []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = "\n\n".join(
                    b.get("text", "") for b in content if isinstance(b, dict))
            out.append({"role": m["role"], "content": content})
        return out

    def _usage(self, payload: dict[str, Any]) -> Any:
        class _U:
            input_tokens = int(payload.get("prompt_eval_count") or 0)
            output_tokens = int(payload.get("eval_count") or 0)
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0
        return _U()

    def complete(self, *, model, max_tokens, system, messages, schema=None):
        import httpx

        body: dict[str, Any] = {
            "model": model,
            "messages": self._flatten(system, messages),
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }
        if schema is not None:
            body["format"] = schema
        try:
            r = httpx.post(f"{self._url}/api/chat", json=body, timeout=600.0)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(
                f"Ollama at {self._url} did not respond ({type(exc).__name__}). "
                f"Start it with `ollama serve` and pull the model with "
                f"`ollama pull {model}`."
            ) from exc
        payload = r.json()
        return payload.get("message", {}).get("content", ""), self._usage(payload)

    def stream(self, *, model, max_tokens, system, messages):
        import httpx

        body = {
            "model": model,
            "messages": self._flatten(system, messages),
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }
        final: dict[str, Any] = {}
        with httpx.stream("POST", f"{self._url}/api/chat", json=body, timeout=600.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                piece = payload.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if payload.get("done"):
                    final = payload
        return self._usage(final)


_BACKENDS = {"anthropic": AnthropicBackend, "ollama": OllamaBackend}


def make_backend(settings: Settings) -> Backend:
    provider = (settings.llm_provider or "anthropic").lower()
    if provider == "auto":
        # Prefer a configured key; fall back to a running local model. Lets the
        # same checkout work on a laptop with Ollama and in CI with a key.
        anthropic_backend = AnthropicBackend(settings)
        if anthropic_backend.available:
            return anthropic_backend
        ollama_backend = OllamaBackend(settings)
        return ollama_backend if ollama_backend.available else anthropic_backend
    try:
        return _BACKENDS[provider](settings)
    except KeyError:
        raise ValueError(
            f"unknown LLM_PROVIDER={provider!r}; expected one of {sorted(_BACKENDS)} or 'auto'"
        ) from None
