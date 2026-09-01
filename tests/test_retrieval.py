"""Pipeline behaviour, including the no-API-key path.

The key-free path matters: retrieval must stay usable when the LLM stages are
unavailable, and every eval run of retrieval metrics goes through it.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import LLM, LLMUnavailable, Usage  # noqa: E402
from app.retrieval import RetrievedChunk, Retriever  # noqa: E402
from app.store import SearchHit  # noqa: E402


class FakeStore:
    """Deterministic two-index store, so pipeline tests need no database."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @staticmethod
    def _hits(ids, retriever):
        return [
            SearchHit(chunk_id=c, text=f"body of {c}", heading_trail=["Doc", c],
                      source="Doc", url=f"https://x/{c}", doc_date=None,
                      score=1.0 / (i + 1), rank=i + 1, retriever=retriever)
            for i, c in enumerate(ids)
        ]

    def lexical_search(self, query, k=None, filters=None):
        return self._hits(["lex1", "shared", "lex2"][:k or 3], "lexical")

    def vector_search(self, vec, k=None, filters=None):
        return self._hits(["vec1", "shared", "vec2"][:k or 3], "vector")


class FakeEmbedder:
    model, dim = "fake", 3

    def embed_documents(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0, 0.0]


def _retriever(**kw) -> Retriever:
    settings = replace(get_settings(), anthropic_api_key=kw.pop("key", ""),
                       llm_provider="anthropic")   # never reach a live model
    return Retriever(store=FakeStore(), embedder=FakeEmbedder(),
                     llm=LLM(settings), settings=settings)


def test_works_without_an_api_key():
    """Retrieval must degrade, not fail, when the LLM stages are unavailable."""
    result = _retriever().retrieve("a question", top_k=3, top_n=3)
    assert result.chunks, "hybrid retrieval returned nothing without a key"
    assert any("no API key" in n for n in result.notes)


def test_fusion_promotes_the_chunk_both_indexes_found():
    result = _retriever().retrieve("a question", top_k=3, top_n=3)
    assert result.chunks[0].chunk_id == "shared"
    assert set(result.chunks[0].found_by) == {"lexical", "vector"}


def test_every_stage_is_recorded_for_debugging():
    result = _retriever().retrieve("a question", top_k=3, top_n=2)
    assert set(result.stages) == {"lexical", "vector", "fused", "final"}
    assert len(result.stages["final"]) == 2
    assert set(result.stages["final"]) <= set(result.stages["fused"])


def test_stages_can_be_switched_off_independently():
    """`make eval` has to answer 'did the reranker help?' with a number, which
    requires running the identical pipeline with one stage disabled."""
    r = _retriever()
    plain = r.retrieve("q", use_rewrite=False, use_rerank=False, top_k=3, top_n=3)
    assert plain.queries_used == ["q"]
    assert all(c.rerank_score is None for c in plain.chunks)


def test_top_n_limits_what_reaches_the_prompt():
    result = _retriever().retrieve("q", top_k=3, top_n=1)
    assert len(result.chunks) == 1


def test_context_block_carries_ids_for_citation():
    result = _retriever().retrieve("q", top_k=3, top_n=2)
    block = result.context_block()
    for chunk in result.chunks:
        assert f"[{chunk.chunk_id}]" in block, "model cannot cite an id it never sees"


def test_confidence_is_none_without_a_reranker():
    """RRF scores are rank reciprocals (~0.03 max), not probabilities. Only
    the reranker, which reads the passage, produces a calibrated 0-1 score."""
    c = RetrievedChunk(chunk_id="x", text="t", heading_trail=[], source="s",
                       url=None, fusion_score=0.01)
    assert c.confidence is None
    c.rerank_score = 8.0
    assert c.confidence == pytest.approx(0.8)


def test_unconfigured_backend_raises_a_typed_error():
    """A distinct exception type so callers can degrade gracefully: retrieval
    works without a model, so the reranker must be skippable, not fatal."""
    from app.llm import AnthropicBackend

    settings = replace(get_settings(), anthropic_api_key="", llm_provider="anthropic")
    backend = AnthropicBackend(settings)
    assert not backend.available
    with pytest.raises(LLMUnavailable):
        backend.client()


def test_provider_auto_prefers_a_configured_key():
    from app.llm import make_backend

    with_key = replace(get_settings(), anthropic_api_key="sk-test", llm_provider="auto")
    assert make_backend(with_key).name == "anthropic"


def test_unknown_provider_is_rejected():
    from app.llm import make_backend

    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        make_backend(replace(get_settings(), llm_provider="gpt5"))


def test_ollama_flattens_system_blocks_and_content_lists():
    """Ollama has no system array and no content blocks, so the Claude-shaped
    request has to be flattened -- including the cached context block, whose
    text must survive even though the cache_control does not."""
    from app.llm import OllamaBackend

    msgs = OllamaBackend._flatten(
        [{"type": "text", "text": "SYSTEM RULES", "cache_control": {"type": "ephemeral"}}],
        [{"role": "user", "content": [
            {"type": "text", "text": "CONTEXT", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Question: x"},
        ]}],
    )
    assert msgs[0] == {"role": "system", "content": "SYSTEM RULES"}
    assert "CONTEXT" in msgs[1]["content"] and "Question: x" in msgs[1]["content"]


def test_ollama_reports_no_cache_because_it_has_none():
    """0% cache hit under Ollama means caching does not exist there, not that
    it silently failed -- the distinction matters when reading the eval report."""
    from app.llm import OllamaBackend

    usage = OllamaBackend(get_settings())._usage({"prompt_eval_count": 900, "eval_count": 120})
    assert usage.input_tokens == 900 and usage.output_tokens == 120
    assert usage.cache_read_input_tokens == 0


def test_usage_reports_cache_hit_rate():
    u = Usage()
    assert u.cache_hit_rate == 0.0
    u.cache_creation_input_tokens, u.cache_read_input_tokens = 100, 900
    assert u.cache_hit_rate == pytest.approx(0.9)


def test_defaults_follow_settings_not_hardcoded_on():
    """A caller that omits the stage flags must get the configured pipeline.

    `scripts/demo.py` omits them. Before this, it silently ran query-rewrite and
    reranking -- two LLM calls per question -- on a deployment that had both
    disabled, so the CLI and the app disagreed about what "retrieval" means.
    """
    from dataclasses import replace as _replace

    base = _replace(get_settings(), anthropic_api_key="k", llm_provider="anthropic")
    off = Retriever(store=FakeStore(), embedder=FakeEmbedder(), llm=LLM(base),
                    settings=_replace(base, enable_rewrite=False, enable_rerank=False))
    result = off.retrieve("q", top_k=3, top_n=3)
    assert result.queries_used == ["q"], "rewrite ran despite enable_rewrite=False"
    assert all(c.rerank_score is None for c in result.chunks)
