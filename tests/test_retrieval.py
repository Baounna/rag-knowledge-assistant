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
    settings = replace(get_settings(), anthropic_api_key=kw.pop("key", ""))
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


def test_confidence_prefers_rerank_score_when_present():
    c = RetrievedChunk(chunk_id="x", text="t", heading_trail=[], source="s",
                       url=None, fusion_score=0.01)
    assert c.confidence == pytest.approx(0.01)
    c.rerank_score = 8.0
    assert c.confidence == pytest.approx(0.8)


def test_llm_raises_a_typed_error_when_unconfigured():
    with pytest.raises(LLMUnavailable):
        LLM(replace(get_settings(), anthropic_api_key="")).client()


def test_usage_reports_cache_hit_rate():
    u = Usage()
    assert u.cache_hit_rate == 0.0
    u.cache_creation_input_tokens, u.cache_read_input_tokens = 100, 900
    assert u.cache_hit_rate == pytest.approx(0.9)
