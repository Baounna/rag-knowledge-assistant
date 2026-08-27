"""The retrieval pipeline.

    question
      -> rewrite      (Haiku)     make it searchable
      -> vector + BM25            two ranked lists, run independently
      -> RRF                      one ranked list
      -> rerank       (Haiku)     read the candidates, reorder by relevance
      -> top N                    what actually reaches the answer prompt

Each stage is individually switchable, because `make eval` has to be able to
answer "did the reranker help?" with a number rather than a belief. Every
stage records the chunk ids it produced, so a retrieval failure can be
attributed to the stage that caused it.

Every LLM stage degrades to a no-op when no API key is configured: without a
key you still get hybrid retrieval, just not rewriting or reranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Settings, get_settings
from .embeddings import EmbeddingProvider, get_embedder
from .fusion import FusedHit, reciprocal_rank_fusion
from .llm import LLM, LLMUnavailable
from .store import Filters, SearchHit, Store

# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = """You rewrite a user's question into search queries for a \
hybrid retrieval system over internal company documents.

Produce two things:

1. `queries`: 1-3 search queries. Include the user's own wording as one of \
them -- their phrasing sometimes matches the document exactly, and discarding \
it loses that. Add variants that use the vocabulary a policy or handbook \
would use rather than the vocabulary an employee would ("expense \
reimbursement" alongside "money back for a work trip"). Resolve pronouns and \
references from the conversation so each query stands alone.

2. `hypothetical_answer`: one or two sentences written as if they were an \
excerpt from the document that answers the question. This is a HyDE passage: \
it is never shown to the user and does not need to be factually correct. Its \
only job is to look like the target document, because a passage embeds closer \
to a passage than a question does.

Be conservative. If the question is already a good query, return it unchanged \
as the only entry."""

REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "hypothetical_answer": {"type": "string"},
    },
    "required": ["queries", "hypothetical_answer"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

RERANK_SYSTEM = """You score how well each passage answers a question.

For each candidate return a relevance score from 0 to 10:

  0-2   unrelated, or about a different subject that shares vocabulary
  3-5   same topic, but does not contain the answer
  6-8   contains part of the answer, or the answer with missing conditions
  9-10  directly and completely answers the question

Judge only what the passage says. Do not use outside knowledge, and do not \
reward a passage for sounding authoritative. A passage that mentions the \
right topic without stating the fact is a 4, not an 8 -- that distinction is \
the entire value of this step, because retrieval already found things that \
merely look relevant.

Score every candidate you are given, using its exact id."""

RERANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 10},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    heading_trail: list[str]
    source: str
    url: str | None
    fusion_score: float
    rerank_score: float | None = None
    found_by: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def confidence(self) -> float:
        """Rerank score on 0-1 when available, else the fusion score.

        The answer stage uses this to decide whether to refuse: fusion scores
        are relative and say nothing about whether ANY chunk is actually
        relevant, whereas a rerank score does.
        """
        return self.rerank_score / 10 if self.rerank_score is not None else self.fusion_score

    def as_context(self) -> str:
        trail = " > ".join(self.heading_trail)
        return f"[{self.chunk_id}] {trail}\n{self.text}"


@dataclass(slots=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    queries_used: list[str] = field(default_factory=list)
    hypothetical_answer: str = ""
    stages: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def top_confidence(self) -> float:
        return self.chunks[0].confidence if self.chunks else 0.0

    def context_block(self) -> str:
        return "\n\n---\n\n".join(c.as_context() for c in self.chunks)


class Retriever:
    def __init__(
        self,
        store: Store | None = None,
        embedder: EmbeddingProvider | None = None,
        llm: LLM | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or Store(self.settings)
        self._embedder = embedder
        self.llm = llm or LLM(self.settings)

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:      # loading the model is slow; defer it
            self._embedder = get_embedder(self.settings)
        return self._embedder

    # -- stages --------------------------------------------------------

    def rewrite(self, query: str, history: Sequence[dict[str, str]] = ()) -> tuple[list[str], str]:
        if not self.llm.available:
            return [query], ""
        context = ""
        if history:
            turns = "\n".join(f"{h['role']}: {h['content']}" for h in list(history)[-4:])
            context = f"Conversation so far:\n{turns}\n\n"
        try:
            out = self.llm.complete_json(
                model=self.settings.model_rewrite,
                system=REWRITE_SYSTEM,
                messages=[{"role": "user", "content": f"{context}Question: {query}"}],
                schema=REWRITE_SCHEMA,
                max_tokens=512,
            )
        except (LLMUnavailable, Exception):
            # Rewriting is an optimisation. If it fails, search the original
            # question rather than failing the user's request.
            return [query], ""
        queries = [q.strip() for q in out.get("queries", []) if q.strip()] or [query]
        if query not in queries:
            queries.append(query)
        return queries[:3], (out.get("hypothetical_answer") or "").strip()

    def rerank(self, query: str, candidates: list[FusedHit], top_n: int) -> list[RetrievedChunk]:
        chunks = [
            RetrievedChunk(
                chunk_id=c.chunk_id, text=c.hit.text, heading_trail=c.hit.heading_trail,
                source=c.hit.source, url=c.hit.url, fusion_score=c.score, found_by=c.found_by,
            )
            for c in candidates
        ]
        if not self.llm.available or not chunks:
            return chunks[:top_n]

        listing = "\n\n".join(
            f"id: {c.chunk_id}\n{' > '.join(c.heading_trail)}\n{c.text[:1200]}" for c in chunks
        )
        try:
            out = self.llm.complete_json(
                model=self.settings.model_rerank,
                system=RERANK_SYSTEM,
                messages=[{"role": "user",
                           "content": f"Question: {query}\n\nCandidates:\n\n{listing}"}],
                schema=RERANK_SCHEMA,
                max_tokens=2048,
            )
        except Exception:
            return chunks[:top_n]

        scored = {s["id"]: s for s in out.get("scores", []) if "id" in s}
        for c in chunks:
            s = scored.get(c.chunk_id)
            if s:
                c.rerank_score = float(s["score"])
                c.reason = s.get("reason", "")
        # A candidate the model failed to score keeps its fusion position
        # rather than being dropped -- silently losing candidates to a partial
        # response is a much worse failure than a slightly wrong order.
        chunks.sort(key=lambda c: (c.rerank_score is None, -(c.rerank_score or 0), -c.fusion_score))
        return chunks[:top_n]

    # -- pipeline ------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        filters: Filters | None = None,
        history: Sequence[dict[str, str]] = (),
        top_k: int | None = None,
        top_n: int | None = None,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        rrf_k: int = 60,
        weights: Sequence[float] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.settings.retrieval_top_k
        top_n = top_n or self.settings.rerank_top_n
        result = RetrievalResult(query=query, chunks=[])

        queries, hyde = ([query], "")
        if use_rewrite:
            queries, hyde = self.rewrite(query, history)
        result.queries_used = queries
        result.hypothetical_answer = hyde
        if not self.llm.available:
            result.notes.append("no API key: rewriting and reranking skipped")

        rankings: list[list[SearchHit]] = []
        for q in queries:
            rankings.append(self.store.lexical_search(q, k=top_k, filters=filters))
            # HyDE only helps the vector arm -- it is a fake passage, and BM25
            # would just match its invented wording.
            vector_text = f"{q}\n\n{hyde}" if hyde else q
            rankings.append(self.store.vector_search(
                self.embedder.embed_query(vector_text), k=top_k, filters=filters))

        if weights is None:
            weights = [1.0] * len(rankings)
        fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights, limit=top_k)

        result.stages = {
            "lexical": [h.chunk_id for h in rankings[0]],
            "vector": [h.chunk_id for h in rankings[1]] if len(rankings) > 1 else [],
            "fused": [f.chunk_id for f in fused],
        }
        result.chunks = (
            self.rerank(query, fused, top_n) if use_rerank else
            [RetrievedChunk(chunk_id=f.chunk_id, text=f.hit.text,
                            heading_trail=f.hit.heading_trail, source=f.hit.source,
                            url=f.hit.url, fusion_score=f.score, found_by=f.found_by)
             for f in fused[:top_n]]
        )
        result.stages["final"] = [c.chunk_id for c in result.chunks]
        return result
