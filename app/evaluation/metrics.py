"""Metrics.

Split deliberately into two groups:

  DETERMINISTIC  recall@K, MRR, precision@K, citation validity, refusal
                 accuracy, required-content coverage. Pure functions over
                 labels and output. No API key, no cost, no variance -- run
                 them on every commit.

  JUDGED         faithfulness and answer relevance. Need an LLM, cost money,
                 and vary between runs.

Prefer the deterministic ones wherever they can answer the question. "Did the
model cite a chunk that does not exist?" is a parsing problem, not a judgement
call, and treating it as one makes it free and exact. The judge is reserved
for what genuinely needs reading comprehension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Sequence

from ..answer import Answer
from ..config import Settings, get_settings
from ..llm import LLM
from .dataset import EvalQuestion

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Share of the relevant chunks that made it into the top K.

    The metric that decides whether generation is even possible: a chunk that
    was never retrieved cannot be cited, no matter how good the model is."""
    if not relevant:
        return float("nan")          # undefined, not zero -- excluded from means
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(retrieved[:k])


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1/rank of the first relevant chunk.

    Rewards putting the answer at #1 rather than merely somewhere in the list
    -- which matters because only the top N reach the prompt, and because a
    reranker's whole job is ordering, invisible to recall."""
    relevant_set = set(relevant)
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / position
    return 0.0


# ---------------------------------------------------------------------------
# Answer metrics (deterministic)
# ---------------------------------------------------------------------------


def citation_validity(answer: Answer) -> float:
    """Share of cited ids that were actually retrieved. 1.0 or it is a bug."""
    if not answer.citations:
        return float("nan")
    return sum(1 for c in answer.citations if c.valid) / len(answer.citations)


def is_grounded(answer: Answer) -> bool:
    """Cites at least one real chunk and fabricates none."""
    if answer.refused:
        return True
    return bool(answer.citations) and not answer.invalid_citations


def refusal_correct(question: EvalQuestion, answer: Answer) -> bool:
    """Refusing an unanswerable question is correct; refusing an answerable
    one is a miss. Both directions matter, so this is scored on all questions."""
    return answer.refused if not question.answerable else not answer.refused


def required_content_coverage(question: EvalQuestion, answer: Answer) -> float:
    """Share of `must_include` substrings present. Catches the answer that is
    fluent, well-cited, and omits the number the user asked for."""
    if not question.must_include:
        return float("nan")
    text = answer.text.lower()
    return sum(1 for s in question.must_include if s.lower() in text) / len(question.must_include)


# ---------------------------------------------------------------------------
# Judged metrics
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You grade an answer produced by a retrieval system against \
the excerpts it was given.

faithfulness (0-10): is every claim supported by the excerpts?
  10  every claim traceable to an excerpt
  5   mostly supported, one claim goes beyond what the excerpts say
  0   contradicts the excerpts, or asserts facts absent from them

A claim that is TRUE IN GENERAL but absent from the excerpts scores low. This \
is the point of the metric: the system must answer from the corpus, not from \
world knowledge, and an answer that happens to be right for the wrong reason \
will be wrong the moment the corpus disagrees with the world.

relevance (0-10): does it answer the question that was asked?
  10  answers directly and completely
  5   partially answers, or answers a neighbouring question
  0   does not address the question

Grade these independently -- an unfaithful answer can be perfectly relevant, \
and a faithful one can miss the question. A correct refusal on a question the \
excerpts do not cover is faithfulness 10; its relevance depends on whether \
refusing was the right response."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number", "minimum": 0, "maximum": 10},
        "relevance": {"type": "number", "minimum": 0, "maximum": 10},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "comment": {"type": "string"},
    },
    "required": ["faithfulness", "relevance", "unsupported_claims", "comment"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class Judgement:
    faithfulness: float = float("nan")
    relevance: float = float("nan")
    unsupported_claims: list[str] = field(default_factory=list)
    comment: str = ""
    judged: bool = False


def judge_answer(
    question: EvalQuestion, answer: Answer, llm: LLM, settings: Settings | None = None
) -> Judgement:
    settings = settings or get_settings()
    if not llm.available:
        return Judgement()
    context = "\n\n---\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in answer.chunks_used)
    try:
        out = llm.complete_json(
            model=llm.model_for("judge"),
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": (
                f"Question: {question.question}\n\n"
                f"Excerpts given to the system:\n\n{context}\n\n"
                f"Answer produced:\n{answer.text}"
            )}],
            schema=JUDGE_SCHEMA,
            max_tokens=800,
        )
    except Exception as exc:  # noqa: BLE001
        return Judgement(comment=f"judge failed: {type(exc).__name__}")
    return Judgement(
        faithfulness=float(out["faithfulness"]),
        relevance=float(out["relevance"]),
        unsupported_claims=list(out.get("unsupported_claims") or []),
        comment=out.get("comment", ""),
        judged=True,
    )


def nanmean(values: Sequence[float]) -> float:
    """Mean over defined values only.

    Metrics that are undefined for a question (recall on an unanswerable one,
    citation validity when nothing was cited) return NaN and must be excluded
    rather than counted as zero -- counting them as zero silently penalises
    correct behaviour.
    """
    defined = [v for v in values if v == v]
    return mean(defined) if defined else float("nan")
