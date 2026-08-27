"""Reciprocal Rank Fusion.

Vector search and BM25 each return a ranked list, and their scores are not
comparable: cosine similarity lives in [0,1] and clusters tightly (0.66 vs
0.50 on this corpus), while BM25 is unbounded and corpus-dependent (4.72 vs
0.98 on the same corpus). Normalising them onto a shared scale means choosing
a normalisation, and every choice is a hyper-parameter that drifts as the
corpus changes.

RRF sidesteps the problem by throwing the scores away and keeping only the
RANKS:

    score(d) = SUM over lists  weight_i / (k + rank_i(d))

A document ranked #1 by either retriever scores 1/61; ranked #2, 1/62. A
document both retrievers like accumulates from both lists and outranks one
that only a single retriever loves. No normalisation, no tuning, and it is
robust to one retriever producing wild scores.

`k = 60` is the value from the original paper (Cormack et al., 2009) and is
what nearly everyone uses. Its role is to flatten the top of the curve: with
k=60 the gap between rank 1 and rank 2 is small, so a single retriever cannot
dominate on the strength of one confident hit. Lower k sharpens that
preference; higher k flattens it further. Tune it against `make eval`, not by
intuition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .store import SearchHit


@dataclass(slots=True)
class FusedHit:
    """A chunk after fusion, carrying where each retriever placed it.

    `contributions` is kept for the eval report and the debug UI: "BM25 had
    this at #1, vector never returned it" is the single most useful line when
    a retrieval failure needs explaining.
    """

    chunk_id: str
    hit: SearchHit
    score: float
    contributions: dict[str, int] = field(default_factory=dict)  # retriever -> rank

    @property
    def found_by(self) -> list[str]:
        return sorted(self.contributions)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """Fuse ranked lists into one.

    `weights` lets one retriever count for more than another -- useful when
    eval shows the corpus is code-heavy (favour lexical) or prose-heavy
    (favour vector). Defaults to equal weighting, which is the honest starting
    point before any measurement exists.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(rankings)} rankings but {len(weights)} weights")
    if k <= 0:
        raise ValueError("k must be positive; k=60 is the standard choice")

    fused: dict[str, FusedHit] = {}
    for ranking, weight in zip(rankings, weights):
        for position, hit in enumerate(ranking, start=1):
            entry = fused.get(hit.chunk_id)
            if entry is None:
                entry = FusedHit(chunk_id=hit.chunk_id, hit=hit, score=0.0)
                fused[hit.chunk_id] = entry
            entry.score += weight / (k + position)
            # Keep the best rank if a retriever somehow returns a duplicate.
            prior = entry.contributions.get(hit.retriever)
            if prior is None or position < prior:
                entry.contributions[hit.retriever] = position

    # Ties broken by chunk_id so the ordering is deterministic -- an eval run
    # that reshuffles ties produces metrics that move for no reason.
    ordered = sorted(fused.values(), key=lambda f: (-f.score, f.chunk_id))
    return ordered[:limit] if limit else ordered
