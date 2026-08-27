"""RRF invariants.

Fusion is pure arithmetic over ranks, so it is fully testable without a
database, an API key, or an embedding model -- which is exactly why it is
worth testing hard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.fusion import reciprocal_rank_fusion  # noqa: E402
from app.store import SearchHit  # noqa: E402


def hit(chunk_id: str, rank: int, retriever: str, score: float = 1.0) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id, text=f"text of {chunk_id}", heading_trail=[],
        source="src", url=None, doc_date=None, score=score, rank=rank,
        retriever=retriever,
    )


def ranking(retriever: str, *ids: str) -> list[SearchHit]:
    return [hit(cid, i + 1, retriever) for i, cid in enumerate(ids)]


def test_agreement_beats_a_single_strong_hit():
    """The core promise of RRF.

    'b' is #2 in both lists; 'a' is #1 in one and absent from the other.
    Agreement across retrievers should win -- that is the entire reason for
    fusing rather than picking one retriever.
    """
    vector = ranking("vector", "a", "b", "c")
    lexical = ranking("lexical", "z", "b", "y")
    fused = reciprocal_rank_fusion([vector, lexical])
    assert fused[0].chunk_id == "b"
    assert fused[0].found_by == ["lexical", "vector"]


def test_scores_are_rank_based_not_score_based():
    """BM25 scores are unbounded and cosine scores are not; fusion must ignore
    both. A retriever reporting score=999 must not outrank agreement."""
    vector = [hit("a", 1, "vector", score=0.51)]
    lexical = [hit("b", 1, "lexical", score=999.0), hit("a", 2, "lexical", score=0.01)]
    fused = reciprocal_rank_fusion([vector, lexical])
    assert fused[0].chunk_id == "a", "fusion leaked raw scores"


def test_a_chunk_only_one_retriever_found_still_ranks():
    """Vector-only hits are the whole point on a paraphrased question, and
    lexical-only hits are the point on an exact identifier. Neither may be
    dropped."""
    fused = reciprocal_rank_fusion([ranking("vector", "v1"), ranking("lexical", "l1")])
    assert {f.chunk_id for f in fused} == {"v1", "l1"}


def test_rank_one_scores_more_than_rank_two():
    fused = reciprocal_rank_fusion([ranking("vector", "first", "second")])
    assert fused[0].chunk_id == "first"
    assert fused[0].score > fused[1].score


def test_k_controls_how_much_rank_one_dominates():
    """Small k sharpens the preference for a single confident #1; large k
    flattens the curve so broad agreement wins instead. This is the knob to
    tune against eval, so its direction must be right.

    The contest has to be between DIFFERENT documents for k to matter: 'solo'
    is #1 in one list only, 'both' is #4 in each. A document appearing in both
    lists at the same rank wins at every k, which is why the first version of
    this test proved nothing.
    """
    v = ranking("vector", "solo", "x", "y", "both")
    l = ranking("lexical", "p", "q", "r", "both")
    def score(fused, chunk_id):
        return next(f.score for f in fused if f.chunk_id == chunk_id)

    sharp = reciprocal_rank_fusion([v, l], k=1)
    flat = reciprocal_rank_fusion([v, l], k=1000)
    # Compare the two contenders directly. Asserting on the global #1 is wrong
    # here: at k=1 'solo' ties with the other list's #1 and loses the
    # alphabetical tie-break, which says nothing about k.
    assert score(sharp, "solo") > score(sharp, "both"), "tiny k should favour a confident #1"
    assert score(flat, "both") > score(flat, "solo"), "huge k should favour agreement"


def test_weights_shift_the_balance():
    v, l = ranking("vector", "v"), ranking("lexical", "x")
    assert reciprocal_rank_fusion([v, l], weights=[5.0, 1.0])[0].chunk_id == "v"
    assert reciprocal_rank_fusion([v, l], weights=[1.0, 5.0])[0].chunk_id == "x"


def test_contributions_record_each_retrievers_rank():
    fused = reciprocal_rank_fusion(
        [ranking("vector", "x", "target"), ranking("lexical", "target")]
    )
    target = next(f for f in fused if f.chunk_id == "target")
    assert target.contributions == {"vector": 2, "lexical": 1}


def test_ties_are_deterministic():
    """Identical inputs must produce identical order, or eval metrics move
    between runs for no reason."""
    a = reciprocal_rank_fusion([ranking("vector", "p", "q"), ranking("lexical", "q", "p")])
    b = reciprocal_rank_fusion([ranking("vector", "p", "q"), ranking("lexical", "q", "p")])
    assert [f.chunk_id for f in a] == [f.chunk_id for f in b]


def test_limit_truncates():
    fused = reciprocal_rank_fusion([ranking("vector", *"abcdef")], limit=3)
    assert len(fused) == 3


def test_empty_and_degenerate_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert len(reciprocal_rank_fusion([ranking("vector", "solo"), []])) == 1


def test_rejects_mismatched_weights_and_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([ranking("vector", "a")], weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([ranking("vector", "a")], k=0)
