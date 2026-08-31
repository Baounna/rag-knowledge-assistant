"""Metric correctness.

A wrong metric is worse than no metric: it produces confident numbers that
justify the wrong decisions. These are pure functions, so there is no excuse
for not pinning them down.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer import Answer, Citation  # noqa: E402
from app.evaluation import (  # noqa: E402
    Config, EvalQuestion, citation_validity, coverage, hit_at_k, is_grounded,
    load_questions, nanmean, precision_at_k, recall_at_k, reciprocal_rank,
    refusal_correct, required_content_coverage, save_questions,
)


# -- retrieval metrics --------------------------------------------------

def test_recall_counts_relevant_chunks_inside_k():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == pytest.approx(0.5)
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == pytest.approx(1.0)
    assert recall_at_k(["x", "y", "a"], ["a"], k=2) == pytest.approx(0.0)


def test_recall_is_undefined_not_zero_without_labels():
    """Unanswerable questions have no relevant chunks. Scoring them 0 would
    silently punish the system for correctly having nothing to retrieve."""
    assert math.isnan(recall_at_k(["a"], [], k=3))


def test_mrr_rewards_rank_one():
    assert reciprocal_rank(["hit", "x"], ["hit"]) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "hit"], ["hit"]) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y", "hit"], ["hit"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], ["hit"]) == 0.0


def test_mrr_distinguishes_ordering_where_recall_cannot():
    """The reranker only changes ORDER, so recall cannot see its effect.
    If MRR could not either, the reranker would be unmeasurable."""
    good, bad = ["hit", "x", "y"], ["x", "y", "hit"]
    assert recall_at_k(good, ["hit"], 3) == recall_at_k(bad, ["hit"], 3)
    assert reciprocal_rank(good, ["hit"]) > reciprocal_rank(bad, ["hit"])


def test_precision_and_hit():
    assert precision_at_k(["a", "x"], ["a"], k=2) == pytest.approx(0.5)
    assert hit_at_k(["x", "a"], ["a"], k=2) == 1.0
    assert hit_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_nanmean_excludes_undefined_values():
    assert nanmean([1.0, float("nan"), 0.0]) == pytest.approx(0.5)
    assert math.isnan(nanmean([float("nan")]))


# -- answer metrics -----------------------------------------------------

def _answer(text="x", citations=(), refused=False) -> Answer:
    return Answer(question="q", text=text, citations=list(citations), refused=refused)


def test_citation_validity_penalises_fabrication():
    good = Citation("a#c0001", "S", "u", True)
    bad = Citation("ghost#c0001", "", None, False)
    assert citation_validity(_answer(citations=[good])) == pytest.approx(1.0)
    assert citation_validity(_answer(citations=[good, bad])) == pytest.approx(0.5)


def test_grounded_requires_a_real_citation():
    assert not is_grounded(_answer(citations=[]))
    assert not is_grounded(_answer(citations=[Citation("g#c0001", "", None, False)]))
    assert is_grounded(_answer(citations=[Citation("a#c0001", "S", "u", True)]))
    assert is_grounded(_answer(refused=True)), "a refusal cites nothing and is still grounded"


def test_refusal_scored_in_both_directions():
    answerable = EvalQuestion(id="a", question="q", relevant_chunk_ids=["a#c0001"])
    unanswerable = EvalQuestion(id="b", question="q", answerable=False)
    assert refusal_correct(answerable, _answer(refused=False))
    assert not refusal_correct(answerable, _answer(refused=True)), "over-refusal is a failure too"
    assert refusal_correct(unanswerable, _answer(refused=True))
    assert not refusal_correct(unanswerable, _answer(refused=False))


def test_required_content_catches_the_missing_number():
    q = EvalQuestion(id="a", question="q", relevant_chunk_ids=["c#c0001"], must_include=["30"])
    assert required_content_coverage(q, _answer("Submit within 30 days.")) == pytest.approx(1.0)
    assert required_content_coverage(q, _answer("Submit promptly.")) == pytest.approx(0.0)


# -- dataset ------------------------------------------------------------

def test_rejects_answerable_questions_with_no_labels(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    save_questions([EvalQuestion(id="q1", question="?", relevant_chunk_ids=[])], path)
    with pytest.raises(ValueError, match="unlabelled"):
        load_questions(path)


def test_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    save_questions([
        EvalQuestion(id="q1", question="a", relevant_chunk_ids=["c#c0001"]),
        EvalQuestion(id="q1", question="b", relevant_chunk_ids=["c#c0002"]),
    ], path)
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


def test_coverage_flags_labels_pointing_at_missing_chunks():
    """Re-chunking the corpus silently invalidates every label. The harness
    must notice rather than report metrics against ghosts."""
    qs = [EvalQuestion(id="q1", question="?", relevant_chunk_ids=["old#c0001"])]
    assert coverage(qs, {"new#c0001"})["stale_labels"] == ["old#c0001"]


def test_shipped_eval_set_is_valid_and_exercises_refusal():
    qs = load_questions("eval/questions.jsonl")
    assert len(qs) >= 20
    assert any(not q.answerable for q in qs), "no unanswerable questions -- refusal unmeasured"


# -- configs ------------------------------------------------------------

def test_isolating_one_retriever_zero_weights_the_other():
    """Config differences must be attributable to the retriever, so the rest
    of the pipeline stays byte-identical."""
    assert Config("v", use_lexical=False).weights == [0.0, 1.0]
    assert Config("l", use_vector=False).weights == [1.0, 0.0]
    assert Config("hybrid").weights is None


def test_answer_metrics_are_not_gated_on_one_provider():
    """Regression: `make eval-full` checked `settings.anthropic_api_key` and so
    silently skipped generation whenever the local model was in use -- exiting
    0 with retrieval numbers only, as if the answer metrics were simply empty.

    The availability check must go through the LLM, which knows about every
    backend, not through one provider's config field.
    """
    import inspect
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent.joinpath("scripts/eval.py").read_text()
    gate = [l for l in source.splitlines() if "skipping generation" in l]
    assert gate, "the generation gate disappeared -- did the check move?"
    window = source[: source.index(gate[0])].splitlines()[-4:]
    assert any("LLM(settings).available" in l for l in window), (
        "the gate must ask the LLM whether a model exists, not check one "
        "provider's API key"
    )
