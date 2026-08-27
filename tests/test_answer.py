"""Citation validation and refusal.

These are the two mechanisms behind the subject's 20% "answer faithfulness",
and both are pure functions over model output -- fully testable with no API
key, which is exactly why the checks live in code rather than in the prompt.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer import (  # noqa: E402
    REFUSAL_TEXT, Answer, AnswerGenerator, link_citations, validate_citations,
)
from app.config import get_settings  # noqa: E402
from app.llm import LLM  # noqa: E402
from app.retrieval import RetrievalResult, RetrievedChunk  # noqa: E402


def chunk(cid: str, conf: float = 0.9) -> RetrievedChunk:
    c = RetrievedChunk(chunk_id=cid, text=f"body of {cid}", heading_trail=["Doc"],
                       source="Policy", url=f"https://x/{cid}", fusion_score=0.5)
    c.rerank_score = conf * 10
    return c


def result(*chunks: RetrievedChunk) -> RetrievalResult:
    return RetrievalResult(query="q", chunks=list(chunks))


# -- citations ----------------------------------------------------------

def test_valid_citation_is_accepted():
    cites = validate_citations("Submit within 30 days [doc-a#c0001].", [chunk("doc-a#c0001")])
    assert len(cites) == 1 and cites[0].valid


def test_fabricated_citation_is_caught():
    """The most damaging failure: an id that looks perfect and refers to
    nothing. Prompting cannot rule it out; parsing can."""
    cites = validate_citations("Policy says X [doc-z#c9999].", [chunk("doc-a#c0001")])
    assert len(cites) == 1 and not cites[0].valid


def test_mixed_valid_and_fabricated():
    text = "A [doc-a#c0001] and B [ghost#c0002]."
    cites = validate_citations(text, [chunk("doc-a#c0001")])
    assert {c.chunk_id: c.valid for c in cites} == {"doc-a#c0001": True, "ghost#c0002": False}


def test_repeated_citation_reported_once():
    cites = validate_citations("X [a#c0001]. Y [a#c0001].", [chunk("a#c0001")])
    assert len(cites) == 1


def test_only_valid_citations_become_links():
    """A fabricated reference must not be dressed up as a clickable source."""
    text = "A [doc-a#c0001] and B [ghost#c0002]."
    cites = validate_citations(text, [chunk("doc-a#c0001")])
    linked = link_citations(text, cites)
    assert "](https://x/doc-a#c0001)" in linked
    assert "[ghost#c0002]" in linked and "](https://x/ghost" not in linked


def test_uncited_claim_is_counted():
    a = Answer(question="q", text="The deadline is thirty days from the expense date.")
    assert a.uncited_claims == 1
    b = Answer(question="q", text="The deadline is thirty days from the expense date [a#c0001].")
    assert b.uncited_claims == 0


def test_refusal_has_no_uncited_claims():
    assert Answer(question="q", text=REFUSAL_TEXT, refused=True).uncited_claims == 0


# -- refusal gates ------------------------------------------------------

def _generator(min_confidence: float = 0.35, key: str = "") -> AnswerGenerator:
    s = replace(get_settings(), min_confidence=min_confidence, anthropic_api_key=key)
    return AnswerGenerator(llm=LLM(s), settings=s)


def test_refuses_when_retrieval_is_empty():
    a = _generator().answer("q", result())
    assert a.refused and "nothing" in a.refusal_reason


def test_refuses_below_the_confidence_threshold_without_calling_the_model():
    """Cheaper and safer than asking a model to be careful with bad context."""
    a = _generator(min_confidence=0.8).answer("q", result(chunk("a#c0001", conf=0.2)))
    assert a.refused and "below threshold" in a.refusal_reason


def test_threshold_of_zero_disables_the_pre_model_gate():
    a = _generator(min_confidence=0.0).answer("q", result(chunk("a#c0001", conf=0.01)))
    assert a.refusal_reason == "no API key configured"  # gate passed, key did not


def test_refusal_text_is_exact():
    """The UI and the eval harness both match on this string."""
    assert _generator().answer("q", result()).text == REFUSAL_TEXT


def test_context_block_is_what_the_model_can_cite():
    r = result(chunk("a#c0001"), chunk("b#c0002"))
    msgs = _generator()._messages("q", r, [])
    context = msgs[-1]["content"][0]["text"]
    assert "[a#c0001]" in context and "[b#c0002]" in context


def test_context_is_cached_and_question_comes_after_it():
    """Prompt caching is a prefix match. Question above context caches nothing."""
    msgs = _generator()._messages("my question", result(chunk("a#c0001")), [])
    blocks = msgs[-1]["content"]
    assert blocks[0].get("cache_control") == {"type": "ephemeral"}
    assert "Question:" in blocks[1]["text"]
    assert "cache_control" not in blocks[1]
