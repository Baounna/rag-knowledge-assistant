"""Chunker invariants.

These tests encode the design rules directly. Each one corresponds to a
failure mode that silently destroys retrieval quality, so a regression here
is far more expensive than it looks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest.chunker import Chunker, split_sentences  # noqa: E402
from app.ingest.models import Block, Document  # noqa: E402


def _doc(*blocks: Block, title: str = "Test Doc") -> Document:
    return Document(doc_id="test", title=title, source=title, blocks=list(blocks))


def _para(text: str) -> Block:
    return Block(kind="paragraph", text=text)


def _heading(text: str, level: int = 1) -> Block:
    return Block(kind="heading", text=text, level=level)


# -- sentence splitting -------------------------------------------------

def test_splits_on_sentence_end():
    assert split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]


def test_does_not_split_on_abbreviation():
    out = split_sentences("Use the portal, e.g. for flights. Then submit.")
    assert out == ["Use the portal, e.g. for flights.", "Then submit."]


def test_does_not_split_on_initials():
    assert split_sentences("Ask J. Smith for approval. He signs off.") == [
        "Ask J. Smith for approval.",
        "He signs off.",
    ]


# -- the core invariant: facts survive ----------------------------------

def test_never_splits_a_sentence_across_chunks():
    """The chunk_17 disaster: '...submitted within' | '30 days...'.

    Every sentence of the source must appear intact inside some chunk.
    """
    sentences = [f"Rule number {i} says you have {i * 3} days to comply." for i in range(60)]
    doc = _doc(_heading("Rules"), _para(" ".join(sentences)))
    chunks = Chunker(target_words=80, max_words=120, min_words=20).chunk(doc)

    assert len(chunks) > 1, "test is meaningless if it all fits in one chunk"
    for sentence in sentences:
        assert any(sentence in c.text for c in chunks), f"lost: {sentence!r}"


def test_never_splits_a_code_block():
    code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(60)) + "\n```"
    doc = _doc(_heading("Config"), _para("Intro."), Block(kind="code", text=code))
    chunks = Chunker(target_words=40, max_words=60, min_words=10).chunk(doc)
    assert any(code in c.text for c in chunks), "code block was split"


# -- self-containment ---------------------------------------------------

def test_every_chunk_has_a_heading_trail():
    doc = _doc(_heading("Policy"), _para("Body text here. More body text."))
    for chunk in Chunker().chunk(doc):
        assert chunk.heading_trail, f"{chunk.chunk_id} has no trail"


def test_trail_is_rooted_at_document_title():
    doc = _doc(_heading("Section", 2), _para("Text. " * 40), title="Handbook")
    for chunk in Chunker(min_words=5).chunk(doc):
        assert chunk.heading_trail[0] == "Handbook"


def test_embedding_text_includes_the_trail():
    """'Set the timeout to 30 seconds' is unfindable without its section."""
    doc = _doc(
        _heading("Payments", 1),
        _heading("Configuration", 2),
        _para("Set the timeout to 30 seconds. " * 12),
        title="Handbook",
    )
    chunk = Chunker(min_words=5).chunk(doc)[0]
    assert "Payments" in chunk.embedding_text()
    assert "Configuration" in chunk.embedding_text()
    assert "Payments" not in chunk.text, "trail must not be baked into the stored text"


def test_merged_chunk_trail_is_honest():
    """A merged chunk must not claim a heading that covers only part of it."""
    doc = _doc(
        _heading("Handbook", 1),
        _heading("Config", 2), _para("Alpha alpha alpha."),
        _heading("Migrations", 2), _para("Beta beta beta."),
        title="Handbook",
    )
    chunks = Chunker(target_words=300, max_words=500, min_words=60).chunk(doc)
    for chunk in chunks:
        if "Alpha" in chunk.text and "Beta" in chunk.text:
            assert "Config" not in chunk.heading_trail
            assert "Migrations" not in chunk.heading_trail


# -- size budget --------------------------------------------------------

def test_respects_max_words_for_prose():
    doc = _doc(_heading("Long"), _para("word " * 3000))
    for chunk in Chunker(target_words=200, max_words=300, min_words=50).chunk(doc):
        assert chunk.word_count <= 300 * 1.2, f"{chunk.chunk_id} = {chunk.word_count}w"


def test_size_is_a_budget_not_a_hard_rule():
    """Chunks land near the target, not exactly on it -- boundaries win."""
    sentences = " ".join(f"Sentence {i} has several words in it." for i in range(200))
    doc = _doc(_heading("S"), _para(sentences))
    sizes = [c.word_count for c in Chunker(target_words=100, max_words=140, min_words=30).chunk(doc)]
    assert len(set(sizes)) > 1, "identical sizes means it cut blindly, not on boundaries"


# -- overlap ------------------------------------------------------------

def test_overlap_repeats_tail_within_a_section():
    sentences = " ".join(f"Fact {i} is recorded here plainly." for i in range(40))
    doc = _doc(_heading("S"), _para(sentences))
    chunks = Chunker(target_words=60, max_words=90, min_words=20, overlap_sentences=2).chunk(doc)
    assert len(chunks) > 1
    prev_tail = split_sentences(chunks[0].text)[-2:]
    assert any(s in chunks[1].text for s in prev_tail), "no overlap carried"


def test_overlap_not_carried_across_a_heading():
    doc = _doc(
        _heading("Alpha", 1), _para("Alpha content. " * 60),
        _heading("Beta", 1), _para("Beta content. " * 60),
    )
    chunks = Chunker(target_words=80, max_words=120, min_words=30).chunk(doc)
    beta = [c for c in chunks if "Beta" in c.heading_trail]
    assert beta, "expected chunks under Beta"
    assert "Alpha content" not in beta[0].text, "overlap leaked across a section boundary"


def test_chunk_ids_are_unique_and_ordered():
    doc = _doc(_heading("S"), _para("Some text here. " * 200))
    chunks = Chunker(target_words=60, max_words=90, min_words=20).chunk(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.position for c in chunks] == list(range(len(chunks)))
