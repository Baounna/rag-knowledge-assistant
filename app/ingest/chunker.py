"""Recursive, structure-aware chunker.

Cut-point priority (this ordering IS the chunking strategy):

  1. Never split an atomic block (code, table). Half a code sample is not a
     smaller answer, it is a wrong answer that still looks well-formed.
  2. Cut at a heading boundary   -- always taken; sections are the natural
     unit of self-containment.
  3. Then at a paragraph boundary.
  4. Then at a sentence boundary.
  5. Never mid-sentence.

Size is a *budget*, not a rule: whole sentences are added until the chunk
approaches `target_words`, then it closes at the last complete sentence. A
chunk may come out at 310 or 480 words; the boundary always wins over the
number.

Undersized chunks are repaired afterwards by `_merge_undersized`, never by
skipping a heading cut. Skipping the cut is how a chunk ends up carrying a
heading trail that lies about half its own content.
"""

from __future__ import annotations

import re

from .models import Chunk, Document, make_content_hash

# Abbreviations ending in '.' that do not end a sentence. Swap for spaCy's
# sentencizer if the corpus is prose-heavy; this covers the common cases
# without a 500MB dependency.
_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.", "no.", "fig.",
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "inc.", "ltd.", "co.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.",
    "sept.", "oct.", "nov.", "dec.",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminal punctuation attached."""
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    for piece in _SENTENCE_END.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if sentences:
            prev_words = sentences[-1].split()
            last = prev_words[-1].lower() if prev_words else ""
            # "e.g." / "Dr." / a single initial "J." do not end a sentence.
            if last in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last):
                sentences[-1] += " " + piece
                continue
        sentences.append(piece)
    return sentences


def _words(text: str) -> int:
    return len(text.split())


def _hard_split(sentence: str, limit: int) -> list[str]:
    """Last-resort level-4 cut: break on word boundaries.

    Only reached when a single 'sentence' exceeds the whole chunk budget --
    which happens constantly with PDF extraction, unpunctuated lists, and
    flattened tables. Without this fallback, one malformed paragraph produces
    one enormous chunk that blows the prompt budget and retrieves for
    everything. Ugly, but bounded, and it is the documented last resort.
    """
    words = sentence.split()
    if len(words) <= limit:
        return [sentence]
    return [" ".join(words[i:i + limit]) for i in range(0, len(words), limit)]


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


class Chunker:
    def __init__(
        self,
        target_words: int = 350,
        max_words: int = 500,
        min_words: int = 60,
        overlap_sentences: int = 2,
    ) -> None:
        if not 0 < min_words < target_words <= max_words:
            raise ValueError("expected 0 < min_words < target_words <= max_words")
        self.target_words = target_words
        self.max_words = max_words
        self.min_words = min_words
        self.overlap_sentences = overlap_sentences
        # Overlap is context, not content: hard-cap it at a quarter of the
        # target so it can never dominate (or grow) a chunk.
        self.overlap_budget = max(1, int(target_words * 0.25))

    # -- public API ----------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        raw = self._split(doc)
        merged = self._merge_undersized(raw)
        return self._finalise(doc, merged)

    # -- stage 1: cut --------------------------------------------------

    def _split(self, doc: Document) -> list[tuple[list[str], str]]:
        """Produce (heading_trail, text) pairs, respecting the cut priority."""
        out: list[tuple[list[str], str]] = []
        # Buffer segments are (group_id, text). Segments sharing a group are
        # sentences of one paragraph and rejoin with a space; different groups
        # are separate blocks and rejoin with a blank line. Without this the
        # chunk text comes back with every sentence on its own paragraph.
        buffer: list[tuple[int, str]] = []
        buffer_words = 0
        group = 0
        stack: list[tuple[int, str]] = []
        root = [doc.title] if doc.title else []
        trail: list[str] = list(root)

        def render(segments: list[tuple[int, str]]) -> str:
            grouped: list[list[str]] = []
            last_gid: int | None = None
            for gid, text in segments:
                if gid == last_gid and grouped:
                    grouped[-1].append(text)
                else:
                    grouped.append([text])
                last_gid = gid
            return "\n\n".join(" ".join(g).strip() for g in grouped).strip()

        def flush(carry_overlap: bool) -> None:
            nonlocal buffer, buffer_words, group
            text = render(buffer)
            if text:
                out.append((list(trail), text))
            buffer = []
            buffer_words = 0
            # Overlap replays the tail of this chunk into the next so dangling
            # references ("this window", "they") keep their referent. Never
            # carried across a heading: a new section is a new context, and
            # dragging the old one in is noise, not help.
            if carry_overlap and self.overlap_sentences > 0 and text:
                tail = split_sentences(text)[-self.overlap_sentences:]
                joined = " ".join(tail).strip()
                # Cap the overlap. Without a cap, a chunk made of one very
                # long unpunctuated "sentence" replays itself in full into the
                # next buffer and every chunk grows by a full chunk's worth.
                words = joined.split()
                if len(words) > self.overlap_budget:
                    joined = " ".join(words[-self.overlap_budget:])
                if joined:
                    group += 1
                    buffer = [(group, joined)]
                    buffer_words = _words(joined)

        for block in doc.blocks:
            if block.is_heading:
                flush(carry_overlap=False)          # headings are always taken
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text.strip()))
                trail = root + [t for _, t in stack]
                continue

            if block.is_atomic:
                w = _words(block.text)
                if buffer and buffer_words + w > self.max_words:
                    flush(carry_overlap=True)
                group += 1
                buffer.append((group, block.text))
                buffer_words += w
                if buffer_words >= self.target_words:
                    flush(carry_overlap=True)
                continue

            group += 1
            sentences = [
                piece
                for sentence in split_sentences(block.text)
                for piece in _hard_split(sentence, self.max_words)
            ]
            for sentence in sentences:
                w = _words(sentence)
                if buffer_words + w > self.max_words and buffer_words > 0:
                    flush(carry_overlap=True)
                    group += 1
                buffer.append((group, sentence))
                buffer_words += w

            if buffer_words >= self.target_words:
                flush(carry_overlap=True)

        flush(carry_overlap=False)
        return out

    # -- stage 2: repair -----------------------------------------------

    def _merge_undersized(
        self, pieces: list[tuple[list[str], str]]
    ) -> list[tuple[list[str], str]]:
        """Merge chunks below `min_words` into a neighbour.

        Only merges pieces that are siblings under a shared parent heading (or
        parent/child), and sets the merged trail to the COMMON PREFIX -- so a
        merged chunk never claims a heading that describes only part of it.
        """
        out: list[tuple[list[str], str]] = []
        for trail, text in pieces:
            if out:
                prev_trail, prev_text = out[-1]
                small = _words(prev_text) < self.min_words or _words(text) < self.min_words
                common = _common_prefix(prev_trail, trail)
                related = len(common) >= min(len(prev_trail), len(trail)) - 1 and bool(common)
                fits = _words(prev_text) + _words(text) <= self.max_words
                if small and related and fits:
                    out[-1] = (common, prev_text + "\n\n" + text)
                    continue
            out.append((trail, text))
        return out

    # -- stage 3: emit -------------------------------------------------

    def _finalise(self, doc: Document, pieces: list[tuple[list[str], str]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for i, (trail, text) in enumerate(pieces):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#c{len(chunks):04d}",
                    doc_id=doc.doc_id,
                    text=text,
                    heading_trail=trail,
                    source=doc.source,
                    url=doc.url,
                    author=doc.author,
                    doc_date=doc.doc_date,
                    position=len(chunks),
                    content_hash=make_content_hash(text),
                )
            )
        return chunks


def chunk_document(doc: Document, **kwargs) -> list[Chunk]:
    return Chunker(**kwargs).chunk(doc)
