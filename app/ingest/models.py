"""Core data structures for ingestion.

The ingestion pipeline turns raw files into `Chunk` records:

    file  ->  Document (a list of Blocks)  ->  [Chunk, Chunk, ...]

Keeping `Block` in the middle is what preserves document *structure*.
A parser that returns one big string has already destroyed the heading
information, and no amount of clever chunking downstream recovers it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any, Literal

BlockKind = Literal["heading", "paragraph", "code", "table", "list"]


@dataclass(slots=True)
class Block:
    """One structural unit of a parsed document.

    `level` is only meaningful for headings (1 = `#`, 2 = `##`, ...).
    """

    kind: BlockKind
    text: str
    level: int = 0

    @property
    def is_heading(self) -> bool:
        return self.kind == "heading"

    @property
    def is_atomic(self) -> bool:
        """True if this block must never be split across two chunks.

        Splitting a code sample or a table in half produces two chunks that
        are each individually useless — the classic mid-fact cut, but worse,
        because the fragments still *look* well-formed.
        """
        return self.kind in ("code", "table")


@dataclass(slots=True)
class Document:
    """A parsed source file, with its structure intact."""

    doc_id: str
    title: str
    source: str                      # human-readable origin, e.g. "Expense Policy"
    url: str | None = None
    author: str | None = None
    doc_date: date | None = None
    blocks: list[Block] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)


@dataclass(slots=True)
class Chunk:
    """A retrievable piece of a document.

    This is the unit that gets embedded, indexed, retrieved, ranked, and
    cited. Every field exists for a downstream reason:

      chunk_id      the `[source:chunk_id]` the model must cite
      text          what actually goes into the prompt
      heading_trail makes the chunk self-contained ("what is this about?")
      url           makes the citation clickable in the UI
      source/date   power the retrieval filters
      token_estimate lets us budget the prompt before we build it
    """

    chunk_id: str
    doc_id: str
    text: str
    heading_trail: list[str] = field(default_factory=list)
    source: str = ""
    url: str | None = None
    author: str | None = None
    doc_date: date | None = None
    position: int = 0                # index of this chunk within its document
    content_hash: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def token_estimate(self) -> int:
        """Rough token count. ~1.3 tokens per word is close enough for
        budgeting; use the real tokenizer when precision matters."""
        return int(self.word_count * 1.3)

    def embedding_text(self) -> str:
        """What we actually embed / index.

        Deliberately NOT the same as `text`. We prepend the heading trail so
        that a chunk reading 'Set the timeout to 30 seconds.' is indexed as
        'Payments Service > Configuration | Set the timeout to 30 seconds.'
        Without this, the chunk is unfindable for any question that names the
        subsystem rather than quoting the sentence.
        """
        if not self.heading_trail:
            return self.text
        return " > ".join(self.heading_trail) + "\n\n" + self.text

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["doc_date"] = self.doc_date.isoformat() if self.doc_date else None
        return d


def chunk_from_dict(row: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from a JSONL row.

    Exists so the indexer can call `Chunk.embedding_text()` instead of
    re-implementing it. Two copies of "what text gets indexed" drift, and when
    they do, the index silently contains something other than what the chunker
    intended -- with no error anywhere.
    """
    d = dict(row)
    raw_date = d.pop("doc_date", None)
    return Chunk(
        chunk_id=d["chunk_id"],
        doc_id=d["doc_id"],
        text=d["text"],
        heading_trail=list(d.get("heading_trail") or []),
        source=d.get("source") or "",
        url=d.get("url"),
        author=d.get("author"),
        doc_date=date.fromisoformat(raw_date) if isinstance(raw_date, str) else raw_date,
        position=int(d.get("position") or 0),
        content_hash=d.get("content_hash") or "",
    )


def make_content_hash(text: str) -> str:
    """Stable id for chunk text, so re-ingesting an unchanged document does
    not churn the index (and does not re-pay for embeddings)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
