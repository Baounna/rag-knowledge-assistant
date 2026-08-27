"""PDF parser.

PDFs have no semantic structure -- no <h1>, no paragraph tags, just glyphs at
coordinates. So headings have to be *inferred*. The heuristic below is
deliberately simple and deliberately documented as a heuristic:

    a line is probably a heading if it is short, is not punctuated like a
    sentence, and is followed by a blank line or body text.

This is the weakest link in the ingestion pipeline and the honest place to
say so. If the corpus is heavily PDF-based, upgrade to a layout-aware parser
(PyMuPDF exposes font size and weight; `unstructured` does document-layout
detection) -- font size is a far better heading signal than line length.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Block, Document

_NUMBERED_HEADING = re.compile(r"^\s*(\d+(\.\d+)*)[.)]?\s+\S")
_ENDS_SENTENCE = re.compile(r"[.;:,]\s*$")


def _looks_like_heading(line: str) -> tuple[bool, int]:
    """Return (is_heading, level)."""
    s = line.strip()
    if not s or len(s) > 90:
        return False, 0

    m = _NUMBERED_HEADING.match(s)
    if m:
        # "4.2 Reimbursement Deadlines" -> depth from the dotted number
        return True, min(m.group(1).count(".") + 1, 6)

    if _ENDS_SENTENCE.search(s):
        return False, 0

    words = s.split()
    if len(words) > 12:
        return False, 0

    if s.isupper() and len(words) >= 1:
        return True, 1
    # Title Case with no terminal punctuation
    capitalised = sum(1 for w in words if w[:1].isupper())
    if len(words) >= 2 and capitalised / len(words) >= 0.6:
        return True, 2
    return False, 0


def parse_pdf(path: Path, doc_id: str, *, source: str | None = None) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PDF parsing needs pypdf. Install with:  pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    meta = reader.metadata or {}

    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        text = " ".join(paragraph).strip()
        if text:
            blocks.append(Block(kind="paragraph", text=text))
        paragraph = []

    for page in reader.pages:
        for raw_line in (page.extract_text() or "").splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                continue
            is_heading, level = _looks_like_heading(line)
            if is_heading:
                flush_paragraph()
                blocks.append(Block(kind="heading", text=line, level=level))
            else:
                paragraph.append(line)
        flush_paragraph()

    flush_paragraph()

    title = (meta.get("/Title") or "").strip() or path.stem.replace("_", " ")
    author = (meta.get("/Author") or "").strip() or None
    return Document(
        doc_id=doc_id,
        title=title,
        source=source or title,
        url=path.resolve().as_uri(),
        author=author,
        blocks=blocks,
    )
