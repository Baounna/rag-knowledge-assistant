"""Parser dispatch: file -> Document.

Adding a connector means adding one entry here. The subject requires at least
three sources; markdown / pdf / notion covers that, and Confluence or Google
Docs slot in the same way (export to Markdown or HTML, add a parser).
"""

from __future__ import annotations

from pathlib import Path

from ..models import Document
from .markdown import parse_markdown
from .notion import parse_notion
from .pdf import parse_pdf

__all__ = ["parse_file", "parse_markdown", "parse_pdf", "parse_notion", "SUPPORTED_SUFFIXES"]

SUPPORTED_SUFFIXES = {".md", ".markdown", ".mdx", ".txt", ".pdf"}


def parse_file(path: Path, doc_id: str, *, source: str | None = None) -> Document:
    """Route a file to the right parser, based on where it lives and its suffix."""
    suffix = path.suffix.lower()

    # A markdown file under corpus/notion/ is a Notion export, not plain md.
    if suffix in {".md", ".markdown", ".mdx", ".txt"}:
        if any(p.name.lower() == "notion" for p in path.parents):
            return parse_notion(path, doc_id, source=source)
        return parse_markdown(path, doc_id, source=source)

    if suffix == ".pdf":
        return parse_pdf(path, doc_id, source=source)

    raise ValueError(f"no parser for {suffix!r} ({path})")
