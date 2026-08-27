"""Notion export parser.

A Notion Markdown export is Markdown -- so we reuse the Markdown parser and
only fix up what Notion does differently:

  * filenames carry a 32-char page id:  "Expense Policy a1b2c3....md"
  * that id is exactly what reconstructs the original Notion URL, which is
    what makes the citation clickable back into the workspace
  * nested pages become nested folders, which gives us a free heading trail
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Document
from .markdown import parse_markdown

_NOTION_ID = re.compile(r"\s+([0-9a-f]{32})$", re.IGNORECASE)


def _split_notion_name(stem: str) -> tuple[str, str | None]:
    m = _NOTION_ID.search(stem)
    if not m:
        return stem, None
    return stem[: m.start()].strip(), m.group(1).lower()


def parse_notion(path: Path, doc_id: str, *, source: str | None = None) -> Document:
    doc = parse_markdown(path, doc_id, source=source)
    name, page_id = _split_notion_name(path.stem)

    if name:
        doc.title = doc.title if doc.title and not _NOTION_ID.search(doc.title) else name
    if page_id and not doc.url:
        doc.url = f"https://www.notion.so/{page_id}"

    # Parent folders in a Notion export are the parent pages. Use them as the
    # document's source label so retrieval filters can scope to a workspace
    # section ("only search the Engineering space").
    parents = [_split_notion_name(p.name)[0] for p in path.parents if p.name]
    parents = [p for p in parents if p and p.lower() not in {"notion", "corpus"}]
    if parents and not source:
        doc.source = f"Notion / {parents[-1]}"
    return doc
