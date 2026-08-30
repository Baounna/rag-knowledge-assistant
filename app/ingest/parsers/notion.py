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

    # Parent folders in a Notion export are the parent pages. The IMMEDIATE
    # parent inside the export is the workspace section -- use that as the
    # source label so retrieval can be scoped to it ("only search the People
    # Group space").
    #
    # Anchor on the `notion` directory rather than walking to the outermost
    # parent: `path.parents` runs from the file outward, so `parents[-1]` is
    # the filesystem root, which produced the nonsense label "Notion / app"
    # from the container's /app working directory.
    if not source:
        parts = [p.name for p in reversed(path.parents) if p.name]
        try:
            anchor = len(parts) - 1 - parts[::-1].index("notion")
        except ValueError:
            anchor = -1
        section = parts[anchor + 1] if anchor >= 0 and anchor + 1 < len(parts) else ""
        section = _split_notion_name(section)[0] if section else ""
        doc.source = f"Notion / {section}" if section else "Notion"
    return doc
