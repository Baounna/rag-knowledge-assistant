"""Markdown parser that preserves structure.

The job here is NOT to produce text. It is to produce *labelled* text: which
lines were headings, which were code, which were tables. That labelling is
what lets the chunker cut at meaningful boundaries and attach heading trails.
A parser that returns `f.read()` has already lost the game.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from ..models import Block, Document

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal YAML frontmatter reader (flat `key: value` pairs only)."""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip().strip("'\"")
    return meta, text[m.end():]


def _coerce_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def parse_markdown(path: Path, doc_id: str, *, source: str | None = None) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _parse_frontmatter(raw)

    blocks: list[Block] = []
    pending: list[str] = []
    pending_kind: str = "paragraph"

    def flush() -> None:
        nonlocal pending, pending_kind
        text = "\n".join(pending).strip()
        if text:
            blocks.append(Block(kind=pending_kind, text=text))  # type: ignore[arg-type]
        pending = []
        pending_kind = "paragraph"

    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Fenced code: consume verbatim to the closing fence. Atomic.
        if _FENCE.match(line):
            flush()
            fence = _FENCE.match(line).group(1)  # type: ignore[union-attr]
            code = [line]
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(fence):
                code.append(lines[i])
                i += 1
            if i < len(lines):
                code.append(lines[i])
            blocks.append(Block(kind="code", text="\n".join(code)))
            i += 1
            continue

        heading = _ATX_HEADING.match(line)
        if heading:
            flush()
            blocks.append(
                Block(kind="heading", text=heading.group(2), level=len(heading.group(1)))
            )
            i += 1
            continue

        # Setext heading ("Title" underlined with === or ---)
        if (
            line.strip()
            and i + 1 < len(lines)
            and re.fullmatch(r"\s*(=+|-+)\s*", lines[i + 1])
            and not _LIST_ITEM.match(line)
        ):
            flush()
            level = 1 if "=" in lines[i + 1] else 2
            blocks.append(Block(kind="heading", text=line.strip(), level=level))
            i += 2
            continue

        if _TABLE_ROW.match(line):
            if pending_kind != "table":
                flush()
                pending_kind = "table"
            pending.append(line)
            i += 1
            continue

        if not line.strip():
            flush()
            i += 1
            continue

        if _LIST_ITEM.match(line):
            if pending_kind not in ("list", "paragraph"):
                flush()
            pending_kind = "list"
        pending.append(line)
        i += 1

    flush()

    title = meta.get("title") or _first_heading(blocks) or path.stem.replace("-", " ").replace("_", " ").title()
    return Document(
        doc_id=doc_id,
        title=title,
        source=source or meta.get("source") or title,
        url=meta.get("url"),
        author=meta.get("author"),
        doc_date=_coerce_date(meta.get("date") or meta.get("updated")),
        blocks=blocks,
    )


def _first_heading(blocks: list[Block]) -> str | None:
    for b in blocks:
        if b.is_heading:
            return b.text
    return None
