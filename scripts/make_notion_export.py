#!/usr/bin/env python3
"""Lay out a few corpus documents as a Notion Markdown export.

A Notion export is Markdown with two conventions the parser depends on:
filenames end in a 32-character page id, and nested pages become nested
folders. Reproducing that shape exercises the real code path -- page-id
stripping and URL reconstruction -- rather than a synthetic file that happens
to be Markdown.

Pure stdlib on purpose: it must run on a host whose Python is missing
compiled extensions.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def page_id(name: str) -> str:
    """Deterministic 32-hex id, so re-running does not churn chunk ids."""
    return hashlib.sha256(name.encode()).hexdigest()[:32]


def title_of(text: str, fallback: str) -> str:
    m = FRONTMATTER.match(text)
    if m:
        for line in m.group(1).splitlines():
            if line.lower().startswith("title:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("corpus/markdown"))
    ap.add_argument("--dest", type=Path, default=Path("corpus/notion"))
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--space", default="People Group")
    ap.add_argument("--move", action="store_true", default=True,
                    help="remove the markdown original so it is not indexed twice")
    args = ap.parse_args()

    if args.dest.exists():
        shutil.rmtree(args.dest)
    space_dir = args.dest / f"{args.space} {page_id(args.space)}"
    space_dir.mkdir(parents=True)

    picks = sorted(args.source.glob("people-group__*.md"))[: args.count]
    if not picks:
        picks = sorted(args.source.glob("*.md"))[: args.count]

    for md in picks:
        text = md.read_text(encoding="utf-8", errors="replace")
        title = title_of(text, md.stem)
        safe = re.sub(r"[^\w\s-]", "", title).strip() or md.stem
        body = FRONTMATTER.sub("", text)
        # A Notion export carries no frontmatter: title comes from the
        # filename and the URL is rebuilt from the page id. Dropping it here
        # is what makes this a real test of the Notion path.
        (space_dir / f"{safe} {page_id(safe)}.md").write_text(
            f"# {title}\n\n{body.lstrip()}", encoding="utf-8")
        if args.move:
            md.unlink()

    print(f"  {len(picks)} pages written to {space_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
