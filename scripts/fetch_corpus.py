#!/usr/bin/env python3
"""Fetch a real corpus: GitLab's public company handbook.

GitLab publishes its entire internal employee handbook openly, which makes it
an unusually good stand-in for "internal company documents nobody can find" --
it is real, it is full of specific rules and numbers, and every answer can be
verified against a public URL.

    python3 scripts/fetch_corpus.py            # ~230 policy documents
    python3 scripts/fetch_corpus.py --sections finance people-group

Two preparation steps matter and are done here rather than by hand:

  * STUB FILTERING. Many handbook pages are one line pointing at an internal
    page ("Please refer to the internal handbook"). Those chunks retrieve for
    their topic and answer nothing, which quietly poisons both retrieval
    metrics and the eval set.

  * URL INJECTION. GitLab's frontmatter carries only a title. Without a `url`
    the citation cannot be clicked, and clickable citations are a graded
    requirement -- so the public handbook URL is reconstructed from each
    file's path and written into its frontmatter.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://gitlab.com/gitlab-com/content-sites/handbook.git"
PUBLIC_BASE = "https://handbook.gitlab.com/handbook"

# Sections an employee actually asks about. Deliberately NOT the whole
# handbook: 4,151 files cannot be verified by hand, and an eval set you cannot
# verify is worse than no eval set.
DEFAULT_SECTIONS = [
    "finance",         # expenses, travel, procurement -- dense with numbers
    "people-group",    # leave, conduct, contracts
    "total-rewards",   # benefits, compensation
    "hiring",
    "communication",
    "legal/privacy",
]

MIN_WORDS = 150        # below this a page is a stub or a link redirect
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def clone(workdir: Path) -> Path:
    checkout = workdir / "handbook"
    if (checkout / "content" / "handbook").exists():
        print(f"  reusing existing checkout at {checkout}")
        return checkout / "content" / "handbook"
    checkout.parent.mkdir(parents=True, exist_ok=True)
    print("  cloning (shallow, sparse -- a full clone is ~1GB)…")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(checkout)],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", "content/handbook"],
                   cwd=checkout, check=True, capture_output=True)
    return checkout / "content" / "handbook"


def public_url(rel: Path) -> str:
    path = rel.as_posix().removesuffix(".md").removesuffix("/_index")
    return f"{PUBLIC_BASE}/{path}/"


def section_label(rel: Path) -> str:
    """Human-readable source name, used for the retrieval source filter."""
    parts = rel.parts
    top = parts[0].replace("-", " ").title() if parts else "Handbook"
    return f"GitLab Handbook / {top}"


def prepare(src_root: Path, dest: Path, sections: list[str]) -> tuple[int, int]:
    dest.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0

    for section in sections:
        src_section = src_root / section
        if not src_section.exists():
            print(f"  ! section not found, skipping: {section}")
            continue
        for path in sorted(src_section.rglob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            body = FRONTMATTER.sub("", text)
            if len(body.split()) < MIN_WORDS:
                skipped += 1
                continue

            rel = path.relative_to(src_root)
            match = FRONTMATTER.match(text)
            meta = match.group(1) if match else ""
            title = "Untitled"
            for line in meta.splitlines():
                if line.strip().lower().startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip("\"'")

            # Rewrite the frontmatter with the fields the ingester reads.
            header = (
                "---\n"
                f'title: "{title}"\n'
                f'source: "{section_label(rel)}"\n'
                f"url: {public_url(rel)}\n"
                'author: "GitLab"\n'
                "---\n\n"
            )
            out = dest / rel.as_posix().replace("/", "__")
            out.write_text(header + body.lstrip(), encoding="utf-8")
            kept += 1

    return kept, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="*", default=DEFAULT_SECTIONS)
    ap.add_argument("--dest", type=Path, default=Path("corpus/markdown"))
    ap.add_argument("--workdir", type=Path, default=Path(".corpus-cache"))
    ap.add_argument("--clean", action="store_true",
                    help="empty the destination first (removes the sample documents)")
    args = ap.parse_args()

    print("Fetching GitLab's public handbook")
    src = clone(args.workdir)

    if args.clean and args.dest.exists():
        shutil.rmtree(args.dest)
        print(f"  cleared {args.dest}")

    kept, skipped = prepare(src, args.dest, args.sections)
    print(f"\n  documents written : {kept}  -> {args.dest}")
    print(f"  stubs skipped     : {skipped}  (under {MIN_WORDS} words: link redirects)")
    print("\nnext: make ingest && make index")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
