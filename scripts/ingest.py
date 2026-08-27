#!/usr/bin/env python3
"""CLI: ingest the corpus into chunk records.

    python3 scripts/ingest.py                   # corpus/ -> data/chunks.jsonl
    python3 scripts/ingest.py --inspect 3       # also print 3 sample chunks
    python3 scripts/ingest.py --target 250 --overlap 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingest import Chunker, ingest_corpus  # noqa: E402


def main() -> int:
    cfg = get_settings()   # .env supplies the defaults; flags override them
    ap = argparse.ArgumentParser(description="Parse and chunk the corpus.")
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/chunks.jsonl"))
    ap.add_argument("--target", type=int, default=cfg.chunk_target_words,
                    help="target words per chunk")
    ap.add_argument("--max", dest="max_words", type=int, default=cfg.chunk_max_words)
    ap.add_argument("--min", dest="min_words", type=int, default=60)
    ap.add_argument("--overlap", type=int, default=cfg.chunk_overlap_sentences,
                    help="overlap sentences")
    ap.add_argument("--inspect", type=int, default=0, metavar="N",
                    help="print N sample chunks after ingesting")
    args = ap.parse_args()

    chunker = Chunker(
        target_words=args.target,
        max_words=args.max_words,
        min_words=args.min_words,
        overlap_sentences=args.overlap,
    )

    chunks, stats = ingest_corpus(args.corpus, args.out, chunker=chunker)

    print(stats.report())
    if chunks:
        sizes = sorted(c.word_count for c in chunks)
        print(f"words/chunk      : min={sizes[0]} p50={sizes[len(sizes)//2]} max={sizes[-1]}")
        no_trail = sum(1 for c in chunks if not c.heading_trail)
        print(f"no heading trail : {no_trail} ({no_trail/len(chunks):.0%})  <- lower is better")
    print(f"written          : {args.out}")

    for chunk in chunks[: args.inspect]:
        print("\n" + "=" * 66)
        print(f"{chunk.chunk_id}  [{chunk.word_count}w ~{chunk.token_estimate}tok]")
        print(f"trail : {' > '.join(chunk.heading_trail) or '(none)'}")
        print(f"source: {chunk.source}   url: {chunk.url or '-'}")
        print("-" * 66)
        print(chunk.text[:600] + ("..." if len(chunk.text) > 600 else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
