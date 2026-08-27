#!/usr/bin/env python3
"""CLI: load chunk records into Postgres, embedding as we go.

    python3 scripts/index.py                 # incremental (skips unchanged)
    python3 scripts/index.py --recreate      # drop and rebuild the table
    python3 scripts/index.py --probe "how many days to submit expenses?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.ingest import read_jsonl  # noqa: E402
from app.store import Store  # noqa: E402


def _index_text(row: dict) -> str:
    """Mirror of Chunk.embedding_text(): heading trail + body."""
    trail = row.get("heading_trail") or []
    return (" > ".join(trail) + "\n\n" + row["text"]) if trail else row["text"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the table")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--probe", type=str, default="", help="run a test search afterwards")
    args = ap.parse_args()

    settings = get_settings()
    store = Store(settings)
    backend = store.init_schema(drop=args.recreate)
    print(f"lexical backend  : {backend}"
          f"{'  (pg_search/Tantivy)' if backend == 'bm25' else '  (postgres FTS fallback)'}")
    print(f"embeddings       : {settings.embedding_provider}/{settings.embedding_model} "
          f"({settings.embedding_dim}d)")

    rows = read_jsonl(args.chunks)
    if not rows:
        print(f"no chunks in {args.chunks} -- run `make ingest` first")
        return 1

    # Incremental: only embed chunks whose text actually changed. Embedding is
    # the expensive step, and most re-ingests change a handful of documents.
    known = {} if args.recreate else store.existing_hashes()
    todo = [r for r in rows if known.get(r["chunk_id"]) != r["content_hash"]]
    print(f"chunks           : {len(rows)} total, {len(todo)} new or changed")

    if not todo:
        print(f"index up to date : {store.count()} chunks")
        return 0

    embedder = get_embedder(settings)
    started = time.time()
    written = 0

    for i in range(0, len(todo), args.batch):
        batch = todo[i : i + args.batch]
        vectors = embedder.embed_documents([_index_text(r) for r in batch])
        payload = [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "text": r["text"],
                "index_text": _index_text(r),
                "heading_trail": r.get("heading_trail") or [],
                "source": r.get("source") or "",
                "url": r.get("url"),
                "author": r.get("author"),
                "doc_date": r.get("doc_date"),
                "position": r.get("position", 0),
                "content_hash": r["content_hash"],
                "embedding": str(v),
            }
            for r, v in zip(batch, vectors)
        ]
        written += store.upsert_chunks(payload)
        print(f"  embedded {written}/{len(todo)}", end="\r", flush=True)

    elapsed = time.time() - started
    print(f"\nindexed          : {written} chunks in {elapsed:.1f}s "
          f"({written / max(elapsed, 0.01):.1f}/s)")
    print(f"table total      : {store.count()} chunks")

    if args.probe:
        print(f"\nprobe: {args.probe!r}")
        qv = embedder.embed_query(args.probe)
        for label, hits in (
            ("vector ", store.vector_search(qv, k=3)),
            ("lexical", store.lexical_search(args.probe, k=3)),
        ):
            print(f"\n  {label} top 3:")
            for h in hits:
                trail = " > ".join(h.heading_trail)
                print(f"    {h.rank}. [{h.score:6.3f}] {h.chunk_id}")
                print(f"       {trail}")
                print(f"       {h.text[:95].replace(chr(10), ' ')}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
