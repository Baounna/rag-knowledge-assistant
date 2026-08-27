#!/usr/bin/env python3
"""Interactive retrieval demo.

    make demo

Type a question, see what each index returns and how they disagree. This is
the honest view of the system as it stands: retrieval works, generation does
not exist yet. Watching the two lists diverge is the best intuition-builder
for why hybrid fusion (slice 3) is needed at all.

Commands:
    :k 10                 how many hits to show
    :source Finance       filter to sources containing this text
    :nosource             clear the source filter
    :stats                what is in the index
    :q                    quit
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.retrieval import Retriever  # noqa: E402
from app.store import Filters, SearchHit, Store  # noqa: E402

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
OFF = "\033[0m"


def snippet(hit: SearchHit, width: int = 88) -> str:
    text = " ".join(hit.text.split())
    return text[:width] + ("…" if len(text) > width else "")


def show(label: str, colour: str, hits: list[SearchHit], other: list[SearchHit]) -> None:
    other_ids = {h.chunk_id for h in other}
    print(f"\n  {colour}{BOLD}{label}{OFF}")
    if not hits:
        print(f"    {DIM}(nothing matched){OFF}")
        return
    for h in hits:
        # Mark hits only this index found -- that disagreement is the point.
        unique = "" if h.chunk_id in other_ids else f" {YELLOW}◄ only here{OFF}"
        trail = " > ".join(h.heading_trail)
        print(f"    {h.rank}. [{h.score:6.3f}] {BOLD}{h.chunk_id}{OFF}{unique}")
        print(f"       {DIM}{trail}{OFF}")
        print(f"       {snippet(h)}")


def main() -> int:
    settings = get_settings()
    store = Store(settings)
    try:
        backend = store.init_schema()
        total = store.count()
    except Exception as exc:  # noqa: BLE001
        print(f"database unavailable ({type(exc).__name__}) -- run `make db-up`")
        return 1
    if total == 0:
        print("index is empty -- run `make ingest && make index`")
        return 1

    print(f"{BOLD}RAG retrieval demo{OFF}  {DIM}(generation arrives in slice 4){OFF}")
    print(f"{DIM}{total} chunks | lexical={backend} | "
          f"embeddings={settings.embedding_provider}/{settings.embedding_model}{OFF}")
    print(f"{DIM}try: 'how do I get money back for a work trip' then '/var/log/db'{OFF}")
    print(f"{DIM}:k N  :source TEXT  :nosource  :stats  :q{OFF}")

    embedder = get_embedder(settings)
    retriever = Retriever(store=store, embedder=embedder, settings=settings)
    k = settings.retrieval_top_k
    source: str | None = None

    while True:
        try:
            query = input(f"\n{CYAN}?{OFF} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue

        if query in (":q", ":quit", ":exit"):
            return 0
        if query.startswith(":k "):
            k = max(1, int(query[3:].strip() or k)); print(f"{DIM}k = {k}{OFF}"); continue
        if query.startswith(":source "):
            source = query[8:].strip(); print(f"{DIM}source filter: {source!r}{OFF}"); continue
        if query == ":nosource":
            source = None; print(f"{DIM}source filter cleared{OFF}"); continue
        if query == ":stats":
            with store.conn() as c, c.cursor() as cur:
                cur.execute("SELECT source, count(*) n FROM chunks GROUP BY 1 ORDER BY 2 DESC")
                for r in cur.fetchall():
                    print(f"    {r['n']:4}  {r['source']}")
            continue

        filters = None
        if source:
            with store.conn() as c, c.cursor() as cur:
                cur.execute("SELECT DISTINCT source FROM chunks WHERE source ILIKE %s",
                            (f"%{source}%",))
                matched = [r["source"] for r in cur.fetchall()]
            if not matched:
                print(f"{DIM}no source matches {source!r}{OFF}")
                continue
            filters = Filters(sources=matched)

        vector = store.vector_search(embedder.embed_query(query), k=k, filters=filters)
        lexical = store.lexical_search(query, k=k, filters=filters)
        result = retriever.retrieve(query, filters=filters, top_k=k, top_n=k)

        show("VECTOR   (meaning)", GREEN, vector, lexical)
        show("LEXICAL  (exact words)", CYAN, lexical, vector)

        print(f"\n  {YELLOW}{BOLD}HYBRID   (RRF fusion){OFF}")
        if result.notes:
            print(f"    {DIM}{'; '.join(result.notes)}{OFF}")
        for i, c in enumerate(result.chunks, 1):
            badge = "+".join(c.found_by)
            score = (f"rerank {c.rerank_score:.1f}/10"
                     if c.rerank_score is not None else f"rrf {c.fusion_score:.4f}")
            print(f"    {i}. [{score}] {BOLD}{c.chunk_id}{OFF} {DIM}({badge}){OFF}")
            print(f"       {DIM}{' > '.join(c.heading_trail)}{OFF}")
            if c.reason:
                print(f"       {DIM}why: {c.reason[:80]}{OFF}")

        v_ids = [h.chunk_id for h in vector]
        l_ids = [h.chunk_id for h in lexical]
        h_ids = [c.chunk_id for c in result.chunks]
        print(f"\n  {DIM}#1 by index -> vector: {v_ids[0].split('#')[-1] if v_ids else '-'}"
              f" | lexical: {l_ids[0].split('#')[-1] if l_ids else '-'}"
              f" | hybrid: {h_ids[0].split('#')[-1] if h_ids else '-'}{OFF}")
        if v_ids and l_ids and h_ids and v_ids[0] != l_ids[0]:
            winner = "vector" if h_ids[0] == v_ids[0] else (
                "lexical" if h_ids[0] == l_ids[0] else "neither -- agreement won")
            print(f"  {DIM}the indexes disagreed on #1; RRF sided with: {winner}{OFF}")


if __name__ == "__main__":
    raise SystemExit(main())
