"""Postgres store: the two indexes, side by side.

One table, two indexes over the same rows:

    embedding  VECTOR(n)  + HNSW      -> semantic search  ("means the same")
    index_text TEXT       + BM25/GIN  -> lexical search   ("same words")

They are complementary, not redundant. Vector search finds the expense policy
for "money back for a work trip"; lexical search finds error code `E_4271`,
which no embedding model has ever seen. Neither alone is enough -- which is
what hybrid retrieval (slice 3) is for.

Both indexes are built over `index_text`, which is the heading trail plus the
chunk body (`Chunk.embedding_text()`), never the bare text. A chunk reading
"Set the timeout to 30 seconds." is unretrievable for "payments timeout"
unless its section path is part of what got indexed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterator, Sequence

import psycopg
from psycopg.rows import dict_row

from .config import Settings, get_settings


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    text: str
    heading_trail: list[str]
    source: str
    url: str | None
    doc_date: date | None
    score: float
    rank: int                 # 1-based position in THIS index's result list
    retriever: str            # "vector" | "lexical"

    def citation(self) -> str:
        return f"[{self.source}:{self.chunk_id}]"


@dataclass(slots=True)
class Filters:
    """Retrieval filters the subject asks for (source, date)."""

    sources: Sequence[str] | None = None
    date_from: date | None = None
    date_to: date | None = None

    def sql(self, params: list[Any]) -> str:
        clauses: list[str] = []
        if self.sources:
            clauses.append("source = ANY(%s)")
            params.append(list(self.sources))
        if self.date_from:
            clauses.append("doc_date >= %s")
            params.append(self.date_from)
        if self.date_to:
            clauses.append("doc_date <= %s")
            params.append(self.date_to)
        return (" AND " + " AND ".join(clauses)) if clauses else ""


class Store:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.lexical_backend = "bm25"   # downgraded to "tsvector" if pg_search absent

    @contextmanager
    def conn(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.settings.database_url, row_factory=dict_row) as c:
            yield c

    # -- schema --------------------------------------------------------

    def init_schema(self, *, drop: bool = False) -> str:
        dim = self.settings.embedding_dim
        with self.conn() as c, c.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Real BM25 (Tantivy) if the image provides it; Postgres full-text
            # search otherwise. Both are lexical rankers -- BM25 is the better
            # one, and the subject names it, so prefer it and say plainly in
            # the report which backend actually ran.
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
                self.lexical_backend = "bm25"
            except psycopg.Error:
                c.rollback()
                self.lexical_backend = "tsvector"

            if drop:
                cur.execute("DROP TABLE IF EXISTS chunks CASCADE")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id            BIGSERIAL PRIMARY KEY,
                    chunk_id      TEXT UNIQUE NOT NULL,
                    doc_id        TEXT NOT NULL,
                    text          TEXT NOT NULL,
                    index_text    TEXT NOT NULL,
                    heading_trail TEXT[] NOT NULL DEFAULT '{{}}',
                    source        TEXT NOT NULL DEFAULT '',
                    url           TEXT,
                    author        TEXT,
                    doc_date      DATE,
                    position      INT NOT NULL DEFAULT 0,
                    content_hash  TEXT NOT NULL,
                    embedding     VECTOR({dim}),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source)")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_date_idx ON chunks (doc_date)")

            # Cosine distance, because embedding models are trained on cosine
            # similarity; L2 on unnormalised vectors ranks by magnitude too.
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )

            if self.lexical_backend == "bm25":
                # English stemming matters more than it looks: without it,
                # a search for "deadline" does not match a document that says
                # "Deadlines", and the miss is completely silent. Postgres FTS
                # (the fallback path) stems by default via the 'english'
                # dictionary; pg_search does not, so ask for it explicitly.
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_bm25_idx ON chunks
                    USING bm25 (id, index_text, source, doc_id)
                    WITH (
                        key_field='id',
                        text_fields='{
                            "index_text": {
                                "tokenizer": {"type": "default", "stemmer": "English"}
                            }
                        }'
                    )
                    """
                )
            else:
                cur.execute(
                    """
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', index_text)) STORED
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv)"
                )
            c.commit()
        return self.lexical_backend

    # -- writes --------------------------------------------------------

    def upsert_chunks(self, rows: Sequence[dict[str, Any]]) -> int:
        """Insert or update by chunk_id.

        `content_hash` lets a re-ingest skip unchanged chunks upstream, so an
        edit to one document does not re-embed the whole corpus.
        """
        if not rows:
            return 0
        sql = """
            INSERT INTO chunks (chunk_id, doc_id, text, index_text, heading_trail,
                                source, url, author, doc_date, position,
                                content_hash, embedding)
            VALUES (%(chunk_id)s, %(doc_id)s, %(text)s, %(index_text)s,
                    %(heading_trail)s, %(source)s, %(url)s, %(author)s,
                    %(doc_date)s, %(position)s, %(content_hash)s, %(embedding)s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                index_text = EXCLUDED.index_text,
                heading_trail = EXCLUDED.heading_trail,
                source = EXCLUDED.source,
                url = EXCLUDED.url,
                author = EXCLUDED.author,
                doc_date = EXCLUDED.doc_date,
                position = EXCLUDED.position,
                content_hash = EXCLUDED.content_hash,
                embedding = EXCLUDED.embedding
        """
        with self.conn() as c, c.cursor() as cur:
            cur.executemany(sql, rows)
            c.commit()
            return len(rows)

    def existing_hashes(self) -> dict[str, str]:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("SELECT chunk_id, content_hash FROM chunks")
            return {r["chunk_id"]: r["content_hash"] for r in cur.fetchall()}

    def prune_to(self, keep_chunk_ids: Sequence[str]) -> list[str]:
        """Delete indexed chunks that are no longer produced by the corpus.

        Without this, deleting a document from `corpus/` leaves its chunks in
        the index forever -- so the assistant keeps answering from, and citing,
        a document that no longer exists. For a policy assistant that is not a
        tidiness problem, it is a wrong-answer problem.

        Also catches documents that SHRANK: chunk ids are positional, so an
        edit that removes a section orphans its trailing chunks.
        """
        with self.conn() as c, c.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE NOT (chunk_id = ANY(%s)) RETURNING chunk_id",
                (list(keep_chunk_ids),),
            )
            removed = [r["chunk_id"] for r in cur.fetchall()]
            c.commit()
            return removed

    def count(self) -> int:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM chunks")
            return cur.fetchone()["n"]

    # -- reads ---------------------------------------------------------

    _SELECT = """
        chunk_id, text, heading_trail, source, url, doc_date
    """

    def vector_search(
        self, query_vec: Sequence[float], k: int | None = None,
        filters: Filters | None = None
    ) -> list[SearchHit]:
        k = k or self.settings.retrieval_top_k
        vec = json.dumps(list(query_vec))
        filter_params: list[Any] = []
        where = (filters or Filters()).sql(filter_params)
        # Placeholder order must match the SQL exactly: SELECT, WHERE, ORDER, LIMIT.
        params: list[Any] = [vec, *filter_params, vec, k]
        sql = f"""
            SELECT {self._SELECT},
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE embedding IS NOT NULL {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self.conn() as c, c.cursor() as cur:
            cur.execute(sql, params)
            return self._hits(cur.fetchall(), "vector")

    def lexical_search(
        self, query: str, k: int | None = None, filters: Filters | None = None
    ) -> list[SearchHit]:
        k = k or self.settings.retrieval_top_k
        filter_params: list[Any] = []
        where = (filters or Filters()).sql(filter_params)
        if self.lexical_backend == "bm25":
            params: list[Any] = [query, *filter_params, k]
            # paradedb.match() TOKENISES the input. The bare `index_text @@@ %s`
            # form instead PARSES it as Tantivy query syntax, so a user typing
            # `--force`, `AND`, or a stray quote crashes the search endpoint
            # with a parse error. Never feed user input to a query parser.
            sql = f"""
                SELECT {self._SELECT}, paradedb.score(id) AS score
                FROM chunks
                WHERE id @@@ paradedb.match('index_text', %s) {where}
                ORDER BY paradedb.score(id) DESC
                LIMIT %s
            """
        else:
            # query appears twice: once to rank, once to filter.
            params = [query, query, *filter_params, k]
            sql = f"""
                SELECT {self._SELECT},
                       ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS score
                FROM chunks
                WHERE tsv @@ plainto_tsquery('english', %s) {where}
                ORDER BY score DESC
                LIMIT %s
            """
        with self.conn() as c, c.cursor() as cur:
            cur.execute("SET LOCAL paradedb.planner_warnings = 'off'")
            cur.execute(sql, params)
            return self._hits(cur.fetchall(), "lexical")

    @staticmethod
    def _hits(rows: list[dict[str, Any]], retriever: str) -> list[SearchHit]:
        return [
            SearchHit(
                chunk_id=r["chunk_id"],
                text=r["text"],
                heading_trail=list(r["heading_trail"] or []),
                source=r["source"],
                url=r["url"],
                doc_date=r["doc_date"],
                score=float(r["score"]),
                rank=i + 1,
                retriever=retriever,
            )
            for i, r in enumerate(rows)
        ]
