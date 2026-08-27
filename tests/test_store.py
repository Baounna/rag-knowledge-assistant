"""Retrieval integration tests. Skipped automatically if the DB is not up.

    make db-up && make test
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.store import Filters, Store  # noqa: E402


def _store_or_skip() -> Store:
    store = Store()
    try:
        store.init_schema()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable ({type(exc).__name__}) -- run `make db-up`")
    return store


@pytest.fixture(scope="module")
def store() -> Store:
    s = _store_or_skip()
    s.init_schema(drop=True)
    s.upsert_chunks([
        {
            "chunk_id": "fin#c0", "doc_id": "fin",
            "text": "Reimbursement Deadlines\n\nAll expense reports must be submitted "
                    "within 30 days of the expense being incurred.",
            "index_text": "Expense Policy > Reimbursement Deadlines\n\nAll expense reports "
                          "must be submitted within 30 days of the expense being incurred.",
            "heading_trail": ["Expense Policy", "Reimbursement Deadlines"],
            "source": "Finance Handbook", "url": "https://x/fin", "author": None,
            "doc_date": date(2025, 3, 14), "position": 0, "content_hash": "h1",
            "embedding": None,
        },
        {
            "chunk_id": "eng#c0", "doc_id": "eng",
            "text": "Database Migrations\n\nIf it fails, check the logs in /var/log/db "
                    "and retry with --force.",
            "index_text": "Engineering Handbook > Database Migrations\n\nIf it fails, check "
                          "the logs in /var/log/db and retry with --force.",
            "heading_trail": ["Engineering Handbook", "Database Migrations"],
            "source": "Engineering Handbook", "url": "https://x/eng", "author": None,
            "doc_date": date(2025, 6, 2), "position": 0, "content_hash": "h2",
            "embedding": None,
        },
    ])
    return s


def test_lexical_finds_exact_terms(store: Store):
    hits = store.lexical_search("/var/log/db", k=5)
    assert hits and hits[0].chunk_id == "eng#c0"


def test_lexical_stems(store: Store):
    """'deadline' must match 'Deadlines' -- without stemming the miss is silent."""
    if store.lexical_backend != "bm25":
        pytest.skip("stemming assertion is backend-specific")
    assert any(h.chunk_id == "fin#c0" for h in store.lexical_search("deadline", k=5))


@pytest.mark.parametrize("query", ["--force", "AND OR NOT", '"unclosed', "e.g. 30 days?", "*"])
def test_lexical_survives_adversarial_input(store: Store, query: str):
    """User input must never reach a query parser. Regression: `--force` raised
    a Tantivy parse error straight out of the search endpoint."""
    store.lexical_search(query, k=3)


def test_source_filter_excludes(store: Store):
    hits = store.lexical_search("expense", k=5, filters=Filters(sources=["Engineering Handbook"]))
    assert all(h.source == "Engineering Handbook" for h in hits)


def test_date_filter_excludes(store: Store):
    hits = store.lexical_search("logs", k=5, filters=Filters(date_from=date(2025, 6, 1)))
    assert all(h.doc_date >= date(2025, 6, 1) for h in hits)


def test_upsert_is_idempotent(store: Store):
    """Uses its own chunk_id: mutating a row other tests read makes the suite
    order-dependent, which is a bug in the tests, not in the code."""
    row = {
        "chunk_id": "tmp#c0", "doc_id": "tmp", "text": "first", "index_text": "first",
        "heading_trail": [], "source": "Temp", "url": None, "author": None,
        "doc_date": None, "position": 0, "content_hash": "t1", "embedding": None,
    }
    store.upsert_chunks([row])
    after_insert = store.count()
    store.upsert_chunks([{**row, "text": "second", "index_text": "second", "content_hash": "t2"}])
    assert store.count() == after_insert, "same chunk_id must update, not duplicate"


def test_hits_carry_citation_metadata(store: Store):
    hit = store.lexical_search("expense", k=1)[0]
    assert hit.url, "no url means no clickable citation"
    assert hit.citation().startswith("[") and ":" in hit.citation()
