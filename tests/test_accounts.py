"""Auth, isolation, and limits.

Security properties are asserted, not assumed: password storage, whether one
user can read another's conversation, and whether the cost ceiling actually
stops anything.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.accounts import Accounts, hash_password, verify_password  # noqa: E402
from app.config import get_settings  # noqa: E402


# -- password hashing (no database needed) ------------------------------

def test_password_is_not_stored_in_plaintext():
    stored = hash_password("hunter2hunter2")
    assert "hunter2hunter2" not in stored
    assert stored.startswith("pbkdf2_sha256$")


def test_verify_accepts_correct_and_rejects_wrong():
    stored = hash_password("hunter2hunter2")
    assert verify_password("hunter2hunter2", stored)
    assert not verify_password("hunter2hunter3", stored)


def test_same_password_gets_a_different_hash():
    """Per-user salt: without it, identical passwords are visibly identical in
    the table and one cracked hash breaks every account that shares it."""
    assert hash_password("samepassword") != hash_password("samepassword")


def test_malformed_hash_is_rejected_not_crashed():
    for bad in ("", "garbage", "pbkdf2_sha256$notanumber$aa$bb", "md5$1$aa$bb"):
        assert not verify_password("x", bad)


# -- database-backed ----------------------------------------------------

@pytest.fixture(scope="module")
def accounts() -> Accounts:
    import psycopg

    base = get_settings()
    test_url = base.database_url + "_test"
    try:
        with psycopg.connect(base.database_url, autocommit=True) as c, c.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_url.rsplit("/", 1)[-1],))
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{test_url.rsplit("/", 1)[-1]}"')
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unavailable ({type(exc).__name__}) -- run `make db-up`")

    a = Accounts(replace(base, database_url=test_url))
    a.init_schema()
    with a.conn() as c, c.cursor() as cur:
        cur.execute("TRUNCATE users, sessions, conversations, messages, feedback CASCADE")
        c.commit()
    return a


def test_signup_login_and_session(accounts: Accounts):
    user = accounts.create_user("alice@example.com", "password123")
    assert accounts.authenticate("alice@example.com", "password123")
    assert accounts.authenticate("alice@example.com", "wrong") is None
    token = accounts.create_session(user.id)
    assert accounts.user_for_token(token).email == "alice@example.com"
    accounts.delete_session(token)
    assert accounts.user_for_token(token) is None


def test_email_is_normalised(accounts: Accounts):
    accounts.create_user("  Bob@Example.COM ", "password123")
    assert accounts.authenticate("bob@example.com", "password123")


def test_unknown_token_returns_nobody(accounts: Accounts):
    assert accounts.user_for_token("not-a-real-token") is None
    assert accounts.user_for_token(None) is None


def test_short_password_is_refused(accounts: Accounts):
    with pytest.raises(ValueError):
        accounts.create_user("carol@example.com", "short")


def test_a_user_cannot_read_another_users_conversation(accounts: Accounts):
    """The one that matters: without the user_id scope, any signed-in user
    reads any conversation by guessing an integer."""
    alice = accounts.create_user("a1@example.com", "password123")
    mallory = accounts.create_user("m1@example.com", "password123")
    conv = accounts.create_conversation(alice.id)
    accounts.add_message(conv, "user", "alice's private question")

    assert accounts.owns_conversation(conv, alice.id)
    assert not accounts.owns_conversation(conv, mallory.id)
    assert accounts.get_messages(conv, alice.id)
    assert accounts.get_messages(conv, mallory.id) == []


def test_question_limit_blocks_further_questions(accounts: Accounts):
    user = accounts.create_user("limited@example.com", "password123")
    user = replace(user, daily_question_limit=2)
    conv = accounts.create_conversation(user.id)
    assert accounts.check_limits(user).allowed
    accounts.add_message(conv, "user", "one")
    accounts.add_message(conv, "user", "two")
    status = accounts.check_limits(user)
    assert not status.allowed and "question limit" in status.reason


def test_cost_ceiling_blocks_further_questions(accounts: Accounts):
    user = accounts.create_user("spender@example.com", "password123")
    user = replace(user, daily_cost_limit_cents=10)
    conv = accounts.create_conversation(user.id)
    accounts.add_message(conv, "user", "q", cost_cents=25)
    status = accounts.check_limits(user)
    assert not status.allowed and "cost ceiling" in status.reason


def test_feedback_is_one_per_user_per_message(accounts: Accounts):
    user = accounts.create_user("rater@example.com", "password123")
    conv = accounts.create_conversation(user.id)
    mid = accounts.add_message(conv, "assistant", "an answer")
    accounts.add_feedback(mid, user.id, 1)
    accounts.add_feedback(mid, user.id, -1, "actually wrong")
    board = accounts.feedback_dashboard()
    assert board["counts"]["total"] == 1, "changing your mind must update, not duplicate"
    assert board["counts"]["down"] == 1


def test_feedback_carries_the_chunks_that_produced_the_answer(accounts: Accounts):
    """A thumbs-down joined to its retrieved chunk ids is debuggable; a
    thumbs-down alone is a mood."""
    user = accounts.create_user("debug@example.com", "password123")
    conv = accounts.create_conversation(user.id)
    accounts.add_message(conv, "user", "why is this wrong?")
    mid = accounts.add_message(conv, "assistant", "bad answer",
                               retrieval={"chunk_ids": ["a#c0001"]})
    accounts.add_feedback(mid, user.id, -1, "not what I asked")
    item = next(i for i in accounts.feedback_dashboard()["items"] if i["comment"] == "not what I asked")
    assert item["retrieval"]["chunk_ids"] == ["a#c0001"]
    assert item["question"] == "why is this wrong?"


def test_invalid_rating_is_rejected(accounts: Accounts):
    user = accounts.create_user("bad@example.com", "password123")
    conv = accounts.create_conversation(user.id)
    mid = accounts.add_message(conv, "assistant", "a")
    with pytest.raises(ValueError):
        accounts.add_feedback(mid, user.id, 5)
