"""Users, sessions, conversations, feedback, and per-user limits.

All of it in the same Postgres that holds the index -- one datastore means one
backup, one connection string, and no consistency problem between "what the
user asked" and "what was retrieved".

Password hashing uses PBKDF2-HMAC-SHA256 from the standard library. Argon2id
is the better algorithm and needs a compiled dependency; PBKDF2 at 480k
iterations is what Django ships as its default and is a defensible choice for
an internal tool. What is NOT defensible, and is the actual risk here, is
storing anything reversible -- so passwords are salted per user and never
recoverable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from .config import Settings, get_settings

PBKDF2_ITERATIONS = 480_000
SESSION_DAYS = 14


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    # Constant-time: a timing difference here leaks the hash one byte at a time.
    return hmac.compare_digest(candidate.hex(), digest_hex)


@dataclass(slots=True)
class User:
    id: int
    email: str
    is_admin: bool
    daily_question_limit: int
    daily_cost_limit_cents: int


@dataclass(slots=True)
class LimitStatus:
    allowed: bool
    reason: str = ""
    questions_used: int = 0
    questions_limit: int = 0
    cents_used: int = 0
    cents_limit: int = 0

    @property
    def questions_remaining(self) -> int:
        return max(0, self.questions_limit - self.questions_used)


class Accounts:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @contextmanager
    def conn(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.settings.database_url, row_factory=dict_row) as c:
            yield c

    # -- schema --------------------------------------------------------

    def init_schema(self) -> None:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id                      BIGSERIAL PRIMARY KEY,
                    email                   TEXT UNIQUE NOT NULL,
                    password_hash           TEXT NOT NULL,
                    is_admin                BOOLEAN NOT NULL DEFAULT false,
                    daily_question_limit    INT NOT NULL DEFAULT 100,
                    daily_cost_limit_cents  INT NOT NULL DEFAULT 200,
                    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token       TEXT PRIMARY KEY,
                    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at  TIMESTAMPTZ NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          BIGSERIAL PRIMARY KEY,
                    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title       TEXT NOT NULL DEFAULT 'New conversation',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id               BIGSERIAL PRIMARY KEY,
                    conversation_id  BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role             TEXT NOT NULL CHECK (role IN ('user','assistant')),
                    content          TEXT NOT NULL,
                    citations        JSONB NOT NULL DEFAULT '[]',
                    retrieval        JSONB NOT NULL DEFAULT '{}',
                    refused          BOOLEAN NOT NULL DEFAULT false,
                    cost_cents       INT NOT NULL DEFAULT 0,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id          BIGSERIAL PRIMARY KEY,
                    message_id  BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
                    comment     TEXT NOT NULL DEFAULT '',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (message_id, user_id)
                )""")
            for sql in (
                "CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions (user_id)",
                "CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations (user_id, updated_at DESC)",
                "CREATE INDEX IF NOT EXISTS messages_conv_idx ON messages (conversation_id, id)",
                "CREATE INDEX IF NOT EXISTS messages_user_day_idx ON messages (conversation_id, created_at)",
            ):
                cur.execute(sql)
            c.commit()

    # -- users ---------------------------------------------------------

    def create_user(self, email: str, password: str, *, is_admin: bool = False) -> User:
        email = email.strip().lower()
        if not email or "@" not in email:
            raise ValueError("invalid email")
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        with self.conn() as c, c.cursor() as cur:
            cur.execute(
                """INSERT INTO users (email, password_hash, is_admin,
                                      daily_question_limit, daily_cost_limit_cents)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id, email, is_admin, daily_question_limit, daily_cost_limit_cents""",
                (email, hash_password(password), is_admin,
                 self.settings.daily_question_limit, self.settings.daily_cost_limit_cents),
            )
            row = cur.fetchone()
            c.commit()
        return User(**row)

    def authenticate(self, email: str, password: str) -> User | None:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email.strip().lower(),))
            row = cur.fetchone()
        # Hash even when the user does not exist, so response time does not
        # reveal which emails are registered.
        stored = row["password_hash"] if row else hash_password("dummy")
        if not verify_password(password, stored) or not row:
            return None
        return User(id=row["id"], email=row["email"], is_admin=row["is_admin"],
                    daily_question_limit=row["daily_question_limit"],
                    daily_cost_limit_cents=row["daily_cost_limit_cents"])

    # -- sessions ------------------------------------------------------

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
        with self.conn() as c, c.cursor() as cur:
            cur.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (%s, %s, %s)",
                        (token, user_id, expires))
            c.commit()
        return token

    def user_for_token(self, token: str | None) -> User | None:
        if not token:
            return None
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                SELECT u.id, u.email, u.is_admin, u.daily_question_limit,
                       u.daily_cost_limit_cents
                FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token = %s AND s.expires_at > now()""", (token,))
            row = cur.fetchone()
        return User(**row) if row else None

    def delete_session(self, token: str) -> None:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE token = %s", (token,))
            c.commit()

    # -- limits --------------------------------------------------------

    def check_limits(self, user: User) -> LimitStatus:
        """Rolling 24-hour window rather than calendar-day.

        A calendar reset lets a user burn the whole budget at 23:59 and again
        at 00:01; a rolling window cannot be gamed that way.
        """
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                SELECT count(*) AS questions, COALESCE(sum(m.cost_cents), 0) AS cents
                FROM messages m JOIN conversations v ON v.id = m.conversation_id
                WHERE v.user_id = %s AND m.role = 'user'
                  AND m.created_at > now() - interval '24 hours'""", (user.id,))
            row = cur.fetchone()
        status = LimitStatus(
            allowed=True, questions_used=row["questions"], cents_used=int(row["cents"]),
            questions_limit=user.daily_question_limit,
            cents_limit=user.daily_cost_limit_cents,
        )
        if status.questions_used >= user.daily_question_limit:
            status.allowed = False
            status.reason = (f"question limit reached ({user.daily_question_limit} per 24h)")
        elif status.cents_used >= user.daily_cost_limit_cents:
            status.allowed = False
            status.reason = (f"cost ceiling reached "
                             f"(${user.daily_cost_limit_cents / 100:.2f} per 24h)")
        return status

    # -- conversations -------------------------------------------------

    def create_conversation(self, user_id: int, title: str = "New conversation") -> int:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id",
                        (user_id, title[:120]))
            cid = cur.fetchone()["id"]
            c.commit()
        return cid

    def list_conversations(self, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                SELECT id, title, updated_at FROM conversations
                WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s""", (user_id, limit))
            return [dict(r) for r in cur.fetchall()]

    def get_messages(self, conversation_id: int, user_id: int) -> list[dict[str, Any]]:
        """Always scoped by user_id: without it, any logged-in user could read
        any conversation by guessing an integer."""
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                SELECT m.id, m.role, m.content, m.citations, m.refused, m.created_at,
                       f.rating, f.comment
                FROM messages m
                JOIN conversations v ON v.id = m.conversation_id
                LEFT JOIN feedback f ON f.message_id = m.id AND f.user_id = %s
                WHERE m.conversation_id = %s AND v.user_id = %s
                ORDER BY m.id""", (user_id, conversation_id, user_id))
            return [dict(r) for r in cur.fetchall()]

    def owns_conversation(self, conversation_id: int, user_id: int) -> bool:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
                        (conversation_id, user_id))
            return cur.fetchone() is not None

    def add_message(
        self, conversation_id: int, role: str, content: str, *,
        citations: list[dict[str, Any]] | None = None,
        retrieval: dict[str, Any] | None = None,
        refused: bool = False, cost_cents: int = 0,
    ) -> int:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (conversation_id, role, content, citations,
                                      retrieval, refused, cost_cents)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (conversation_id, role, content, json.dumps(citations or []),
                 json.dumps(retrieval or {}), refused, cost_cents))
            mid = cur.fetchone()["id"]
            cur.execute("UPDATE conversations SET updated_at = now() WHERE id = %s",
                        (conversation_id,))
            c.commit()
        return mid

    def set_title_from_first_question(self, conversation_id: int, question: str) -> None:
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                UPDATE conversations SET title = %s
                WHERE id = %s AND title = 'New conversation'""",
                (question[:120], conversation_id))
            c.commit()

    # -- feedback ------------------------------------------------------

    def add_feedback(self, message_id: int, user_id: int, rating: int, comment: str = "") -> None:
        if rating not in (-1, 1):
            raise ValueError("rating must be -1 or 1")
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback (message_id, user_id, rating, comment)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (message_id, user_id)
                DO UPDATE SET rating = EXCLUDED.rating, comment = EXCLUDED.comment,
                              created_at = now()""",
                (message_id, user_id, rating, comment[:2000]))
            c.commit()

    def feedback_dashboard(self, limit: int = 100) -> dict[str, Any]:
        """Admin view. The subject asks for feedback "stored for offline
        analysis" -- which means joining it back to what was retrieved, not
        just counting thumbs. A thumbs-down plus the chunk ids that produced
        it is a debuggable retrieval failure; a thumbs-down alone is a mood."""
        with self.conn() as c, c.cursor() as cur:
            cur.execute("""
                SELECT count(*) FILTER (WHERE rating = 1)  AS up,
                       count(*) FILTER (WHERE rating = -1) AS down,
                       count(*)                            AS total
                FROM feedback""")
            counts = dict(cur.fetchone())
            cur.execute("""
                SELECT f.rating, f.comment, f.created_at, u.email,
                       m.content AS answer, m.refused, m.retrieval,
                       (SELECT content FROM messages q
                        WHERE q.conversation_id = m.conversation_id AND q.id < m.id
                          AND q.role = 'user'
                        ORDER BY q.id DESC LIMIT 1) AS question
                FROM feedback f
                JOIN messages m ON m.id = f.message_id
                JOIN users u ON u.id = f.user_id
                ORDER BY f.created_at DESC LIMIT %s""", (limit,))
            items = [dict(r) for r in cur.fetchall()]
            cur.execute("""
                SELECT count(*) AS n,
                       count(*) FILTER (WHERE refused) AS refusals,
                       COALESCE(sum(cost_cents), 0) AS cents
                FROM messages WHERE role = 'assistant'""")
            answers = dict(cur.fetchone())
        return {"counts": counts, "answers": answers, "items": items}
