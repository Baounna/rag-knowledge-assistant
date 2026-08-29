"""FastAPI app: auth, chat with streaming, history, feedback, admin.

Streaming is done with Server-Sent Events rather than WebSockets: the traffic
is one-directional (server -> browser), SSE reconnects on its own, and it
survives proxies that mangle WebSocket upgrades. A chat UI does not need
duplex.

The answer is streamed for latency, then persisted once complete -- so the
citation validation in `answer.py` runs on the finished text rather than on
partial output, where a half-written `[chunk_id]` would look invalid.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .accounts import Accounts, User
from .answer import AnswerGenerator, link_citations, validate_citations
from .config import get_settings
from .llm import LLM
from .retrieval import Retriever
from .store import Filters, Store

settings = get_settings()
app = FastAPI(title="Internal Knowledge Assistant", version="1.0")

accounts = Accounts(settings)
store = Store(settings)
llm = LLM(settings)
retriever = Retriever(store=store, llm=llm, settings=settings)
generator = AnswerGenerator(llm=llm, settings=settings)

WEB_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "web"


@app.on_event("startup")
def startup() -> None:
    store.init_schema()
    accounts.init_schema()


# -- auth ---------------------------------------------------------------

def current_user(session: str | None = Cookie(default=None)) -> User:
    user = accounts.user_for_token(session)
    if not user:
        raise HTTPException(status_code=401, detail="not signed in")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin only")
    return user


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)


def _set_session(response: Response, token: str) -> None:
    # httponly: JavaScript cannot read it, so an XSS bug cannot steal the
    # session. samesite=lax blocks the cookie on cross-site POSTs, which is
    # the CSRF defence for this app.
    response.set_cookie(
        "session", token, httponly=True, samesite="lax",
        secure=settings.cookie_secure, max_age=60 * 60 * 24 * 14, path="/",
    )


@app.post("/api/signup")
def signup(body: Credentials, response: Response) -> dict[str, Any]:
    if not settings.allow_signup:
        raise HTTPException(status_code=403, detail="signup is disabled")
    try:
        user = accounts.create_user(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # unique violation
        raise HTTPException(status_code=409, detail="email already registered") from exc
    _set_session(response, accounts.create_session(user.id))
    return {"email": user.email, "is_admin": user.is_admin}


@app.post("/api/login")
def login(body: Credentials, response: Response) -> dict[str, Any]:
    user = accounts.authenticate(body.email, body.password)
    if not user:
        # One message for both wrong-email and wrong-password: distinguishing
        # them tells an attacker which emails are registered.
        raise HTTPException(status_code=401, detail="invalid email or password")
    _set_session(response, accounts.create_session(user.id))
    return {"email": user.email, "is_admin": user.is_admin}


@app.post("/api/logout")
def logout(response: Response, session: str | None = Cookie(default=None)) -> dict[str, str]:
    if session:
        accounts.delete_session(session)
    response.delete_cookie("session", path="/")
    return {"status": "signed out"}


@app.get("/api/me")
def me(user: User = Depends(current_user)) -> dict[str, Any]:
    limits = accounts.check_limits(user)
    return {
        "email": user.email, "is_admin": user.is_admin,
        "questions_used": limits.questions_used, "questions_limit": limits.questions_limit,
        "cents_used": limits.cents_used, "cents_limit": limits.cents_limit,
        "llm_available": llm.available,
    }


# -- conversations ------------------------------------------------------

@app.get("/api/conversations")
def list_conversations(user: User = Depends(current_user)) -> list[dict[str, Any]]:
    return accounts.list_conversations(user.id)


@app.post("/api/conversations")
def new_conversation(user: User = Depends(current_user)) -> dict[str, Any]:
    return {"id": accounts.create_conversation(user.id)}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int, user: User = Depends(current_user)) -> dict[str, Any]:
    if not accounts.owns_conversation(conversation_id, user.id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"messages": accounts.get_messages(conversation_id, user.id)}


# -- chat ---------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    sources: list[str] | None = None


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/api/ask")
def ask(body: AskRequest, user: User = Depends(current_user)) -> StreamingResponse:
    limits = accounts.check_limits(user)
    if not limits.allowed:
        raise HTTPException(status_code=429, detail=limits.reason)

    conversation_id = body.conversation_id
    if conversation_id is None:
        conversation_id = accounts.create_conversation(user.id)
    elif not accounts.owns_conversation(conversation_id, user.id):
        raise HTTPException(status_code=404, detail="conversation not found")

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in accounts.get_messages(conversation_id, user.id)
    ][-6:]

    def stream() -> AsyncIterator[str]:
        yield _sse("start", {"conversation_id": conversation_id})
        accounts.add_message(conversation_id, "user", body.question)
        accounts.set_title_from_first_question(conversation_id, body.question)

        filters = Filters(sources=body.sources) if body.sources else None
        before = llm.usage.output_tokens + llm.usage.input_tokens

        result = retriever.retrieve(
            body.question, filters=filters, history=history,
            use_rewrite=settings.enable_rewrite, use_rerank=settings.enable_rerank,
        )
        yield _sse("retrieval", {
            "chunks": [
                {"chunk_id": c.chunk_id, "source": c.source, "url": c.url,
                 "heading_trail": c.heading_trail, "snippet": c.text[:220],
                 "found_by": c.found_by, "rerank_score": c.rerank_score}
                for c in result.chunks
            ],
            "queries_used": result.queries_used,
            "notes": result.notes,
        })

        parts: list[str] = []
        try:
            for piece in generator.stream(body.question, result, history):
                parts.append(piece)
                yield _sse("token", {"text": piece})
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})

        text = "".join(parts).strip()
        # Validate on the COMPLETE text: a half-streamed "[chunk-" would look
        # like a fabricated citation mid-flight.
        citations = validate_citations(text, result.chunks)
        refused = text.lower().startswith("i don't know based on")
        tokens = (llm.usage.output_tokens + llm.usage.input_tokens) - before
        # Price per token comes from the BACKEND, not from config: a local
        # model costs nothing, and charging it against the user's ceiling
        # would lock them out after ~60 free questions for no reason.
        cost_cents = max(0, round(tokens * llm.backend.cents_per_1k_tokens / 1000))

        message_id = accounts.add_message(
            conversation_id, "assistant", text,
            citations=[{"chunk_id": c.chunk_id, "source": c.source,
                        "url": c.url, "valid": c.valid} for c in citations],
            retrieval={"chunk_ids": [c.chunk_id for c in result.chunks],
                       "queries": result.queries_used},
            refused=refused, cost_cents=cost_cents,
        )
        yield _sse("done", {
            "message_id": message_id,
            "linked_text": link_citations(text, citations),
            "citations": [{"chunk_id": c.chunk_id, "source": c.source,
                           "url": c.url, "valid": c.valid} for c in citations],
            "invalid_citations": [c.chunk_id for c in citations if not c.valid],
            "refused": refused,
            "cost_cents": cost_cents,
        })

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# -- feedback -----------------------------------------------------------

class FeedbackRequest(BaseModel):
    message_id: int
    rating: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=2000)


@app.post("/api/feedback")
def feedback(body: FeedbackRequest, user: User = Depends(current_user)) -> dict[str, str]:
    if body.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be -1 or 1")
    accounts.add_feedback(body.message_id, user.id, body.rating, body.comment)
    return {"status": "recorded"}


@app.get("/api/admin/feedback")
def admin_feedback(_: User = Depends(admin_user)) -> dict[str, Any]:
    return accounts.feedback_dashboard()


@app.get("/api/sources")
def sources(_: User = Depends(current_user)) -> list[str]:
    with store.conn() as c, c.cursor() as cur:
        cur.execute("SELECT DISTINCT source FROM chunks ORDER BY 1")
        return [r["source"] for r in cur.fetchall()]


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        chunks = store.count()
    except Exception:  # noqa: BLE001
        chunks = -1
    return {"status": "ok" if chunks >= 0 else "degraded",
            "indexed_chunks": chunks, "llm_available": llm.available}


# -- static -------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(WEB_DIR / "admin.html")
