"""Streamlit interface.

Talks to the application modules directly rather than over HTTP. Streamlit is
already a Python process with the code importable, so an internal API call
would add a network hop, a second service to run, and cookie handling, for no
benefit.

What that costs, honestly: there is no REST API for anything else to consume.
If a Slack bot or a second client is ever needed, the FastAPI layer comes back
(it is in git history) -- the modules it called are untouched.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

import streamlit as st

from app.accounts import Accounts, User
from app.answer import AnswerGenerator, link_citations, validate_citations
from app.config import get_settings
from app.llm import LLM
from app.retrieval import RetrievalResult, Retriever
from app.store import Filters, Store

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="centered")


# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def services() -> dict[str, Any]:
    """Built once per server process.

    Without the cache Streamlit re-runs this file top to bottom on every
    interaction, which would reload the embedding model and reconnect to
    Postgres on every keystroke.
    """
    settings = get_settings()
    store = Store(settings)
    store.init_schema()
    accounts = Accounts(settings)
    accounts.init_schema()
    llm = LLM(settings)
    return {
        "settings": settings,
        "store": store,
        "accounts": accounts,
        "llm": llm,
        "retriever": Retriever(store=store, llm=llm, settings=settings),
        "generator": AnswerGenerator(llm=llm, settings=settings),
    }


@st.cache_data(ttl=300, show_spinner=False)
def source_names() -> list[str]:
    with services()["store"].conn() as c, c.cursor() as cur:
        cur.execute("SELECT DISTINCT source FROM chunks ORDER BY 1")
        return [r["source"] for r in cur.fetchall()]


def user() -> User | None:
    return st.session_state.get("user")


# ---------------------------------------------------------------------------
# Sign in
# ---------------------------------------------------------------------------


def login_screen() -> None:
    accounts = services()["accounts"]
    st.title("📚 Internal Knowledge Assistant")
    st.caption("Answers from your internal documents, with citations you can click.")

    tab_in, tab_up = st.tabs(["Sign in", "Create account"])

    with tab_in:
        with st.form("signin"):
            email = st.text_input("Email", key="in_email")
            password = st.text_input("Password", type="password", key="in_pw")
            if st.form_submit_button("Sign in", type="primary"):
                found = accounts.authenticate(email, password)
                if found:
                    st.session_state.user = found
                    st.rerun()
                # One message for both wrong-email and wrong-password:
                # distinguishing them tells an attacker which emails exist.
                st.error("Invalid email or password.")

    with tab_up:
        if not services()["settings"].allow_signup:
            st.info("Signup is disabled. Ask an administrator for an account.")
            return
        with st.form("signup"):
            email = st.text_input("Email", key="up_email")
            password = st.text_input("Password (8+ characters)", type="password", key="up_pw")
            if st.form_submit_button("Create account"):
                try:
                    st.session_state.user = accounts.create_user(email, password)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("That email is already registered.")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_CITATION = re.compile(r"\[([A-Za-z0-9_./#-]+?#c\d{4})\]")


def render_answer(text: str, citations: list[dict[str, Any]]) -> None:
    """Render `[chunk_id]` as a markdown link.

    Only citations the validator marked valid become links. A fabricated id
    stays as plain code, so a made-up reference is visibly not a source rather
    than being dressed up as one.
    """
    by_id = {c["chunk_id"]: c for c in citations}

    def replace(match: re.Match[str]) -> str:
        cid = match.group(1)
        c = by_id.get(cid)
        if c and c["valid"] and c["url"]:
            return f"[[{c['source']}]]({c['url']})"
        return f"`[{cid}]`"

    st.markdown(_CITATION.sub(replace, text))

    invalid = [c["chunk_id"] for c in citations if not c["valid"]]
    if invalid:
        st.warning(
            f"{len(invalid)} citation(s) refer to passages that were not retrieved: "
            f"{', '.join(invalid)}. Treat those claims as unsupported.",
            icon="⚠️",
        )


def render_sources(result: RetrievalResult) -> None:
    with st.expander(f"{len(result.chunks)} passages retrieved", expanded=False):
        if len(result.queries_used) > 1:
            st.caption("Searched: " + " · ".join(result.queries_used))
        for c in result.chunks:
            found = "+".join(c.found_by) or "—"
            score = f" · rerank {c.rerank_score:.0f}/10" if c.rerank_score is not None else ""
            st.markdown(f"**{c.source}** — {' > '.join(c.heading_trail)}")
            st.caption(f"`{c.chunk_id}` · found by {found}{score}")
            st.text(c.text[:400] + ("…" if len(c.text) > 400 else ""))
            if c.url:
                st.caption(f"[open source]({c.url})")
            st.divider()


def feedback_widget(message_id: int, existing: int | None) -> None:
    accounts, me = services()["accounts"], user()
    key = f"fb_{message_id}"
    cols = st.columns([1, 1, 6])
    if cols[0].button("👍", key=key + "_up", type="primary" if existing == 1 else "secondary"):
        accounts.add_feedback(message_id, me.id, 1)
        st.rerun()
    if cols[1].button("👎", key=key + "_down", type="primary" if existing == -1 else "secondary"):
        st.session_state[key + "_comment_open"] = True
    if st.session_state.get(key + "_comment_open"):
        with st.form(key + "_form"):
            comment = st.text_input("What was wrong with this answer? (optional)")
            if st.form_submit_button("Send"):
                accounts.add_feedback(message_id, me.id, -1, comment)
                st.session_state[key + "_comment_open"] = False
                st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def sidebar() -> None:
    s, accounts, me = services(), services()["accounts"], user()
    with st.sidebar:
        st.markdown(f"**{me.email}**")
        limits = accounts.check_limits(me)
        st.progress(
            min(1.0, limits.questions_used / max(limits.questions_limit, 1)),
            text=f"{limits.questions_used}/{limits.questions_limit} questions today",
        )
        if s["llm"].backend.cents_per_1k_tokens > 0:
            st.caption(f"Spend today: ${limits.cents_used / 100:.2f} "
                       f"of ${limits.cents_limit / 100:.2f}")
        else:
            st.caption(f"Local model ({s['llm'].model_for('answer')}) — no cost")

        st.divider()
        if st.button("＋ New conversation", use_container_width=True):
            st.session_state.conversation_id = None
            st.rerun()

        st.caption("RECENT")
        for conv in accounts.list_conversations(me.id, limit=25):
            if st.button(conv["title"][:38], key=f"c{conv['id']}", use_container_width=True):
                st.session_state.conversation_id = conv["id"]
                st.rerun()

        st.divider()
        st.session_state.source_filter = st.selectbox(
            "Search in", ["All sources", *source_names()],
            help="Restrict retrieval to one document source.",
        )
        if me.is_admin:
            st.session_state.show_admin = st.toggle(
                "Feedback dashboard", value=st.session_state.get("show_admin", False))
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        if not s["llm"].available:
            st.error("No model configured — every answer will be a refusal.", icon="🚫")


def replay_history() -> None:
    accounts, me = services()["accounts"], user()
    cid = st.session_state.get("conversation_id")
    if not cid:
        return
    for m in accounts.get_messages(cid, me.id):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                render_answer(m["content"], m["citations"] or [])
                feedback_widget(m["id"], m["rating"])
            else:
                st.markdown(m["content"])


def stream_answer(question: str) -> None:
    s = services()
    accounts, me = s["accounts"], user()

    limits = accounts.check_limits(me)
    if not limits.allowed:
        st.error(f"Limit reached: {limits.reason}", icon="🛑")
        return

    cid = st.session_state.get("conversation_id") or accounts.create_conversation(me.id)
    st.session_state.conversation_id = cid

    history = [{"role": m["role"], "content": m["content"]}
               for m in accounts.get_messages(cid, me.id)][-6:]
    accounts.add_message(cid, "user", question)
    accounts.set_title_from_first_question(cid, question)

    with st.chat_message("user"):
        st.markdown(question)

    chosen = st.session_state.get("source_filter", "All sources")
    filters = None if chosen == "All sources" else Filters(sources=[chosen])
    before = s["llm"].usage.input_tokens + s["llm"].usage.output_tokens

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            result = s["retriever"].retrieve(
                question, filters=filters, history=history,
                use_rewrite=s["settings"].enable_rewrite,
                use_rerank=s["settings"].enable_rerank,
            )
        if result.notes:
            st.caption(" · ".join(result.notes))

        def tokens() -> Iterator[str]:
            yield from s["generator"].stream(question, result, history)

        text = st.write_stream(tokens())
        if isinstance(text, list):
            text = "".join(text)
        text = (text or "").strip()

        # Validate on the COMPLETE text: mid-stream, a half-written
        # "[markdown-exp" would look like a fabricated citation.
        citations = validate_citations(text, result.chunks)
        payload = [{"chunk_id": c.chunk_id, "source": c.source,
                    "url": c.url, "valid": c.valid} for c in citations]
        refused = text.lower().startswith("i don't know based on")

        st.empty()
        render_answer(text, payload)
        render_sources(result)

        tokens_used = (s["llm"].usage.input_tokens + s["llm"].usage.output_tokens) - before
        cost = max(0, round(tokens_used * s["llm"].backend.cents_per_1k_tokens / 1000))
        message_id = accounts.add_message(
            cid, "assistant", text, citations=payload,
            retrieval={"chunk_ids": [c.chunk_id for c in result.chunks],
                       "queries": result.queries_used},
            refused=refused, cost_cents=cost,
        )
        feedback_widget(message_id, None)


def admin_dashboard() -> None:
    board = services()["accounts"].feedback_dashboard()
    counts, answers = board["counts"], board["answers"]
    st.subheader("Feedback dashboard")
    st.caption("Each rating is joined to the passages that produced the answer. "
               "A thumbs-down plus its chunk ids is a debuggable retrieval failure; "
               "a thumbs-down alone is a mood.")
    cols = st.columns(4)
    cols[0].metric("Ratings", counts["total"])
    cols[1].metric("Negative", counts["down"],
                   f"{(counts['down'] / counts['total'] * 100):.0f}%" if counts["total"] else "—")
    cols[2].metric("Answers", answers["n"])
    cols[3].metric("Refusal rate",
                   f"{(answers['refusals'] / answers['n'] * 100):.0f}%" if answers["n"] else "—")

    if not board["items"]:
        st.info("No feedback yet.")
        return
    for item in board["items"]:
        icon = "👍" if item["rating"] == 1 else "👎"
        with st.expander(f"{icon} {item['question'] or '(question missing)'}"):
            st.markdown(item["answer"])
            if item["comment"]:
                st.info(item["comment"])
            ids = (item["retrieval"] or {}).get("chunk_ids", [])
            st.caption(f"{item['email']} · {item['created_at']:%Y-%m-%d %H:%M} · "
                       f"retrieved: {', '.join(ids) or '—'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if user() is None:
        login_screen()
        return

    sidebar()
    if st.session_state.get("show_admin"):
        admin_dashboard()
        return

    st.title("📚 Knowledge Assistant")
    if services()["store"].count() == 0:
        st.warning("The index is empty. Run `make ingest && make index`.", icon="📭")

    replay_history()
    if question := st.chat_input("Ask about internal docs…"):
        stream_answer(question)


main()
