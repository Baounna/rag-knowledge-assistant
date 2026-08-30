#!/usr/bin/env python3
"""Draft candidate evaluation questions from the indexed corpus.

    python3 scripts/draft_questions.py --count 60
    # review eval/questions.draft.jsonl, then:
    python3 scripts/draft_questions.py --review

THIS PRODUCES A DRAFT, NOT AN EVAL SET. Every question needs a human to
confirm that the labelled chunk really answers it. A generated label that is
merely plausible turns the whole harness into theatre -- it would report
confident numbers computed against wrong ground truth, which is worse than
having no numbers at all.

Two things the generator is pushed hard on, because they are what separates a
useful eval set from a flattering one:

  * Questions must be phrased the way a USER would ask, not the way the
    document is written. A question built by copying a sentence out of the
    corpus measures string overlap and flatters every retriever.

  * Some questions must be UNANSWERABLE -- plausible for this company, absent
    from these documents. Without them, refusal is never exercised, and a
    system that answers everything confidently scores perfectly.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.evaluation import EvalQuestion, load_questions, save_questions  # noqa: E402
from app.llm import LLM  # noqa: E402
from app.store import Store  # noqa: E402

FACT = re.compile(r"\$[\d,]+|\b\d+\s*(?:days?|hours?|weeks?|months?|%|per cent)\b|\b\d{1,2}:\d{2}\b",
                  re.IGNORECASE)


def is_prose(text: str) -> bool:
    """Reject chunks that are mostly markup rather than sentences.

    A mermaid diagram or a wide table contains facts a human can read but no
    prose for a question to be grounded in -- the first trial produced a
    question about a flow chart whose answer was not visibly in the passage.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    markup = sum(1 for l in lines
                 if l.lstrip().startswith(("|", ">", "```", "-->", "graph ", "flowchart"))
                 or "-->" in l or l.count("|") >= 3)
    if markup / len(lines) > 0.3:
        return False
    # Needs enough running text for a question to have an answer in it.
    sentences = [x for x in re.split(r"(?<=[.!?])\s", text) if len(x.split()) >= 6]
    return len(sentences) >= 3

DRAFT_SYSTEM = """You write evaluation questions for a document retrieval \
system, from passages of a company handbook.

For each passage, write ONE question that the passage answers completely.

The questions are used to measure retrieval, so how you phrase them decides \
whether the measurement is worth anything:

- Write as an EMPLOYEE would ask, not as the document is written. If the \
passage says "reimbursement shall be submitted within 30 days of \
incurrence", ask "how long do I have to claim expenses?" -- do not reuse \
"reimbursement", "submitted" or "incurrence". A question that shares its \
wording with the passage measures string overlap, not retrieval.
- Ask about ONE specific fact the passage actually states: a number, a \
deadline, a rule, a procedure. Not "what does this section cover".
- The answer must be in the passage. If the passage states no specific fact, \
say so by setting `usable` to false rather than inventing a question.
- Vary the form across the batch. Do not write "how long…?" for every \
passage. Mix: direct lookups ("what is the mileage rate?"), situational ("I \
paid for a client dinner, can I claim it?"), small inferences ("I have a \
receipt for $18, do I need to keep it?"), yes/no ("do contractors get the \
learning stipend?"), and who/where ("who approves a job offer?").
- The question and the answer must match in KIND. If the answer is a list of \
topics, do not ask "how long"; ask "what does X cover". A mismatched pair is \
unusable even when both halves are individually true.

Also give `answer` (the fact, briefly) and `must_include` (1-2 short strings \
that MUST appear in any correct answer -- usually the number or the key \
term, exactly as a correct answer would write it)."""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "passage_id": {"type": "string"},
                    "usable": {"type": "boolean"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "must_include": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["passage_id", "usable", "question", "answer", "must_include"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

UNANSWERABLE_SYSTEM = """You write questions that a company's internal \
knowledge assistant should REFUSE to answer.

You are given the list of topics its documents actually cover. Write \
questions that an employee could plausibly ask but that these documents do \
NOT answer.

The useful ones sit close to the corpus without being in it -- that is what \
catches a system substituting a neighbouring fact for the one asked about. If \
the documents cover expense deadlines but not procurement approval limits, \
"what's the approval limit for a new laptop purchase?" is an excellent test \
and "what is the capital of France?" is a worthless one.

Avoid anything the listed topics plainly cover."""

UNANSWERABLE_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}


def pick_chunks(store: Store, count: int, seed: int = 7) -> list[dict]:
    """Stratified sample: spread across sources, prefer chunks stating facts.

    Sampling uniformly would over-weight whichever source happens to be
    largest, and would mostly pick prose chunks that state nothing checkable.
    """
    with store.conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, text, heading_trail, source, url
            FROM chunks WHERE length(text) > 400
        """)
        rows = [dict(r) for r in cur.fetchall() if is_prose(r["text"])]

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    rng = random.Random(seed)
    for group in by_source.values():
        # Chunks containing a number or a deadline are the ones a question can
        # have a checkable answer to.
        group.sort(key=lambda r: (-len(FACT.findall(r["text"])), rng.random()))

    picked: list[dict] = []
    sources = sorted(by_source)
    i = 0
    while len(picked) < count and any(by_source[s] for s in sources):
        src = sources[i % len(sources)]
        if by_source[src]:
            picked.append(by_source[src].pop(0))
        i += 1
    return picked


def draft(llm: LLM, chunks: list[dict], batch: int) -> list[EvalQuestion]:
    out: list[EvalQuestion] = []
    for start in range(0, len(chunks), batch):
        group = chunks[start : start + batch]
        listing = "\n\n".join(
            f"passage_id: {c['chunk_id']}\n"
            f"section: {' > '.join(c['heading_trail'])}\n"
            f"{c['text'][:1400]}"
            for c in group
        )
        print(f"  drafting {start + 1}-{start + len(group)} of {len(chunks)}…",
              end="", flush=True)
        try:
            result = llm.complete_json(
                model=llm.model_for("rewrite"),
                system=DRAFT_SYSTEM,
                messages=[{"role": "user", "content": listing}],
                schema=DRAFT_SCHEMA,
                max_tokens=2000,
            )
        except Exception as exc:  # noqa: BLE001
            print(f" failed ({type(exc).__name__})")
            continue

        by_id = {c["chunk_id"]: c for c in group}
        kept = 0
        for item in result.get("questions", []):
            if not item.get("usable") or item["passage_id"] not in by_id:
                continue
            out.append(EvalQuestion(
                id=f"q{len(out) + 1:03d}",
                question=item["question"].strip(),
                relevant_chunk_ids=[item["passage_id"]],
                expected_answer=item.get("answer", "").strip(),
                must_include=[s for s in item.get("must_include", []) if s.strip()][:2],
                answerable=True,
                tags=["drafted", "needs-review"],
                note=f"section: {' > '.join(by_id[item['passage_id']]['heading_trail'][:3])}",
            ))
            kept += 1
        print(f" {kept} usable")
    return out


def draft_unanswerable(llm: LLM, store: Store, count: int) -> list[EvalQuestion]:
    with store.conn() as c, c.cursor() as cur:
        cur.execute("SELECT DISTINCT heading_trail[2] h FROM chunks "
                    "WHERE array_length(heading_trail,1) > 1 LIMIT 60")
        topics = sorted({r["h"] for r in cur.fetchall() if r["h"]})
    # Over-generate: the model is guessing what the corpus lacks from a list
    # of topic names, and it guesses badly. Generating a surplus and keeping
    # only the ones retrieval cannot answer turns a guess into a measurement.
    wanted = count
    count = count * 3
    print(f"  drafting {count} unanswerable candidates…", end="", flush=True)
    try:
        result = llm.complete_json(
            model=llm.model_for("rewrite"),
            system=UNANSWERABLE_SYSTEM,
            messages=[{"role": "user", "content":
                       f"Write {count} questions.\n\nTopics the documents DO cover:\n"
                       + "\n".join(f"- {t}" for t in topics[:50])}],
            schema=UNANSWERABLE_SCHEMA,
            max_tokens=900,
        )
    except Exception as exc:  # noqa: BLE001
        print(f" failed ({type(exc).__name__})")
        return []
    qs = [q.strip() for q in result.get("questions", []) if q.strip()][:count]
    print(f" {len(qs)} candidates")

    # A generated "unanswerable" question is a GUESS about what the corpus
    # lacks, and the model guesses badly -- it proposed "how do I change my
    # medical coverage?" for a corpus that documents benefits in detail.
    # Labelling an answerable question unanswerable penalises the system for
    # being right, so every candidate is checked against retrieval and the
    # suspicious ones are flagged rather than silently trusted.
    scored = []
    for q in qs:
        hits = store.lexical_search(q, k=1)
        scored.append((hits[0].score if hits else 0.0, hits[0] if hits else None, q))
    scored.sort(key=lambda t: t[0])          # weakest retrieval first = most likely absent

    out: list[EvalQuestion] = []
    for i, (score, top, q) in enumerate(scored[:wanted]):
        tags = ["drafted", "needs-review", "unanswerable"]
        note = f"best retrieval score {score:.1f} (low = probably genuinely absent)"
        if score > 3.0:
            tags.append("SUSPECT")
            note = (f"retrieval found a strong match ({score:.1f}): {top.source} > "
                    f"{' > '.join(top.heading_trail[1:3])}. CHECK whether the corpus "
                    f"answers this -- if it does, DELETE this question.")
        out.append(EvalQuestion(id=f"u{i + 1:03d}", question=q, relevant_chunk_ids=[],
                                answerable=False, tags=tags, note=note))
    flagged = sum(1 for q in out if "SUSPECT" in q.tags)
    print(f"    kept {len(out)} of {len(qs)} (weakest retrieval)"
          + (f", {flagged} still look answerable -- flagged SUSPECT" if flagged else ""))
    return out


def review(path: Path, store: Store) -> int:
    """Print each question beside the passage it is labelled against."""
    questions = load_questions(path)
    with store.conn() as c, c.cursor() as cur:
        cur.execute("SELECT chunk_id, text, heading_trail, source FROM chunks")
        chunks = {r["chunk_id"]: dict(r) for r in cur.fetchall()}

    for q in questions:
        print("=" * 78)
        flag = "  <-- SUSPECT" if "SUSPECT" in q.tags else ""
        print(f"{q.id}  {'ANSWERABLE' if q.answerable else 'UNANSWERABLE'}{flag}")
        if q.note:
            print(f"  note: {q.note}")
        print(f"  Q: {q.question}")
        if q.expected_answer:
            print(f"  A: {q.expected_answer}")
        if q.must_include:
            print(f"  must contain: {q.must_include}")
        for cid in q.relevant_chunk_ids:
            chunk = chunks.get(cid)
            if not chunk:
                print(f"  !! label {cid} is not in the index")
                continue
            print(f"  label: {cid}  ({chunk['source']} > "
                  f"{' > '.join(chunk['heading_trail'][1:3])})")
            print(f"  passage: {' '.join(chunk['text'].split())[:400]}")
        print("  -> does that passage really answer that question? "
              "edit or delete the line in the file if not.")
    print("=" * 78)
    print(f"{len(questions)} questions. When they are correct:")
    print(f"  mv {path} eval/questions.jsonl && make eval")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=60, help="answerable questions to draft")
    ap.add_argument("--unanswerable", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4, help="passages per model call")
    ap.add_argument("--out", type=Path, default=Path("eval/questions.draft.jsonl"))
    ap.add_argument("--review", action="store_true", help="print the draft for checking")
    args = ap.parse_args()

    settings = get_settings()
    store = Store(settings)
    store.init_schema()

    if args.review:
        return review(args.out, store)

    llm = LLM(settings)
    if not llm.available:
        print("no model available -- set ANTHROPIC_API_KEY or LLM_PROVIDER=ollama")
        return 1
    if store.count() == 0:
        print("index is empty -- run `make ingest && make index`")
        return 1

    calls = -(-args.count // args.batch) + 1
    per_call = 60 if llm.backend.name == "ollama" else 4
    print(f"Drafting from {store.count()} indexed chunks using {llm.backend.name}")
    print(f"  ~{calls} model calls, roughly {calls * per_call / 60:.0f} minutes\n")

    chunks = pick_chunks(store, args.count)
    questions = draft(llm, chunks, args.batch)
    questions += draft_unanswerable(llm, store, args.unanswerable)

    save_questions(questions, args.out)
    answerable = sum(1 for q in questions if q.answerable)
    print(f"\n  {len(questions)} drafted ({answerable} answerable, "
          f"{len(questions) - answerable} unanswerable) -> {args.out}")
    print(f"\nNEXT: every label needs checking. Run:")
    print(f"  python3 scripts/draft_questions.py --review | less")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
