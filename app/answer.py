"""Answer generation: citations, refusal, and verification of both.

Two ideas carry this module.

CITATIONS ARE VERIFIED, NOT TRUSTED. The prompt asks for `[chunk_id]` after
every claim, and then `validate_citations` checks in code that every cited id
was actually retrieved. A model can emit a citation that looks perfect and
refers to nothing -- that is the single most damaging failure a knowledge
assistant has, because it manufactures the appearance of evidence. Prompting
alone cannot rule it out; parsing the output can.

REFUSAL IS GATED TWICE. The prompt tells the model to refuse when the context
does not support an answer, and a retrieval-confidence threshold refuses
BEFORE the model is called at all. The second gate exists because a model
handed five irrelevant chunks will often find something to say about them.
Not calling the model is both cheaper and safer than asking it to be careful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .config import Settings, get_settings
from .llm import LLM
from .retrieval import RetrievalResult, RetrievedChunk

REFUSAL_TEXT = "I don't know based on internal docs."

ANSWER_SYSTEM = """You answer questions using ONLY the excerpts from internal \
company documents provided in each message.

CITATIONS
Every factual claim must be followed by the id of the excerpt it came from, in \
square brackets: [markdown-expense-policy#c0001]. Cite the id exactly as it \
appears. Multiple sources for one claim get multiple brackets. A sentence \
stating a fact with no citation is a defect, and so is a citation whose id was \
not in the excerpts you were given -- never invent, abbreviate, or reformat an \
id.

WHEN YOU CANNOT ANSWER
If the excerpts do not contain the answer, reply with exactly:
"I don't know based on internal docs."
and nothing else. Then, only if useful, add one sentence naming what document \
would need to exist.

Do this whenever the excerpts are about the right topic but do not state the \
specific fact asked for. Partial relevance is not an answer, and a plausible \
answer built from general knowledge is worse than no answer: the person asking \
cannot tell the difference, and will act on it. You have no knowledge of this \
company beyond the excerpts.

If the excerpts answer only part of the question, answer that part, cite it, \
and say plainly which part is not covered.

STYLE
Lead with the answer in the first sentence -- the reader wants the number, the \
deadline, or the rule, not a preamble. Keep it short. Quote exact figures, \
dates, and thresholds rather than paraphrasing them. If two excerpts conflict, \
say so and cite both rather than silently picking one."""

_CITATION = re.compile(r"\[([A-Za-z0-9_./#-]+?#c\d{4})\]")


@dataclass(slots=True)
class Citation:
    chunk_id: str
    source: str
    url: str | None
    valid: bool


@dataclass(slots=True)
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    chunks_used: list[RetrievedChunk] = field(default_factory=list)
    top_confidence: float = 0.0

    @property
    def invalid_citations(self) -> list[str]:
        return [c.chunk_id for c in self.citations if not c.valid]

    @property
    def uncited_claims(self) -> int:
        """Sentences that assert something with no citation attached.

        A blunt heuristic, deliberately: it is a signal for the eval report,
        not a gate. Short sentences and questions are ignored.
        """
        if self.refused:
            return 0
        count = 0
        for sentence in re.split(r"(?<=[.!?])\s+", self.text):
            s = sentence.strip()
            if len(s.split()) >= 6 and not s.endswith("?") and not _CITATION.search(s):
                count += 1
        return count


def validate_citations(text: str, chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    """Check every cited id against what was actually retrieved."""
    by_id = {c.chunk_id: c for c in chunks}
    seen: dict[str, Citation] = {}
    for chunk_id in _CITATION.findall(text):
        if chunk_id in seen:
            continue
        chunk = by_id.get(chunk_id)
        seen[chunk_id] = Citation(
            chunk_id=chunk_id,
            source=chunk.source if chunk else "",
            url=chunk.url if chunk else None,
            valid=chunk is not None,
        )
    return list(seen.values())


def link_citations(text: str, citations: Sequence[Citation]) -> str:
    """Rewrite `[id]` as a markdown link, for the UI.

    Only valid citations become links. An invalid one stays as plain text, so
    a fabricated reference is visibly not clickable rather than dressed up as
    a real source.
    """
    urls = {c.chunk_id: c.url for c in citations if c.valid and c.url}
    return _CITATION.sub(
        lambda m: f"[[{m.group(1)}]]({urls[m.group(1)]})" if m.group(1) in urls else m.group(0),
        text,
    )


class AnswerGenerator:
    def __init__(self, llm: LLM | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLM(self.settings)

    # -- prompt assembly -----------------------------------------------

    def _messages(
        self, question: str, result: RetrievalResult, history: Sequence[dict[str, str]]
    ) -> list[dict[str, Any]]:
        turns = [{"role": h["role"], "content": h["content"]} for h in list(history)[-6:]]
        # Retrieved context first with a cache breakpoint, question last.
        # Caching is a prefix match: putting the question above the context
        # would make every request a fresh prefix and cache nothing.
        return turns + [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Excerpts from internal documents:\n\n{result.context_block()}",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": f"Question: {question}"},
            ],
        }]

    def _refusal(self, question: str, result: RetrievalResult, reason: str) -> Answer:
        return Answer(
            question=question, text=REFUSAL_TEXT, refused=True, refusal_reason=reason,
            chunks_used=list(result.chunks), top_confidence=result.top_confidence,
        )

    def _gate(self, result: RetrievalResult) -> str | None:
        """Reasons to refuse before spending a model call."""
        if not result.chunks:
            return "retrieval returned nothing"
        if (self.settings.min_confidence > 0
                and result.top_confidence < self.settings.min_confidence):
            return (f"top confidence {result.top_confidence:.2f} below "
                    f"threshold {self.settings.min_confidence:.2f}")
        return None

    # -- generation ----------------------------------------------------

    def answer(
        self, question: str, result: RetrievalResult,
        history: Sequence[dict[str, str]] = (),
    ) -> Answer:
        gate = self._gate(result)
        if gate:
            return self._refusal(question, result, gate)
        if not self.llm.available:
            return self._refusal(question, result, "no API key configured")

        text = self.llm.complete_text(
            model=self.settings.model_answer,
            system=ANSWER_SYSTEM,
            messages=self._messages(question, result, history),
            max_tokens=1500,
        )
        return self._finish(question, text, result)

    def stream(
        self, question: str, result: RetrievalResult,
        history: Sequence[dict[str, str]] = (),
    ) -> Iterator[str]:
        gate = self._gate(result) or (None if self.llm.available else "no API key configured")
        if gate:
            yield REFUSAL_TEXT
            return
        yield from self.llm.stream_text(
            model=self.settings.model_answer,
            system=ANSWER_SYSTEM,
            messages=self._messages(question, result, history),
            max_tokens=1500,
        )

    def _finish(self, question: str, text: str, result: RetrievalResult) -> Answer:
        text = text.strip()
        refused = text.startswith(REFUSAL_TEXT) or text.lower().startswith("i don't know based on")
        return Answer(
            question=question,
            text=text,
            citations=validate_citations(text, result.chunks),
            refused=refused,
            refusal_reason="model declined: context did not support an answer" if refused else "",
            chunks_used=list(result.chunks),
            top_confidence=result.top_confidence,
        )
