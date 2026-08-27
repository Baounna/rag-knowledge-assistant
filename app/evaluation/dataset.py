"""The evaluation set.

A question is labelled with the chunk ids that genuinely answer it. Those
labels are the ground truth every retrieval metric is computed against, which
makes this file the most important data in the project: metrics computed
against sloppy labels are worse than no metrics, because they look like
evidence.

Two rules for writing questions, both learned the hard way:

1. Phrase them the way a USER would, not the way the document does. A question
   built by copying a sentence out of the corpus measures string matching and
   flatters every retriever.
2. Include unanswerable questions. A system that never refuses scores well on
   answerable questions alone while being unsafe in production. `answerable:
   false` questions are how refusal gets measured at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(slots=True)
class EvalQuestion:
    id: str
    question: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    expected_answer: str = ""
    must_include: list[str] = field(default_factory=list)  # substrings that must appear
    answerable: bool = True
    tags: list[str] = field(default_factory=list)
    note: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.question.strip():
            problems.append(f"{self.id}: empty question")
        if self.answerable and not self.relevant_chunk_ids:
            problems.append(f"{self.id}: answerable but no relevant_chunk_ids -- unlabelled")
        if not self.answerable and self.relevant_chunk_ids:
            problems.append(f"{self.id}: marked unanswerable but has relevant chunks")
        if not self.answerable and self.must_include:
            problems.append(f"{self.id}: unanswerable questions cannot require content")
        return problems


def load_questions(path: Path | str) -> list[EvalQuestion]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no eval set at {path}. Write one: 50-100 questions about YOUR corpus, "
            f"each labelled with the chunk ids that answer it."
        )
    questions = [
        EvalQuestion(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    problems = [p for q in questions for p in q.validate()]
    if problems:
        raise ValueError("eval set has problems:\n  " + "\n  ".join(problems))
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("eval set has duplicate ids")
    return questions


def save_questions(questions: list[EvalQuestion], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def coverage(questions: list[EvalQuestion], indexed_chunk_ids: set[str]) -> dict[str, Any]:
    """Health check on the eval set itself.

    Catches the two ways an eval set silently rots: labels pointing at chunk
    ids that no longer exist (the corpus was re-chunked), and having no
    unanswerable questions at all (refusal is never exercised).
    """
    labelled = {cid for q in questions for cid in q.relevant_chunk_ids}
    missing = sorted(labelled - indexed_chunk_ids)
    unanswerable = sum(1 for q in questions if not q.answerable)
    return {
        "questions": len(questions),
        "answerable": len(questions) - unanswerable,
        "unanswerable": unanswerable,
        "labelled_chunks": len(labelled),
        "indexed_chunks": len(indexed_chunk_ids),
        "corpus_covered": len(labelled & indexed_chunk_ids) / max(len(indexed_chunk_ids), 1),
        "stale_labels": missing,
    }


def iter_batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
