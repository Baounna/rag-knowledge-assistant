"""Ingestion pipeline: corpus directory -> chunk records.

    corpus/**  ->  parse  ->  chunk  ->  data/chunks.jsonl

Slice 1 stops at JSONL on disk. Slice 2 loads that JSONL into Postgres and
adds embeddings. Keeping the boundary here is deliberate: you can inspect,
diff, and eyeball your chunks before paying to embed a single one of them.
Most RAG bugs are visible at this stage and invisible after it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .chunker import Chunker
from .models import Chunk, Document
from .parsers import SUPPORTED_SUFFIXES, parse_file

_SLUG = re.compile(r"[^a-z0-9]+")


def _doc_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    slug = _SLUG.sub("-", str(rel).lower()).strip("-")
    return slug or path.stem.lower()


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    skipped: list[tuple[str, str]] = None  # (path, reason)

    def __post_init__(self) -> None:
        if self.skipped is None:
            self.skipped = []

    def report(self) -> str:
        lines = [
            f"documents parsed : {self.documents}",
            f"chunks produced  : {self.chunks}",
        ]
        if self.documents:
            lines.append(f"chunks/document  : {self.chunks / self.documents:.1f}")
        if self.skipped:
            lines.append(f"skipped          : {len(self.skipped)}")
            for path, reason in self.skipped[:10]:
                lines.append(f"  - {path}: {reason}")
        return "\n".join(lines)


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            if any(part.startswith(".") for part in path.parts):
                continue
            yield path


def ingest_corpus(
    corpus_dir: Path,
    output: Path | None = None,
    *,
    chunker: Chunker | None = None,
) -> tuple[list[Chunk], IngestStats]:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    chunker = chunker or Chunker()
    stats = IngestStats()
    all_chunks: list[Chunk] = []

    for path in iter_source_files(corpus_dir):
        try:
            doc: Document = parse_file(path, _doc_id(path, corpus_dir))
        except Exception as exc:  # a bad file must not kill the whole run
            stats.skipped.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue

        if not doc.blocks:
            stats.skipped.append((str(path), "no content extracted"))
            continue

        chunks = chunker.chunk(doc)
        if not chunks:
            stats.skipped.append((str(path), "produced no chunks"))
            continue

        all_chunks.extend(chunks)
        stats.documents += 1
        stats.chunks += len(chunks)

    if output:
        write_jsonl(all_chunks, Path(output))

    return all_chunks, stats


def write_jsonl(chunks: list[Chunk], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
