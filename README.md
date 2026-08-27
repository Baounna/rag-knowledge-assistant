# RAG-based Internal Knowledge Assistant

Answers natural-language questions over an internal document corpus, with
clickable citations, and refuses when the corpus doesn't support an answer.

Status: **slice 1 of 7 complete** (ingestion).

```
[x] 1. Ingestion    parse -> chunk -> chunk records          <- you are here
[ ] 2. Indexing     embeddings (pgvector) + BM25 (tsvector)
[ ] 3. Retrieval    hybrid + RRF -> rerank -> query rewriting
[ ] 4. Generation   citations + refusal + prompt caching
[ ] 5. Eval         recall@K, MRR, faithfulness -> make eval
[ ] 6. UI           streaming chat, clickable citations, feedback
[ ] 7. Ops          auth, rate limits, cost ceiling, deploy
```

## Quick start

```bash
make setup     # venv + dependencies + .env
make ingest    # corpus/ -> data/chunks.jsonl
make inspect   # same, but prints sample chunks to eyeball
make test      # chunker invariants
```

Drop your own documents into `corpus/` and re-run `make ingest`. Nothing in
the code needs to change:

```
corpus/
  markdown/   *.md          -- Markdown repo / exported docs
  pdf/        *.pdf         -- PDF folder
  notion/     */*.md        -- Notion Markdown export (page ids kept)
```

## Slice 1 — ingestion

```
corpus/**  ->  parse  ->  chunk  ->  data/chunks.jsonl
```

**Parsing** produces a `Document` of labelled `Block`s (heading / paragraph /
code / table / list) rather than one flat string. Structure has to survive
parsing, because chunking depends on it and nothing downstream can recover it.
`parsers/pdf.py` infers headings heuristically and says so in its docstring —
it is the weakest link in the pipeline, and the honest place to admit it.

**Chunking** follows a cut-point priority:

| | cut at | why |
|---|---|---|
| 1 | never inside code/tables | half a code sample is a wrong answer that still looks valid |
| 2 | heading boundary | sections are the natural unit of self-containment |
| 3 | paragraph boundary | |
| 4 | sentence boundary | a fact split in half is a fact the system has lost |
| 5 | word boundary | last resort, for unpunctuated PDF extraction |

Size (`target_words=350`) is a **budget, not a rule**: whole sentences are
added until the chunk approaches target, then it closes at the last complete
sentence. Chunks land anywhere from ~90 to ~500 words. The boundary wins.

**Self-containment** is the actual design goal, not any word count. Someone
reading one chunk with zero surrounding text must understand it. Two
mechanisms enforce it:

- **Heading trail** — every chunk carries its section path, rooted at the
  document title. `Chunk.embedding_text()` prepends it before indexing, so
  `"Set the timeout to 30 seconds."` is indexed as
  `"Handbook > Payments Service > Configuration | Set the timeout..."`.
  Without it, that chunk is unfindable by any question naming the subsystem.
- **Overlap** — each chunk replays the last ~2 sentences of its predecessor,
  so dangling references (*"this window"*, *"they"*) keep their referent.
  Capped at 25% of target: uncapped, a long unpunctuated sentence replays
  itself in full and every chunk grows without bound (see
  `test_respects_max_words_for_prose`).

Undersized chunks are repaired by a merge pass, **never** by skipping a
heading cut — skipping it produces chunks whose heading trail lies about half
their own content.

### Chunk record

```python
{
  "chunk_id": "markdown-expense-policy#c0001",   # the [source:chunk_id] cited
  "text": "...",                                  # what the LLM reads
  "heading_trail": ["Expense Policy", "Receipts"],
  "source": "Finance Handbook",
  "url": "https://...",                           # makes the citation clickable
  "author": "Finance Team",
  "doc_date": "2025-03-14",                       # powers retrieval filters
  "position": 1,
  "content_hash": "a3f1..."                       # skip re-embedding unchanged text
}
```

Every field earns its place downstream. Metadata dropped at ingestion cannot
be recovered later — no `url` means no clickable citation, and citations are
20% of the grade.

## Tuning

`make inspect` prints real chunks. Read them. Most retrieval bugs are visible
here and invisible after embedding.

```bash
python3 scripts/ingest.py --target 250 --overlap 1 --inspect 5
```

Watch two numbers in the report: the word-count spread (a chunk at the `min`
is probably a fragment) and `no heading trail` (should be 0%).

## Layout

```
app/ingest/
  models.py            Block / Document / Chunk
  chunker.py           the cut-point priority + merge pass
  parsers/             markdown.py, pdf.py, notion.py, dispatch
  pipeline.py          corpus -> chunks -> jsonl
scripts/ingest.py      CLI
tests/test_chunker.py  invariants (every test = one failure mode)
corpus/                your documents
data/chunks.jsonl      output
```
