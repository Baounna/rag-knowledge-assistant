# RAG-based Internal Knowledge Assistant

Answers natural-language questions over an internal document corpus, with
clickable citations, and refuses when the corpus doesn't support an answer.

Status: **all 7 slices built.** Architecture: [ARCHITECTURE.md](ARCHITECTURE.md) ·
Deployment: [DEPLOY.md](DEPLOY.md)

```
[x] 1. Ingestion    parse -> chunk -> chunk records
[x] 2. Indexing     embeddings (pgvector) + BM25 (pg_search)
[x] 3. Retrieval    hybrid + RRF -> rerank -> query rewriting
[x] 4. Generation   citations + refusal + prompt caching
[x] 5. Eval         recall@K, MRR, faithfulness -> make eval
[x] 6. UI           Streamlit: streaming, clickable citations, feedback
[x] 7. Ops          auth, rate limits, cost ceiling, deploy
```

## Quick start

Everything runs in Docker. Nothing is installed on your machine — no Python,
no Postgres, no model runtime.

```bash
make up-local     # database + app + local LLM   (free, no API key)
make ingest
make index
# open http://localhost:8502
```

Or with a Claude key in `.env` (faster, better answers, ~2c per question):

```bash
make up           # database + app
make ingest && make index
```

Other commands:

```bash
make ps       # what is running and what it is using
make logs     # follow the app log
make test     # 95 tests
make eval     # retrieval metrics across 6 configurations
make down     # stop everything, keep the data
make nuke     # delete everything this project created (~15GB back)
```

Drop your own documents into `corpus/` and re-run `make ingest && make index`.
Nothing in the code changes:

```
corpus/
  markdown/   *.md          -- Markdown repo / exported docs
  pdf/        *.pdf         -- PDF folder
  notion/     */*.md        -- Notion Markdown export (page ids kept)
```

### What lives where

| | where it runs | disk |
|---|---|---|
| App (Streamlit UI) | `rag-app` container | ~1 GB image |
| Postgres + pgvector + BM25 | `rag-db` container | ~1.4 GB image |
| Local LLM (optional) | `rag-ollama` container | ~7 GB image + ~5 GB model |
| Your documents | `corpus/`, mounted read-only | yours |

`make local-setup` still exists if you would rather run it from a host venv,
but it is optional and not the supported path.

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


## Slice 2 — indexing

One table, two indexes over the same rows:

| arm | index | finds |
|---|---|---|
| semantic | `VECTOR(384)` + HNSW, cosine | *meaning* — "money back for a work trip" -> the expense policy |
| lexical | `pg_search` BM25 (Tantivy) | *exact words* — `/var/log/db`, `--force`, error codes |

They fail on opposite inputs, which is the entire argument for hybrid
retrieval. Measured on this corpus:

```
query: "how do I get my money back for a work trip?"     (0 shared words)
  vector  -> expense policy at #1, #2      correct
  lexical -> unrelated PDF chunks at #2,#3  wrong

query: "/var/log/db"
  vector  -> correct chunk at #2, behind an unrelated PDF chunk
  lexical -> correct chunk at #1
```

Both indexes are built over `index_text` = heading trail + chunk body, never
the bare text.

### Design decisions worth defending

**One database.** ParadeDB is Postgres with `pgvector` *and* `pg_search`
(Tantivy embedded in Postgres — so the lexical arm is genuine BM25, which is
what the subject asks for). Vectors, lexical index, chat history, and feedback
live in one datastore. `Store.init_schema` falls back to Postgres full-text
search if `pg_search` is missing, and reports which backend actually ran.

**Cosine, not L2.** Embedding models are trained with cosine similarity; L2 on
unnormalised vectors also ranks by magnitude.

**English stemming on the BM25 index.** Without it, `deadline` does not match
`Deadlines` — and the miss is completely silent.

**Asymmetric embedding.** Questions and passages get different prefixes
(`embed_query` vs `embed_documents`). Using the wrong side degrades every
score with no error.

**Never feed user input to a query parser.** Lexical search goes through
`paradedb.match()`, which tokenises. The bare `@@@ 'text'` form parses Tantivy
query syntax, so a user typing `--force` crashed the endpoint
(`test_lexical_survives_adversarial_input`).

**Incremental indexing.** `content_hash` means a re-ingest only re-embeds
chunks whose text changed.

### Swapping the embedding model

`EMBEDDING_PROVIDER` accepts `fastembed` (default, local), `ollama`, `voyage`,
or `openai`. Dimensions are a property of the model, so changing it means
`make reindex` — the vector column has a fixed width and will reject a
mismatch at insert time, which is the right time to find out.


## Known limits (read before slice 3)

Honest list of what is *not* solved yet. Each one is a real behaviour of the
current code, verified rather than assumed.

**HNSW is bypassed when a filter is applied.** `EXPLAIN` on a filtered vector
query shows `Index Scan using chunks_source_idx` + `Sort` — Postgres filters
first, then computes exact distances over what survives. That is *correct*
and, on a small corpus, faster than the approximate index. At corpus scale
with a selective filter it becomes a full distance scan over the filtered set.
The lever is pgvector 0.8's `SET hnsw.iterative_scan = relaxed_order`, and it
should be turned on only once `make eval` can show it is needed.

**A new connection per query.** `Store` opens a connection for every call — six
call sites. Fine for CLI use, wrong for the API in slice 6: connection setup
plus auth on every request. Slice 6 adds `psycopg_pool`.

**No hybrid fusion yet.** `vector_search` and `lexical_search` return two
separate ranked lists. Merging them (RRF) is slice 3.

**PDF heading inference is a heuristic.** `parsers/pdf.py` guesses headings
from line shape. It reads table rows in the subject PDF as headings. A
layout-aware parser (PyMuPDF font sizes, or `unstructured`) is the upgrade if
the corpus is PDF-heavy.

**No `make eval`.** Every tuning decision so far — chunk size, top-K, stemming,
model choice — is currently justified by reasoning and spot checks, not by a
metric. That is exactly the gap slice 5 closes, and it is why nothing has been
tuned aggressively yet.


## Slices 3-5 — retrieval, generation, evaluation

### Retrieval pipeline

```
question -> rewrite (Haiku) -> vector + BM25 -> RRF -> rerank (Haiku) -> top N
```

Every stage is switchable, because `make eval` has to answer "did the
reranker help?" with a number. Every stage records the chunk ids it produced,
so a failure can be attributed to the stage that caused it.

**RRF over score normalisation.** Cosine sits in [0,1] and clusters tightly
(0.66 vs 0.50 here); BM25 is unbounded and corpus-dependent (4.72 vs 0.98 on
the same corpus). Any normalisation onto a shared scale is a hyper-parameter
that drifts with the corpus. RRF discards the scores and fuses ranks:
`score(d) = Σ weight / (k + rank)`, `k=60` from Cormack et al. Verified on the
sample corpus: fusion promoted a chunk to #1 that *neither* index ranked
first.

**All LLM stages degrade to no-ops without a key** — hybrid retrieval still
runs, and `make eval` still reports retrieval metrics.

### Generation

**Citations are verified, not trusted.** `validate_citations()` parses every
`[chunk_id]` out of the answer and checks it against what was actually
retrieved. A fabricated id that looks perfect is the most damaging failure a
knowledge assistant has, because it manufactures the appearance of evidence —
and prompting cannot rule it out. Invalid citations are also left unlinked in
the UI, so a fake reference is visibly not clickable.

**Refusal is gated twice.** The prompt instructs refusal when context is
insufficient; `MIN_CONFIDENCE` refuses *before* the model is called at all. A
model handed five irrelevant chunks will usually find something to say about
them — not calling it is cheaper and safer.

**Prompt caching.** Retrieved context carries the cache breakpoint and the
question comes after it. Caching is a prefix match, so the reverse order
caches nothing, silently. `Usage.cache_hit_rate` exists to verify it rather
than assume it.

### Evaluation

```bash
make eval        # retrieval-only configs: fast, free, no model calls
make eval-llm    # + rewrite/rerank configs (1-2 model calls per question)
make eval-full   # + answers, citations, refusal, LLM judge
```

`make eval` deliberately runs only the configurations that touch no model.
The rewrite and rerank configs cost one or two model calls **per question**:
seconds on a hosted API, but roughly a minute each on a local CPU model — so
a command advertised as fast and free would quietly become an hour of
inference. `--with-llm` opts in, and the harness prints a call count and time
estimate before it starts.

Metrics split deliberately:

| deterministic (free, exact, every commit) | judged (costs money, varies) |
|---|---|
| recall@K, MRR, precision@K, hit@K | faithfulness |
| citation validity, groundedness | answer relevance |
| refusal accuracy (both directions) | |
| required-content coverage | |

"Did the model cite a chunk that doesn't exist?" is a parsing problem, not a
judgement call. Treating it as one makes it free and exact.

**The harness refuses to flatter you.** It warns when `RETRIEVAL_TOP_K`
reaches the corpus size (recall becomes 1.0 by construction), when labels
point at chunks that no longer exist after re-chunking, when there are fewer
than 50 questions, when no unanswerable questions exist, and when no model is
available so several configurations are secretly identical.

### Measured

58 questions (49 answerable, 9 unanswerable) over 1,997 chunks:

| config | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| lexical-only | 0.776 | 0.898 | 0.939 | 0.709 |
| vector-only | 0.694 | 0.796 | 0.878 | 0.627 |
| **hybrid (RRF)** | **0.796** | **0.918** | **0.939** | **0.713** |

Hybrid wins on every column, which is the result RRF predicts — but read the
gap honestly: it beats lexical alone by only 2 points of recall@1. Lexical
also beats vector by 8 points, and that ordering deserves a caveat rather than
a victory lap: **the questions were drafted by a model that had read the
passages**, so despite being instructed to avoid the document's wording, some
vocabulary leaks through and favours BM25. A question set written by hand from
memory would likely narrow that gap.

Neither the reranker nor query rewriting is in these numbers — both are one
model call per question, which is a minute each on a local CPU model. Run
`make eval-llm` to add them.

**The eval set is `eval/questions.jsonl`, versioned in the repo** (a
deliverable). 20 starter questions written against the sample corpus:
paraphrased rather than copied from the documents, four unanswerable —
including one near-miss (asking for the *notifications* timeout when only the
*payments* timeout is documented) that catches a system substituting a
neighbouring fact for the one asked about.

Replace them with 50-100 questions about **your** corpus. Rules: phrase them
the way a user would, not the way the document does — questions built by
copying sentences out of the corpus measure string matching and flatter every
retriever.
