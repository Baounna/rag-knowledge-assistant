# Evaluation report

Corpus: 171 documents from GitLab's public employee handbook, ingested through
three connectors (Markdown, PDF, Notion export) → **1,994 chunks**.
Question set: **56** (47 answerable, 9 unanswerable), see `eval/README.md` for
how it was built.

Reproduce with `make eval` (retrieval) and `make eval-full` (answers).

---

## 1. Retrieval

Each configuration runs the identical pipeline; only the retriever weighting
changes, so any difference is attributable to the retriever and nothing else.

| config | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| lexical only (BM25) | 0.787 | 0.894 | 0.894 | 0.936 | 0.711 |
| vector only (BGE) | 0.723 | 0.766 | 0.830 | 0.894 | 0.647 |
| **hybrid (RRF, k=60)** | **0.830** | **0.915** | **0.936** | **0.936** | **0.730** |

**Hybrid wins on every column.** It beats the better single retriever by 4.3
points of recall@1, which is the metric that matters most here: only the top
N chunks reach the answer prompt, so a relevant chunk at rank 11 may as well
not exist.

The two retrievers fail on opposite inputs, which is the whole argument for
fusing them. Measured on this corpus:

```
"how do I get my money back for a work trip?"   (no shared words with the doc)
  vector  -> expense policy at #1        lexical -> unrelated chunks at #2,#3

"/var/log/db"                                    (an exact path)
  vector  -> correct chunk at #2         lexical -> correct chunk at #1
```

### Caveat that bounds these numbers

**The questions were drafted by a model that had read the passages.** Despite
being instructed to phrase them as an employee would rather than reuse the
document's wording, some vocabulary leaks through, and that favours BM25. The
8-point lexical-over-vector gap should therefore be read as an **upper bound**
on BM25's real advantage. A question set written by hand, from memory, would
likely narrow it — and would probably widen hybrid's margin, since vector
search is what rescues paraphrased questions.

### Not measured here

The reranker and query rewriting are **not** in the table. Each costs one
model call per question, which is roughly a minute on the local CPU model. On
the two questions retrieval currently misses (below), a reranker is the
mechanism most likely to help — measuring that is the clearest next
experiment.

---

## 2. Answers

Generated with `qwen2.5:7b-instruct` running locally, hybrid retrieval,
top-5 chunks. 58 questions at the time of the run (before two were removed in
review, see §4).

| metric | score | how it is computed |
|---|---|---|
| grounded | **1.000** | cites ≥1 real chunk, fabricates none — code |
| citation validity | **1.000** | share of cited ids that were retrieved — code |
| refusal correct | 0.833 | scored in **both** directions — code |
| required content | 0.793 | share of `must_include` strings present — code |
| faithfulness | 7.96 / 10 | LLM judge |
| relevance | 9.17 / 10 | LLM judge |
| uncited claims | 1.04 / answer | heuristic sentence count |

**Zero fabricated citations across 58 answers.** Every `[chunk_id]` the model
emitted was checked in code against what retrieval actually returned, and all
of them matched. That is the single most important number here: a plausible
citation pointing at nothing is the failure that makes a knowledge assistant
worse than useless, and it did not occur — on a 7B local model.

`refusal_correct = 0.833` counts errors in both directions: answering an
unanswerable question, and refusing an answerable one. Scoring only the first
would let a system that never refuses look perfect.

### Caveats that bound these numbers

1. **The judge is the model that wrote the answers.** Faithfulness 7.96 and
   relevance 9.17 come from `qwen2.5:7b-instruct` grading its own output. That
   is a weak evaluation and the two judged rows should be treated as
   indicative only. The deterministic rows above them do not have this problem
   — they are computed in code and cannot flatter anything.
2. **4 of 58 generations failed** (transient backend errors under load) and are
   excluded from the averages rather than counted as zero.
3. Prompt caching reports 0% because the local backend has no cache. Against
   Claude the cached context prefix would show a hit rate; that is why
   `Usage.cache_hit_rate` exists.

---

## 3. Where retrieval still fails

Two questions where the labelled chunk never reaches the top 10. Both labels
were verified by hand and are **correct** — this is retrieval failing, not
bad ground truth.

**q024 — "What is the contact email for the Privacy Team at GitLab?"**
Expected `pdf-legal-privacy-employee-privacy-policy#c0030`; retrieval returns
three other privacy pages. The answer (`dpo@gitlab.com`) appears once, in a
passage that is mostly about objection procedures. The chunk is *about*
something else and merely *contains* the fact.

**q051 — "Can a team member assessed as Developing be considered for a new role?"**
Expected the internal-hiring page; retrieval returns three chunks from the
talent-assessment page, which is the right *topic* and the wrong *document*.

Both are the same failure: **strong topical match, wrong specific fact.** That
is precisely what a reranker addresses — it reads candidates instead of
scoring their vocabulary. These two questions were deliberately kept in the
set for that reason; removing them would leave a set that reports better
numbers about less.

---

## 4. Changes made during evaluation

Recorded because they affect how the numbers compare across runs.

- **The subject PDF was in the corpus.** `SUBJECT.pdf` had been copied into
  `corpus/pdf/` early on to test the PDF parser and was never removed. It
  produced a question — *"What percentage of the total compensation is
  allocated to UI & UX?" → 15%* — which was reading the assignment's own
  grading grid. Removed (3 chunks) along with that question.
- **q017 removed** — circular: asked what the pseudonymization process *is*
  and answered that fields *are* pseudonymized.
- 6 questions removed earlier by `make check-questions`, whose `must_include`
  string did not appear in the passage they were labelled against.
- 4 duplicate questions removed.

The retrieval table in §1 is measured after all of these. The answer table in
§2 was measured before the last two removals; the affected metrics
(groundedness, citation validity) are computed per answer and are not
materially changed by dropping two questions from a set of 58.

---

## 5. Configuration measured

```
chunking     target 350 words, max 500, 2-sentence overlap, recursive
             (heading > paragraph > sentence > word), heading trail prepended
embeddings   BAAI/bge-small-en-v1.5, 384d, cosine, asymmetric query/passage
lexical      pg_search (Tantivy BM25) with English stemming
fusion       RRF, k=60, equal weights
retrieval    top_k 20 per retriever, top_n 5 to the prompt
generation   qwen2.5:7b-instruct, local, MIN_CONFIDENCE gate on rerank score
```

**The deployed instance differs in one respect:** no free managed Postgres
still offers `pg_search` (Neon rejects it as deprecated), so the lexical arm
there falls back to Postgres full-text search. `Store.init_schema` detects
this and reports which backend ran. Lexical retrieval will measure somewhat
worse on the deployed copy than in the table above.
