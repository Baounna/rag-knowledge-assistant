# Evaluation set

`questions.jsonl` — 58 questions over the indexed corpus (49 answerable,
9 unanswerable).

## How these were produced, and what that means

**Machine-drafted, machine-screened, not yet human-reviewed.** Stated plainly
because it bounds how much the resulting numbers are worth.

1. `make draft-questions` sampled fact-dense prose chunks stratified across
   sources and drafted one question per passage, instructed to phrase them as
   an employee would rather than reusing the document's wording.
2. `make check-questions FIX=1` removed 6 questions whose `must_include`
   string did not appear in the passage they were labelled against — the model
   had read a neighbouring chunk or invented the fact.
3. Duplicates were removed (the model repeated 4 questions across batches).
4. Unanswerable questions were over-generated 3x and the weakest-retrieving
   kept, then checked against a threshold calibrated from the corpus itself.

## The caveat that matters

**The questions were drafted by a model that had read the passages.** Despite
being instructed to avoid the document's wording, vocabulary leaks through,
and that favours lexical retrieval. The measured 8-point gap between BM25 and
vector search is therefore an upper bound on BM25's real advantage; a set
written by hand, from memory, would likely narrow it.

## Two questions are deliberately hard

`q024` and `q051` are labelled against chunks neither retriever reaches in its
top 10. They were kept on purpose. Removing questions the system currently
fails would leave a set that reports excellent numbers about nothing — and
they are the only questions that can demonstrate a reranker or better chunking
actually helping.

## To improve this set

Run `make review-questions`, which prints each question beside the passage it
is labelled against, and correct the labels that are wrong. Questions written
by hand from your own knowledge of the corpus are worth more than any number
of generated ones.
