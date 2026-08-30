# Corpus

The documents the assistant answers from. **Not committed** — reproduce with:

```bash
make corpus       # GitLab's public handbook, ~170 policy documents
make ingest && make index
```

## Why this corpus

GitLab publishes its entire internal employee handbook openly. That makes it
an unusually honest stand-in for the subject's premise — "internal docs nobody
can find when they need them" — because it is real, it is dense with specific
rules and numbers, and every answer is verifiable against a public URL.

`scripts/fetch_corpus.py` does two things that matter beyond copying files:

- **Skips stubs.** Many handbook pages are a single line pointing at an
  internal page. Those chunks retrieve for their topic and answer nothing,
  which poisons retrieval metrics and any eval set built on top of them.
- **Injects the public URL** into each file's frontmatter. GitLab's own
  frontmatter carries only a title, and without a `url` the citation cannot be
  clicked.

## Adding your own documents

Drop them in and re-run `make ingest && make index`. Nothing in the code
changes:

```
corpus/
  markdown/   *.md    Markdown repo, exported docs
  pdf/        *.pdf   PDF folder
  notion/     */*.md  Notion Markdown export (page ids preserved in filenames)
```

Front-matter the ingester reads, all optional except that `url` is what makes
a citation clickable:

```yaml
---
title: "Expense Policy"
source: "Finance Handbook"     # groups documents for the source filter
url: https://intranet/expenses # makes the citation clickable
author: "Finance Team"
date: 2025-03-14               # enables date filtering
---
```
