# Architecture

## System

```mermaid
flowchart TB
    subgraph ingest["Ingestion — batch job, not the web path"]
        C[("corpus/<br/>markdown · pdf · notion")]
        P["parsers<br/>structure preserved:<br/>headings · tables · code"]
        K["chunker<br/>heading > paragraph > sentence > word<br/>+ heading trail + overlap"]
        C --> P --> K
    end

    subgraph db[("Postgres — one datastore")]
        V["chunks.embedding<br/>VECTOR(384) + HNSW cosine"]
        B["chunks.index_text<br/>pg_search BM25 (Tantivy)"]
        U["users · sessions<br/>conversations · messages · feedback"]
    end

    K -->|"embed + upsert<br/>prune deleted docs"| V
    K --> B

    subgraph retrieval["Retrieval"]
        Q["question"]
        RW["rewrite + HyDE<br/>(Haiku 4.5)"]
        VS["vector search"]
        LS["lexical search"]
        RRF["RRF fusion<br/>score = Σ w/(60+rank)"]
        RR["rerank 0-10<br/>(Haiku 4.5)"]
        Q --> RW --> VS & LS
        VS & LS --> RRF --> RR
    end

    V --> VS
    B --> LS

    subgraph gen["Generation"]
        G["answer<br/>(Sonnet 5)<br/>cached context prefix"]
        CV["validate citations<br/>against retrieved ids"]
        RF["refusal gate<br/>confidence + prompt"]
        RR --> G --> CV --> RF
    end

    RF -->|st.write_stream| UI["Streamlit UI<br/>streaming · clickable citations<br/>feedback"]
    UI --> U
    U --> AD["admin dashboard<br/>rating ↔ retrieved chunks"]

    subgraph eval["Evaluation — offline"]
        EQ[("eval/questions.jsonl")]
        M1["deterministic<br/>recall@K · MRR · citation validity<br/>refusal accuracy"]
        M2["judged (Opus 5)<br/>faithfulness · relevance"]
        EQ --> M1 & M2
    end

    RRF -.-> M1
    RF -.-> M2
```

## Request path for one question

Streamlit calls the application modules directly — no HTTP hop, no second
service, no cookie handling. The cost is that there is no REST API for another
client to consume; the FastAPI layer that provided one is in git history and
the modules it called are unchanged.

```
st.chat_input
  ├─ session_state holds the signed-in user
  ├─ check rolling 24h limits                 blocks if over
  ├─ rewrite question              Haiku      1 call   (skipped without a key)
  ├─ lexical + vector search       Postgres   2 queries per rewritten query
  ├─ RRF fuse                      pure code  0 calls
  ├─ rerank candidates             Haiku      1 call   (skipped without a key)
  ├─ st.write_stream(generator)    model      1 call, context prefix cached
  ├─ validate citations            pure code  against the retrieved ids
  ├─ render: valid ids become links, invalid ones stay plain code
  └─ persist message + cost
```

Three model calls per question. Two are Haiku on the critical path; one is
Sonnet for the answer. The split is the cost-awareness decision: reranking is
the high-volume call and does not need a frontier model.

## Why one database

Vectors, lexical index, sessions, conversations and feedback all live in one
Postgres. One connection string, one backup, one place where "what the user
asked" and "what was retrieved" stay consistent. ParadeDB provides pgvector
and pg_search in a single image, so the lexical arm is genuine BM25 (Tantivy)
rather than an approximation — and no second datastore has to be operated.

## Failure behaviour

| failure | behaviour |
|---|---|
| No API key | retrieval runs; rewrite and rerank become no-ops; answers refuse |
| Retrieval returns nothing | refuses before any model call |
| Confidence below `MIN_CONFIDENCE` | refuses before any model call |
| Model fabricates a citation | flagged, rendered unlinked, counted in eval |
| Rerank call fails | falls back to fusion order rather than dropping candidates |
| Document deleted from corpus | pruned at next index; stops being cited |
| User over limit | 429 before any model call |
```
