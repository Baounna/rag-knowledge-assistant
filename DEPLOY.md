# Deployment

Two pieces: the app (Streamlit in a container) and a Postgres that has
**pgvector** and, ideally, **pg_search**. Everything else — sessions,
conversations, feedback, the vector index, the lexical index — lives in that
one database.

## Local

```bash
make up-local     # database + app + local LLM (free)
make ingest && make index
# http://localhost:8502
make admin EMAIL=you@company.com     # unlock the feedback dashboard
```

---

## Fly.io (config is in the repo)

`flyctl` is not installed on this machine yet. Install it, then run these —
**they need your own Fly login, which is why they are written out rather than
already run.**

```bash
brew install flyctl        # or: curl -L https://fly.io/install.sh | sh
fly auth signup            # free tier is enough for a demo
```

### 1. Database first

```bash
fly apps create rag-db
fly volumes create pgdata --size 3 --app rag-db --region cdg
fly secrets set POSTGRES_PASSWORD='pick-a-long-random-one' --app rag-db
fly deploy --config deploy/paradedb.fly.toml --app rag-db
```

### 2. Then the app

```bash
fly apps create rag-knowledge-assistant
fly secrets set \
  DATABASE_URL='postgresql://rag:THE-PASSWORD@rag-db.internal:5432/rag' \
  ANTHROPIC_API_KEY='sk-ant-...' \
  --app rag-knowledge-assistant
fly deploy
```

`rag-db.internal` resolves only on Fly's private network — the database is
never publicly exposed.

### 3. Index the corpus, from your machine against the deployed database

```bash
fly proxy 15432:5432 --app rag-db &          # tunnel the private DB to localhost
DATABASE_URL='postgresql://rag:THE-PASSWORD@localhost:15432/rag' make ingest
DATABASE_URL='postgresql://rag:THE-PASSWORD@localhost:15432/rag' make index
```

Ingestion is a **batch job**, deliberately not part of the web process. Re-run
it whenever the corpus changes; it is incremental and prunes chunks whose
documents were deleted, so a removed document stops being cited.

### 4. Give yourself admin

```bash
fly ssh console --app rag-knowledge-assistant \
  -C "python scripts/admin.py --email you@company.com"
```

---

## Before you share the URL

| | why |
|---|---|
| `ALLOW_SIGNUP=false` | already set in `fly.toml`. Provision accounts, then keep the door shut — signup is the front door |
| `COOKIE_SECURE=true` | already set. The session must never travel over plain HTTP |
| Database password not `rag` | the compose default is for local work only |
| `ANTHROPIC_API_KEY` via `fly secrets` | never in the image, never committed |
| `DAILY_COST_LIMIT_CENTS` set deliberately | the only thing between one user and your API bill |
| Back up the volume | it holds the index *and* every conversation |

## VPN-only instead

The subject allows a VPN-only URL, which is easier for an internal tool:
deploy to a private network and skip public exposure entirely. Tailscale
Funnel or a WireGuard subnet both work with no application changes — the app
does not care what is in front of it.

---

## Choosing the database

| option | BM25 | notes |
|---|---|---|
| **ParadeDB** (`deploy/paradedb.fly.toml`) | ✅ real BM25 via Tantivy | what the code is written for |
| Neon / Supabase / Fly Postgres + pgvector | ⚠️ falls back to Postgres FTS | works, weaker lexical retrieval |
| Anything without pgvector | ❌ | vector search will not start |

The fallback is automatic — `Store.init_schema` tries `CREATE EXTENSION
pg_search`, downgrades to a `tsvector` + GIN index if it fails, and reports
which backend actually ran. That report belongs in your eval write-up: the
lexical arm is measurably weaker on the fallback, and pretending otherwise
would be exactly the kind of unmeasured claim this project is graded against.

## Why not the local model in production

`fly.toml` sets `LLM_PROVIDER=anthropic`. A 7B model needs a GPU machine to
answer in seconds rather than minutes; on shared CPU it is ~100s per answer,
which is unusable for a live demo. Running locally for development and a
hosted model in production is the honest split — and `make eval-full` under
each gives you the measured comparison for the report.

## Scaling notes, honestly

- **No connection pooling.** `Store` and `Accounts` open a connection per
  call. Fine to a few requests/second; add `psycopg_pool` before real traffic.
- **`fastembed` runs in-process**, so embedding competes with request handling
  for CPU. Under load, switch `EMBEDDING_PROVIDER=voyage`.
- **Streamlit holds server-side session state per browser connection**, so it
  does not scale horizontally without sticky sessions. One machine is correct
  for an internal tool; more than one needs `min_machines_running` raised and
  session affinity, or a move back to the API + stateless client split (that
  code is in git history).
- **`auto_stop_machines = "suspend"`, not `"stop"`** — a cold start reloads the
  embedding model and takes ~30 s. Suspend keeps memory resident.
