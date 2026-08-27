# Deployment

The app is one container plus a Postgres that has **pgvector** and
**pg_search**. Everything else — sessions, conversations, feedback, the vector
index, the lexical index — lives in that one database.

## Local

```bash
make db-up          # Postgres on :5432
make setup
make ingest && make index
make serve          # http://localhost:8000
make admin EMAIL=you@company.com   # unlock /admin
```

## Production checklist

Non-negotiable before exposing a URL:

| | why |
|---|---|
| `COOKIE_SECURE=true` | the session cookie must never travel over plain HTTP |
| `ALLOW_SIGNUP=false` | after provisioning accounts, or anyone with the URL gets in |
| Rotate `DATABASE_URL` off `rag:rag` | the compose password is a local-dev default |
| `ANTHROPIC_API_KEY` from a secret store | never baked into the image or committed |
| TLS terminated at the proxy | Fly/Railway/Render do this; a bare VPS does not |
| Set `DAILY_COST_LIMIT_CENTS` deliberately | it is the only thing between one user and your API bill |
| Back up the database | it holds the index *and* every conversation |

`ALLOW_SIGNUP=false` is the important one. The subject asks for "SSO or simple
email/password"; this ships email/password, which means the signup route is the
front door and must be closed once accounts exist.

## Fly.io

```bash
fly launch --no-deploy
fly postgres create --name rag-db          # then attach pgvector/pg_search, or use a managed
                                           # ParadeDB / Neon instance and set DATABASE_URL
fly secrets set ANTHROPIC_API_KEY=sk-ant-... COOKIE_SECURE=true ALLOW_SIGNUP=false
fly deploy
```

## Railway / Render

Point at the `Dockerfile`, add a Postgres with pgvector, set the same
environment variables. Health check: `GET /api/health`.

## VPN-only

The subject allows a VPN-only URL, which is the easier path for an internal
tool: deploy to a private network and skip public exposure entirely. Tailscale
Funnel or a WireGuard subnet both work with no application changes — the app
does not care what is in front of it.

## Indexing in production

Ingestion is a **batch job**, not part of the web process:

```bash
docker compose run --rm app python scripts/ingest.py
docker compose run --rm app python scripts/index.py
```

Run it on a schedule. `--recreate` rebuilds from scratch; the default is
incremental and prunes chunks whose documents were deleted, so a document
removed from the corpus stops being cited.

## Scaling notes, honestly

- **Connection pooling is not implemented.** `Store` and `Accounts` open a
  connection per call. Fine to a few requests/second; add `psycopg_pool`
  before real traffic.
- **`fastembed` runs in-process**, so embedding competes with request handling
  for CPU. At load, move embedding to a worker or switch
  `EMBEDDING_PROVIDER=voyage`.
- **Sessions are database rows**, so the app scales horizontally with no
  sticky sessions required.
- **SSE needs proxy buffering off.** The app sends `X-Accel-Buffering: no`;
  nginx also needs `proxy_buffering off`, or answers arrive all at once at the
  end instead of streaming.
