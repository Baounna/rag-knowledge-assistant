# Single image: API + static UI. The database is external (managed Postgres
# with pgvector + pg_search, or the compose service for local work).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

# Bake the embedding model into the image. Downloading it on first request
# would make the first user wait minutes and would re-download on every
# container restart.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Non-root: a container escape should not land on root.
RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app /home/app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
