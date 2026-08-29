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
COPY scripts/ ./scripts/
COPY eval/ ./eval/
COPY streamlit_app.py .
COPY .streamlit/ ./.streamlit/

# Bake the embedding model into the image. Downloading it on first request
# would make the first user wait minutes and re-download on every restart.
#
# The cache path is pinned and made writable by the runtime user: fastembed
# defaults to /tmp, which is created here by ROOT and then unreadable to the
# non-root user the container actually runs as -- the model silently falls
# back to re-downloading, on every start.
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/huggingface
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Non-root: a container escape should not land on root.
RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app /home/app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
