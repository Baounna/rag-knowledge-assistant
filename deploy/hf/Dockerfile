# Hugging Face Spaces variant of the root Dockerfile.
#
# Two differences, both forced by the platform:
#   * Spaces routes to port 7860, not 8501.
#   * Spaces runs the container as uid 1000, so every path the app writes to
#     must be owned by it -- including the embedding-model cache, which the
#     root image places under /app/.cache.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
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

# Bake the embedding model so the first visitor does not wait for a download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Spaces runs as uid 1000; the cache written above by root must be readable
# and writable by it, or the model silently re-downloads on every start.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
    CMD curl -fsS http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py"]
