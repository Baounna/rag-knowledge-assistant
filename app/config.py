"""Configuration, read once from the environment (and .env if present).

No pydantic-settings dependency: a 40-line loader keeps the dependency list
short and makes it obvious where every setting comes from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from .env. Real environment always wins."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    # -- models --------------------------------------------------------
    anthropic_api_key: str
    model_answer: str            # final answer -- quality matters most
    model_rerank: str            # one call per candidate: cheap and fast
    model_rewrite: str           # one short call per query: cheap and fast
    model_judge: str             # offline eval only: strongest available

    # -- embeddings ----------------------------------------------------
    embedding_provider: str      # ollama | voyage | openai
    embedding_model: str
    embedding_dim: int
    ollama_url: str
    voyage_api_key: str
    openai_api_key: str

    # -- storage -------------------------------------------------------
    database_url: str

    # -- retrieval -----------------------------------------------------
    retrieval_top_k: int         # candidates pulled from EACH index
    rerank_top_n: int            # chunks that actually reach the prompt

    # -- chunking ------------------------------------------------------
    chunk_target_words: int
    chunk_max_words: int
    chunk_overlap_sentences: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    provider = os.environ.get("EMBEDDING_PROVIDER", "fastembed").lower()

    # Dimensions are a property of the model, not a free choice: the vector
    # column is declared with a fixed width, so switching model means
    # re-embedding the whole corpus. Getting this wrong fails loudly at
    # insert time, which is the right time for it to fail.
    defaults = {
        "fastembed": ("BAAI/bge-small-en-v1.5", 384),
        "ollama": ("nomic-embed-text", 768),
        "voyage": ("voyage-3", 1024),
        "openai": ("text-embedding-3-small", 1536),
    }
    default_model, default_dim = defaults.get(provider, defaults["fastembed"])

    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        model_answer=os.environ.get("MODEL_ANSWER", "claude-sonnet-5"),
        model_rerank=os.environ.get("MODEL_RERANK", "claude-haiku-4-5"),
        model_rewrite=os.environ.get("MODEL_REWRITE", "claude-haiku-4-5"),
        model_judge=os.environ.get("MODEL_JUDGE", "claude-opus-5"),
        embedding_provider=provider,
        embedding_model=os.environ.get("EMBEDDING_MODEL", default_model),
        embedding_dim=_int("EMBEDDING_DIM", default_dim),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        voyage_api_key=os.environ.get("VOYAGE_API_KEY", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"
        ),
        retrieval_top_k=_int("RETRIEVAL_TOP_K", 20),
        rerank_top_n=_int("RERANK_TOP_N", 5),
        chunk_target_words=_int("CHUNK_TARGET_WORDS", 350),
        chunk_max_words=_int("CHUNK_MAX_WORDS", 500),
        chunk_overlap_sentences=_int("CHUNK_OVERLAP_SENTENCES", 2),
    )
