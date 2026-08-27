"""Embedding providers.

An embedding turns text into a vector, so that "money back for a work trip"
and "expense reimbursement" land near each other despite sharing no words.
That is the whole point of the vector arm of retrieval.

Two things here are easy to get wrong and expensive to discover late:

1. ASYMMETRY. Questions and documents are different kinds of text -- a
   question is short and interrogative, a passage is long and declarative.
   Most retrieval models are trained with distinct prefixes for each side
   ("search_query:" / "search_document:" for nomic, "query:" / "passage:"
   for E5). Embed a question with the document prefix and every score
   degrades quietly -- no error, just worse retrieval. `embed_query` and
   `embed_documents` are separate methods for exactly this reason.

2. DIMENSION LOCK-IN. The vector column has a fixed width, so changing model
   means re-embedding the entire corpus. Chosen deliberately: it fails loudly
   at insert time rather than silently returning nonsense.
"""

from __future__ import annotations

import time
from typing import Protocol, Sequence

import httpx

from .config import Settings, get_settings


class EmbeddingProvider(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _retry(fn, attempts: int = 4, base: float = 0.6):
    """Embedding calls are the highest-volume network calls in the pipeline;
    one flaky response should not lose a whole ingest run."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            # 4xx means the request is wrong, not unlucky. Retrying a 401 four
            # times with backoff just delays a clear error by ~5 seconds and
            # buries the cause. 429 is the exception: it IS worth waiting out.
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                raise RuntimeError(
                    f"embedding request rejected ({exc.response.status_code}): "
                    f"{exc.response.text[:200]}"
                ) from exc
            last = exc
            if i == attempts - 1:
                break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i == attempts - 1:
                break
            time.sleep(base * (2**i))
    raise RuntimeError(f"embedding request failed after {attempts} attempts: {last}")


class FastEmbedEmbeddings:
    """Local embeddings via ONNX -- no server, no torch, no API key.

    Default model is BAAI/bge-small-en-v1.5, one of the open-source models the
    subject names. fastembed exposes `query_embed` / `passage_embed`, which
    apply the model's own asymmetric prefixes -- BGE prepends "Represent this
    sentence for searching relevant passages:" to queries only. Using the
    wrong side degrades every score silently, so never call the raw `embed`.

    Small models buy speed with separation: on this corpus a correct chunk
    scores ~0.54 against a question while an unrelated one scores ~0.49. That
    thin margin is precisely why retrieval does not stop at vector search --
    see the reranker in slice 3.
    """

    def __init__(self, settings: Settings) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy

        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self._model = TextEmbedding(self.model)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = [v.tolist() for v in self._model.passage_embed(list(texts))]
        if vecs and len(vecs[0]) != self.dim:
            raise ValueError(
                f"{self.model} returned {len(vecs[0])} dims, config says {self.dim}. "
                f"Set EMBEDDING_DIM={len(vecs[0])} and re-create the table."
            )
        return vecs

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()


class OllamaEmbeddings:
    """Local embeddings via the Ollama container. No API key required."""

    # Prefixes are model-specific. nomic-embed-text is trained with these two.
    QUERY_PREFIX = "search_query: "
    DOC_PREFIX = "search_document: "

    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self._url = settings.ollama_url.rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    def _embed(self, text: str) -> list[float]:
        def call() -> list[float]:
            r = self._client.post(
                f"{self._url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]

        vec = _retry(call)
        if len(vec) != self.dim:
            raise ValueError(
                f"{self.model} returned {len(vec)} dims, config says {self.dim}. "
                f"Set EMBEDDING_DIM={len(vec)} and re-create the table."
            )
        return vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(self.DOC_PREFIX + t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(self.QUERY_PREFIX + text)


class VoyageEmbeddings:
    """Hosted embeddings from Voyage AI (Anthropic's recommended provider).

    Voyage takes the asymmetry as an `input_type` parameter rather than a text
    prefix -- same idea, cleaner interface.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self._key = settings.voyage_api_key
        self._client = httpx.Client(timeout=120.0)

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        def call() -> list[list[float]]:
            r = self._client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self.model, "input": list(texts), "input_type": input_type},
            )
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]

        return _retry(call)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):          # provider batch limit
            out.extend(self._embed(texts[i : i + 128], "document"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class OpenAIEmbeddings:
    """OpenAI embeddings. Symmetric -- no query/document distinction."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self._key = settings.openai_api_key
        self._client = httpx.Client(timeout=120.0)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        def call() -> list[list[float]]:
            r = self._client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self.model, "input": list(texts)},
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]

        return _retry(call)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            out.extend(self._embed(texts[i : i + 256]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


_PROVIDERS = {
    "fastembed": FastEmbedEmbeddings,
    "ollama": OllamaEmbeddings,
    "voyage": VoyageEmbeddings,
    "openai": OpenAIEmbeddings,
}


def get_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    try:
        cls = _PROVIDERS[settings.embedding_provider]
    except KeyError:
        raise ValueError(
            f"unknown EMBEDDING_PROVIDER={settings.embedding_provider!r}; "
            f"expected one of {sorted(_PROVIDERS)}"
        ) from None
    return cls(settings)  # type: ignore[return-value]
